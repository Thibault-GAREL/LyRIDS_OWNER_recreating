"""Trainer pour Entity Typing.

Le cœur original d'OWNER : pour chaque entité, on calcule un embedding via un
prompt à [MASK], on raffine ces embeddings avec une Triplet Margin Loss en
mode batch, puis on clusterise via K-means + sélection auto de k via BIC.

Évaluation : AMI et ARI entre les vrais types et les clusters prédits.
"""
import os

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from tqdm.auto import tqdm

from ..data.schema import Dataset, Entity, MiniDocument
from ..data.serialization import parse_owner_dataset
from ..data.datasets.entity_typing import EntityTypingDataset
from ..models.entity_typing import EntityEncodingModel, AutoKmeans
from ..evaluation.entity_typing import evaluate_entity_typing
from ..utils.mlflow_helpers import (
    log_artifacts_safe,
    log_metrics_safe,
    log_params_safe,
    mlflow_run,
)
from .base import BaseTrainer


class BatchTripletMarginLoss(nn.Module):
    """Triplet margin loss qui extrait tous les triplets valides du batch.

    Un triplet (i, j, k) est valide si :
        - types[i] == types[j]   (ancre et positif du même type)
        - types[i] != types[k]   (ancre et négatif de types différents)

    Loss par triplet : max(0, d(i,j) - d(i,k) + margin)
    """

    def __init__(self, margin: float = 1.0):
        super().__init__()
        self.margin = margin

    def forward(
        self,
        entity_types: torch.Tensor,    # [B]      - id du type
        embeddings: torch.Tensor,      # [B, H]
    ) -> torch.Tensor:                 # -> scalaire
        # type_eq[i, j] = True ssi types[i] == types[j]
        type_eq = entity_types.unsqueeze(1) == entity_types.unsqueeze(0)         # [B, B]

        # valid_triplets[i, j, k] = type_eq[i, j] AND NOT type_eq[i, k]
        valid_triplets = type_eq.unsqueeze(2) & ~type_eq.unsqueeze(1)            # [B, B, B]

        # Matrice de distances euclidiennes pairwise
        distances = torch.cdist(embeddings, embeddings, p=2)                     # [B, B]

        # Pour chaque triplet (i, j, k) : d(i,j) - d(i,k) + margin
        triplet_raw = distances.unsqueeze(2) - distances.unsqueeze(1) + self.margin   # [B, B, B]
        triplet_loss = torch.clamp(triplet_raw, min=0.0)

        # Moyenne sur les triplets valides uniquement
        n_valid = valid_triplets.sum().clamp(min=1)
        return (triplet_loss * valid_triplets.float()).sum() / n_valid


class EntityTypingTrainer(BaseTrainer):
    """Entraîne EntityEncodingModel avec une Triplet Loss en mode batch."""

    def __init__(
        self,
        train_path: str,
        test_path: str,
        plm_name: str = 'bert-base-uncased',
        max_len: int = 256,
        batch_size: int = 16,
        num_epochs: int = 4,
        learning_rate: float = 2e-5,
        margin: float = 1.0,
        # Bornes pour AutoKmeans à l'évaluation
        k_min: int = 2,
        k_max: int = 30,
        k_step: int = 2,
        seed: int = 42,
        template: str = "{entity} is a [MASK]. Context: {sentence}",
        device: str | None = None,
    ):
        self.train_path = train_path
        self.test_path = test_path
        self.plm_name = plm_name
        self.max_len = max_len
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.margin = margin
        self.k_min = k_min
        self.k_max = k_max
        self.k_step = k_step
        self.seed = seed
        self.template = template
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        self.model = EntityEncodingModel(plm_name).to(self.device)
        self.train_dataset: EntityTypingDataset | None = None
        self.test_dataset: EntityTypingDataset | None = None

    def load_data(self, training: bool = True) -> None:
        if training:
            self.train_dataset = EntityTypingDataset(
                parse_owner_dataset(self.train_path),
                tokenizer_name=self.plm_name,
                max_len=self.max_len,
                template=self.template,
            )
        self.test_dataset = EntityTypingDataset(
            parse_owner_dataset(self.test_path),
            tokenizer_name=self.plm_name,
            max_len=self.max_len,
            template=self.template,
        )

    def train(self, checkpoint_dir: str | None = None) -> None:
        """Entraîne ET. Si `checkpoint_dir` est fourni, sauve le meilleur epoch
        (au sens de l'AMI test) dans ce dossier et le recharge à la fin de l'entraînement.
        """
        with mlflow_run('entity_typing'):
            log_params_safe({
                'et_plm_name': self.plm_name,
                'et_max_len': self.max_len,
                'et_batch_size': self.batch_size,
                'et_num_epochs': self.num_epochs,
                'et_learning_rate': self.learning_rate,
                'et_margin': self.margin,
                'et_k_min': self.k_min,
                'et_k_max': self.k_max,
                'et_k_step': self.k_step,
                'et_seed': self.seed,
                'et_template': self.template,
                'et_device': self.device,
                'et_best_checkpoint_dir': checkpoint_dir or '',
            })

            train_loader = DataLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=0,
            )
            n_steps = len(train_loader) * self.num_epochs

            optimizer = optim.AdamW(self.model.parameters(), lr=self.learning_rate)
            scheduler = get_linear_schedule_with_warmup(
                optimizer, num_warmup_steps=0, num_training_steps=n_steps,
            )
            loss_fn = BatchTripletMarginLoss(margin=self.margin)

            # Eval avant entraînement (baseline)
            print('\n--- Avant entraînement ---')
            baseline = self.evaluate()
            print(f'  AMI={baseline["ami"]:.4f} ARI={baseline["ari"]:.4f} k={baseline["k"]}')
            log_metrics_safe({
                'et_baseline_ami': baseline['ami'],
                'et_baseline_ari': baseline['ari'],
                'et_baseline_k': baseline['k'],
            }, step=0)

            best_ami = -1.0
            best_epoch = 0
            global_step = 0
            for epoch in range(1, self.num_epochs + 1):
                print(f'\n=== Epoch {epoch}/{self.num_epochs} ===')
                self.model.train()
                epoch_loss_sum = 0.0
                n_batches = 0
                for batch in tqdm(train_loader, desc='train'):
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    mask_index = batch['mask_index'].to(self.device)
                    type_labels = batch['entity_type_label'].to(self.device)

                    embeddings = self.model(input_ids, attention_mask, mask_index)
                    loss = loss_fn(type_labels, embeddings)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    scheduler.step()

                    epoch_loss_sum += loss.item()
                    n_batches += 1
                    global_step += 1
                    log_metrics_safe({'et_train_batch_loss': loss.item()}, step=global_step)

                metrics = self.evaluate()
                print(
                    f'  AMI={metrics["ami"]:.4f} ARI={metrics["ari"]:.4f} '
                    f'k_trouvé={metrics["k"]} (vrai k={metrics["true_k"]})'
                )
                log_metrics_safe({
                    'et_train_epoch_loss': epoch_loss_sum / max(n_batches, 1),
                    'et_test_ami': metrics['ami'],
                    'et_test_ari': metrics['ari'],
                    'et_test_k_found': metrics['k'],
                    'et_test_k_true': metrics['true_k'],
                }, step=epoch)

                if checkpoint_dir is not None and metrics['ami'] > best_ami:
                    best_ami = metrics['ami']
                    best_epoch = epoch
                    os.makedirs(checkpoint_dir, exist_ok=True)
                    torch.save(
                        self.model.state_dict(),
                        f'{checkpoint_dir}/entity_typing.pt',
                    )
                    print(f'  ★ best so far (epoch {best_epoch}, AMI={best_ami:.4f}) → {checkpoint_dir}/entity_typing.pt')
                    log_metrics_safe({
                        'et_best_ami': best_ami,
                        'et_best_epoch': best_epoch,
                    }, step=epoch)

            # Reload des meilleurs poids pour que la suite (eval end-to-end) en bénéficie
            if checkpoint_dir is not None and best_epoch > 0:
                print(f'\n→ Reload du meilleur ET : epoch {best_epoch}, AMI={best_ami:.4f}')
                self.load_model(checkpoint_dir)

    def evaluate(self) -> dict[str, float]:
        """Compute embeddings sur le test set, fit AutoKmeans, mesure AMI/ARI."""
        self.model.eval()
        loader = DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=0)

        all_embeddings = []
        all_type_ids = []
        with torch.no_grad():
            for batch in loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                mask_index = batch['mask_index'].to(self.device)

                embeddings = self.model(input_ids, attention_mask, mask_index)
                all_embeddings.append(embeddings.cpu())
                all_type_ids.append(batch['entity_type_label'])

        all_embeddings = torch.cat(all_embeddings, dim=0)            # [N, H]
        all_type_ids = torch.cat(all_type_ids, dim=0)                # [N]

        # Borner k_max au nombre d'entités (KMeans plante si k > n)
        n_entities = all_embeddings.shape[0]
        k_max = min(self.k_max, n_entities - 1)
        k_min = min(self.k_min, k_max)

        auto_km = AutoKmeans(
            k_min=k_min, k_max=k_max, k_step=self.k_step, seed=self.seed,
        )
        auto_km.fit(all_embeddings)
        pred_clusters = auto_km.predict(all_embeddings)              # [N]

        metrics = evaluate_entity_typing(
            true_type_ids=all_type_ids.tolist(),
            pred_cluster_ids=pred_clusters.tolist(),
        )
        metrics['k'] = auto_km.k
        metrics['true_k'] = len(set(all_type_ids.tolist()))
        return metrics

    def predict_dataset(self, dataset: Dataset) -> list[MiniDocument]:
        """Type les entités d'un dataset arbitraire via clustering.

        Calcule les embeddings de chaque entité, fit AutoKmeans, attribue à
        chaque entité son cluster comme nouveau type.
        """
        # Si le dataset n'a aucune entité, renvoie des MiniDocuments vides
        if not any(len(d.entities) > 0 for d in dataset.documents):
            return [MiniDocument(id=d.id) for d in dataset.documents]

        et_dataset = EntityTypingDataset(
            dataset,
            tokenizer_name=self.plm_name,
            max_len=self.max_len,
            template=self.template,
        )
        loader = DataLoader(et_dataset, batch_size=self.batch_size, num_workers=0)

        self.model.eval()
        all_embeddings: list[torch.Tensor] = []
        metadata: list[dict] = []

        with torch.no_grad():
            for batch in loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                mask_index = batch['mask_index'].to(self.device)

                embeddings = self.model(input_ids, attention_mask, mask_index)
                all_embeddings.append(embeddings.cpu())

                for i in range(len(batch['document_idx'])):
                    metadata.append({
                        'document_idx': int(batch['document_idx'][i].item()),
                        'sentence_idx': int(batch['sentence_idx'][i].item()),
                        'start_word_idx': int(batch['start_word_idx'][i].item()),
                        'end_word_idx': int(batch['end_word_idx'][i].item()),
                    })

        all_embeddings = torch.cat(all_embeddings, dim=0)

        # Cluster
        n = all_embeddings.shape[0]
        k_max = min(self.k_max, max(n - 1, self.k_min))
        k_min = min(self.k_min, k_max)
        auto_km = AutoKmeans(k_min, k_max, self.k_step, seed=self.seed)
        auto_km.fit(all_embeddings)
        clusters = auto_km.predict(all_embeddings).tolist()

        # Construire les MiniDocuments
        preds_by_doc: dict[int, MiniDocument] = {}
        for meta, cluster in zip(metadata, clusters):
            doc_idx = meta['document_idx']
            if doc_idx not in preds_by_doc:
                preds_by_doc[doc_idx] = MiniDocument(id=dataset.documents[doc_idx].id)
            preds_by_doc[doc_idx].entities.append(Entity(
                type=int(cluster),
                sentence_idx=meta['sentence_idx'],
                start_word_idx=meta['start_word_idx'],
                end_word_idx=meta['end_word_idx'],
            ))

        return [
            preds_by_doc.get(i, MiniDocument(id=dataset.documents[i].id))
            for i in range(len(dataset.documents))
        ]

    def save_model(self, folder: str) -> None:
        os.makedirs(folder, exist_ok=True)
        torch.save(self.model.state_dict(), f'{folder}/entity_typing.pt')
        log_artifacts_safe(folder)

    def load_model(self, folder: str) -> None:
        self.model.load_state_dict(torch.load(f'{folder}/entity_typing.pt'))
        self.model.eval()
