"""Trainer pour Mention Detection.

Boucle classique : tokenize → forward → loss CE (avec ignore_index) → backward
→ optim step → scheduler step. Évaluation à la fin de chaque epoch.
"""
import os

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from tqdm.auto import tqdm

from ..data.schema import Dataset, MiniDocument
from ..data.serialization import parse_owner_dataset
from ..data.datasets.mention_detection import MentionDetectionDataset
from ..models.mention_detection import MdBioModel
from ..utils.pytorch import IGNORE_VALUE
from ..utils.mlflow_helpers import (
    log_artifacts_safe,
    log_metrics_safe,
    log_params_safe,
    mlflow_run,
)
from ..evaluation.mention_detection import (
    convert_bio_to_entities,
    convert_document_to_entities,
    evaluate_mention_detection,
)
from .base import BaseTrainer


class MentionDetectionTrainer(BaseTrainer):
    """Entraîne et évalue MdBioModel sur des datasets au format OWNER."""

    def __init__(
        self,
        train_path: str,
        test_path: str,
        plm_name: str = 'microsoft/deberta-v3-base',
        max_len: int = 256,
        batch_size: int = 8,
        num_epochs: int = 4,
        learning_rate: float = 2e-5,
        device: str | None = None,
    ):
        self.train_path = train_path
        self.test_path = test_path
        self.plm_name = plm_name
        self.max_len = max_len
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        self.model = MdBioModel(plm_name).to(self.device)
        self.train_dataset: MentionDetectionDataset | None = None
        self.test_dataset: MentionDetectionDataset | None = None

    def load_data(self, training: bool = True) -> None:
        if training:
            self.train_dataset = MentionDetectionDataset(
                parse_owner_dataset(self.train_path),
                tokenizer_name=self.plm_name,
                max_len=self.max_len,
            )
        self.test_dataset = MentionDetectionDataset(
            parse_owner_dataset(self.test_path),
            tokenizer_name=self.plm_name,
            max_len=self.max_len,
        )

    def train(self) -> None:
        with mlflow_run('mention_detection'):
            log_params_safe({
                'md_plm_name': self.plm_name,
                'md_max_len': self.max_len,
                'md_batch_size': self.batch_size,
                'md_num_epochs': self.num_epochs,
                'md_learning_rate': self.learning_rate,
                'md_device': self.device,
            })

            train_loader = DataLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=0,   # 0 = synchrone (Windows-friendly)
            )
            n_steps = len(train_loader) * self.num_epochs

            optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
            scheduler = get_linear_schedule_with_warmup(
                optimizer, num_warmup_steps=0, num_training_steps=n_steps,
            )
            # ignore_index=IGNORE_VALUE : la loss n'inclut PAS les positions à -100
            loss_fn = nn.CrossEntropyLoss(ignore_index=IGNORE_VALUE)

            global_step = 0
            for epoch in range(1, self.num_epochs + 1):
                print(f'\n=== Epoch {epoch}/{self.num_epochs} ===')
                self.model.train()
                epoch_loss_sum = 0.0
                n_batches = 0
                for batch in tqdm(train_loader, desc='train'):
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    labels = batch['labels'].to(self.device)

                    logits = self.model(input_ids, attention_mask)        # [B, T, 3]
                    # CrossEntropyLoss attend [B, C, T] et target [B, T]
                    loss = loss_fn(logits.permute(0, 2, 1), labels)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    scheduler.step()

                    epoch_loss_sum += loss.item()
                    n_batches += 1
                    global_step += 1
                    log_metrics_safe({'md_train_batch_loss': loss.item()}, step=global_step)

                metrics = self.evaluate()
                print(
                    f'  test → F1={metrics["f1"]:.4f} '
                    f'P={metrics["precision"]:.4f} R={metrics["recall"]:.4f} '
                    f'(TP={metrics["tp"]} FP={metrics["fp"]} FN={metrics["fn"]})'
                )
                log_metrics_safe({
                    'md_train_epoch_loss': epoch_loss_sum / max(n_batches, 1),
                    'md_test_f1': metrics['f1'],
                    'md_test_precision': metrics['precision'],
                    'md_test_recall': metrics['recall'],
                    'md_test_tp': metrics['tp'],
                    'md_test_fp': metrics['fp'],
                    'md_test_fn': metrics['fn'],
                }, step=epoch)

    def evaluate(self) -> dict[str, float]:
        self.model.eval()
        loader = DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=0)

        # On accumule par document : 1 doc peut couvrir plusieurs phrases
        truths_by_doc: dict[int, MiniDocument] = {}
        preds_by_doc: dict[int, MiniDocument] = {}
        documents = self.test_dataset.dataset.documents

        with torch.no_grad():
            for batch in loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                ignored = batch['ignored']           # [B, T] bool
                word_ids = batch['word_ids']         # [B, T] long
                doc_idxs = batch['document_idx'].tolist()
                sent_idxs = batch['sentence_idx'].tolist()

                logits = self.model(input_ids, attention_mask)
                pred_labels = logits.argmax(dim=2).cpu()       # [B, T] in {0,1,2}

                # Forcer à 0 (O) tous les sub-tokens ignorés (ils ne doivent pas
                # contribuer aux entités)
                pred_labels = torch.where(ignored, torch.zeros_like(pred_labels), pred_labels)

                for i, doc_idx in enumerate(doc_idxs):
                    sent_idx = sent_idxs[i]
                    doc_id = documents[doc_idx].id

                    if doc_idx not in truths_by_doc:
                        truths_by_doc[doc_idx] = convert_document_to_entities(documents[doc_idx])

                    pred_mini = convert_bio_to_entities(
                        pred_labels[i],
                        word_ids[i],
                        sentence_idx=sent_idx,
                        document_id=doc_id,
                    )
                    if doc_idx not in preds_by_doc:
                        preds_by_doc[doc_idx] = pred_mini
                    else:
                        preds_by_doc[doc_idx].entities.extend(pred_mini.entities)

        truths = [truths_by_doc[i] for i in sorted(truths_by_doc)]
        preds = [preds_by_doc[i] for i in sorted(preds_by_doc)]

        return evaluate_mention_detection(truths, preds)

    def predict_dataset(self, dataset: Dataset) -> list[MiniDocument]:
        """Prédit les entités d'un dataset arbitraire (pour Phase 5 NER).

        Renvoie 1 MiniDocument par document du dataset, dans le MÊME ordre.
        """
        md_dataset = MentionDetectionDataset(
            dataset, tokenizer_name=self.plm_name, max_len=self.max_len,
        )
        loader = DataLoader(md_dataset, batch_size=self.batch_size, num_workers=0)

        self.model.eval()
        preds_by_doc: dict[int, MiniDocument] = {}

        with torch.no_grad():
            for batch in loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                ignored = batch['ignored']
                word_ids = batch['word_ids']
                doc_idxs = batch['document_idx'].tolist()
                sent_idxs = batch['sentence_idx'].tolist()

                logits = self.model(input_ids, attention_mask)
                pred_labels = logits.argmax(dim=2).cpu()
                pred_labels = torch.where(ignored, torch.zeros_like(pred_labels), pred_labels)

                for i, doc_idx in enumerate(doc_idxs):
                    pred_mini = convert_bio_to_entities(
                        pred_labels[i],
                        word_ids[i],
                        sentence_idx=sent_idxs[i],
                        document_id=dataset.documents[doc_idx].id,
                    )
                    if doc_idx not in preds_by_doc:
                        preds_by_doc[doc_idx] = pred_mini
                    else:
                        preds_by_doc[doc_idx].entities.extend(pred_mini.entities)

        # Garantir 1 MiniDocument par document (même vide), dans l'ordre
        return [
            preds_by_doc.get(i, MiniDocument(id=dataset.documents[i].id))
            for i in range(len(dataset.documents))
        ]

    def save_model(self, folder: str) -> None:
        os.makedirs(folder, exist_ok=True)
        torch.save(self.model.state_dict(), f'{folder}/mention_detection.pt')
        log_artifacts_safe(folder)

    def load_model(self, folder: str) -> None:
        self.model.load_state_dict(torch.load(f'{folder}/mention_detection.pt'))
        self.model.eval()
