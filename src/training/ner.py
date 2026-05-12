"""Trainer NER end-to-end.

Orchestre Mention Detection + Entity Typing :
    1. Entraîne MD et ET séparément (sur ground truth).
    2. À l'évaluation, MD prédit les positions, ET les types par cluster,
       puis on compare avec la ground truth via AMI / ARI.
"""
from ..evaluation.base import (
    align_entities_by_position,
    convert_document_to_entities,
    merge_dataset_with_predictions,
)
from ..evaluation.entity_typing import evaluate_entity_typing
from ..utils.mlflow_helpers import log_metrics_safe
from .base import BaseTrainer
from .mention_detection import MentionDetectionTrainer
from .entity_typing import EntityTypingTrainer


class NerTrainer(BaseTrainer):
    """Pipeline NER : MD → ET, évalué de bout en bout."""

    def __init__(
        self,
        mention_detection: MentionDetectionTrainer,
        entity_typing: EntityTypingTrainer,
    ):
        self.mention_detection = mention_detection
        self.entity_typing = entity_typing

    def load_data(self, training: bool = True) -> None:
        self.mention_detection.load_data(training)
        self.entity_typing.load_data(training)

    def train(self, checkpoint_dir: str | None = None) -> None:
        """Entraîne MD puis ET. Si `checkpoint_dir` est fourni, chaque sous-trainer
        sauve son meilleur epoch en cours de route (best F1 pour MD, best AMI pour ET)
        et recharge ces poids à la fin — l'éval end-to-end utilise donc le meilleur
        modèle, pas le dernier.
        """
        print('\n========== Training Mention Detection ==========')
        self.mention_detection.train(checkpoint_dir=checkpoint_dir)
        print('\n========== Training Entity Typing ==========')
        self.entity_typing.train(checkpoint_dir=checkpoint_dir)

    def evaluate(self) -> dict[str, float]:
        """Évaluation end-to-end : MD prédit positions, ET cluster, AMI/ARI sur l'ensemble."""
        print('\n========== Évaluation NER end-to-end ==========')

        # 1. Ground truth (avec les vrais types)
        test_dataset = self.mention_detection.test_dataset.dataset
        truth = [convert_document_to_entities(d) for d in test_dataset.documents]

        # 2. MD prédit les positions des entités (type = "entity" placeholder)
        pred_md = self.mention_detection.predict_dataset(test_dataset)
        n_pred_md = sum(len(p.entities) for p in pred_md)
        n_truth = sum(len(t.entities) for t in truth)
        print(f'  MD a prédit {n_pred_md} entités (vérité : {n_truth})')

        # 3. On construit un dataset où les "entités" sont celles prédites par MD
        merged = merge_dataset_with_predictions(test_dataset, pred_md)

        # 4. ET type ces entités prédites via clustering
        pred_typed = self.entity_typing.predict_dataset(merged)

        # 5. Alignement par position et calcul AMI/ARI
        true_labels, pred_labels = align_entities_by_position(truth, pred_typed)
        if len(true_labels) == 0:
            print('  Aucune entité ni en truth ni en pred, AMI/ARI indéfinis.')
            return {'ami': 0.0, 'ari': 0.0, 'n_truth': 0, 'n_pred': 0}

        metrics = evaluate_entity_typing(
            true_type_ids=true_labels,
            pred_cluster_ids=pred_labels,
        )
        metrics['n_truth'] = n_truth
        metrics['n_pred'] = n_pred_md
        log_metrics_safe({
            'ner_end_to_end_ami': metrics['ami'],
            'ner_end_to_end_ari': metrics['ari'],
            'ner_n_truth_entities': n_truth,
            'ner_n_pred_entities': n_pred_md,
        })
        return metrics

    def save_model(self, folder: str) -> None:
        self.mention_detection.save_model(folder)
        self.entity_typing.save_model(folder)

    def load_model(self, folder: str) -> None:
        self.mention_detection.load_model(folder)
        self.entity_typing.load_model(folder)
