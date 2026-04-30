"""Dataset PyTorch pour Entity Typing.

Construit un prompt par entité, du type :
    "{sentence} {entity_text} is a [MASK]."

L'embedding du token [MASK] sera l'embedding de l'entité, raffiné ensuite
par contrastive learning (Triplet Loss) et clusterisé par K-means.

Simplification vs original : pas de réduction adaptative de la phrase si le
prompt est trop long (on tronque via le tokenizer, et on lève si [MASK] est
perdu dans la troncature).
"""
from typing import TypedDict

import torch
from torch.utils.data import Dataset as TorchDataset
from transformers import AutoTokenizer

from ..schema import Dataset


class EtInstance(TypedDict):
    """Instance retournée par __getitem__."""
    document_idx: int
    sentence_idx: int
    start_word_idx: int
    end_word_idx: int
    input_ids: torch.Tensor      # [max_len]
    attention_mask: torch.Tensor # [max_len]
    mask_index: int              # position du token [MASK] dans la séquence
    entity_type_label: int       # id numérique du type (mappé via entity_type_to_id)


class EntityTypingDataset(TorchDataset):
    """1 entité = 1 instance, avec un prompt contenant un [MASK]."""

    def __init__(
        self,
        dataset: Dataset,
        tokenizer_name: str,
        max_len: int = 256,
        template: str = "{sentence} {entity} is a [MASK].",
    ):
        super().__init__()
        self.dataset = dataset
        self.max_len = max_len
        self.template = template

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.mask_token_id = self.tokenizer.mask_token_id
        if self.mask_token_id is None:
            raise ValueError(
                f"Le tokenizer {tokenizer_name!r} n'a pas de token [MASK]. "
                "Entity Typing par prompt nécessite un PLM de type MLM "
                "(BERT, DeBERTa, RoBERTa...)."
            )

        # Mapping type d'entité <-> id numérique (pour la Triplet Loss)
        self.entity_type_to_id: dict[str | int, int] = {
            t: i for i, t in enumerate(sorted(dataset.metadata.entity_types))
        }
        self.id_to_entity_type: dict[int, str | int] = {
            i: t for t, i in self.entity_type_to_id.items()
        }

        # Aplatir : 1 entité = 1 row
        self.rows: list[dict] = []
        for doc_idx, document in enumerate(dataset.documents):
            for entity in document.entities:
                sentence = document.sentences[entity.sentence_idx]
                entity_text = ' '.join(sentence[entity.start_word_idx:entity.end_word_idx])
                prompt = self.template.format(
                    sentence=' '.join(sentence),
                    entity=entity_text,
                )
                self.rows.append({
                    'document_idx': doc_idx,
                    'sentence_idx': entity.sentence_idx,
                    'start_word_idx': entity.start_word_idx,
                    'end_word_idx': entity.end_word_idx,
                    'entity_type': entity.type,
                    'prompt': prompt,
                })

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> EtInstance:
        row = self.rows[index]

        encoded = self.tokenizer(
            row['prompt'],
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        input_ids = encoded['input_ids'].view(-1)        # [max_len]
        attention_mask = encoded['attention_mask'].view(-1)

        # Position du token [MASK]
        mask_positions = (input_ids == self.mask_token_id).nonzero().flatten()
        if len(mask_positions) == 0:
            raise ValueError(
                f"Pas de [MASK] trouvé dans la séquence (sans doute tronquée). "
                f"Augmente max_len. Prompt : {row['prompt'][:120]}..."
            )
        mask_index = int(mask_positions[0].item())

        return {
            'document_idx': row['document_idx'],
            'sentence_idx': row['sentence_idx'],
            'start_word_idx': row['start_word_idx'],
            'end_word_idx': row['end_word_idx'],
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'mask_index': mask_index,
            'entity_type_label': self.entity_type_to_id[row['entity_type']],
        }
