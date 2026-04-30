"""Dataset PyTorch pour Mention Detection (BIO sequence tagging).

Une instance = une phrase tokenisée + ses labels BIO au niveau token.
Si la phrase dépasse max_len après tokenisation, elle est tronquée
(simplification : pas de window splitting comme dans l'OWNER original).
"""
from typing import TypedDict

import torch
from torch.utils.data import Dataset as TorchDataset
from transformers import AutoTokenizer

from ..schema import Dataset
from ...utils.pytorch import IGNORE_VALUE


class MdInstance(TypedDict):
    """Une instance retournée par __getitem__."""
    document_idx: int
    sentence_idx: int
    word_ids: torch.Tensor       # [max_len] : index du mot d'origine pour chaque sub-token
    input_ids: torch.Tensor      # [max_len]
    attention_mask: torch.Tensor # [max_len]
    ignored: torch.Tensor        # [max_len] : True pour les sub-tokens à ignorer (sub-mots, [CLS], [SEP], [PAD])
    labels: torch.Tensor         # [max_len] : label BIO par sub-token, IGNORE_VALUE sinon


class MentionDetectionDataset(TorchDataset):
    """Dataset pour BIO sequence labeling. 1 phrase = 1 instance."""

    # 3 classes seulement (sans le type) : on cherche juste OÙ sont les entités
    BIO_TO_ID = {'o': 0, 'b': 1, 'i': 2}

    def __init__(self, dataset: Dataset, tokenizer_name: str, max_len: int):
        super().__init__()
        self.dataset = dataset
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_len = max_len

        # On aplatit : 1 phrase = 1 ligne. On garde document_idx pour reconstituer les docs après.
        self.rows: list[dict] = []
        for doc_idx, document in enumerate(dataset.documents):
            for sent_idx, sentence in enumerate(document.sentences):
                # Garder uniquement les entités de cette phrase
                entities = [e for e in document.entities if e.sentence_idx == sent_idx]
                self.rows.append({
                    'document_idx': doc_idx,
                    'sentence_idx': sent_idx,
                    'sentence': sentence,
                    'entities': entities,
                })

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> MdInstance:
        row = self.rows[index]
        words: list[str] = row['sentence']
        entities = row['entities']

        # === 1) Tokenisation ===
        # is_split_into_words=True : on lui donne déjà une liste de mots
        tok = self.tokenizer(
            words,
            is_split_into_words=True,
            add_special_tokens=True,
            padding='max_length',
            max_length=self.max_len,
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        # === 2) Labels au niveau MOT (BIO sans le type) ===
        word_labels = torch.full((len(words),), self.BIO_TO_ID['o'], dtype=torch.long)
        for entity in entities:
            for i in range(entity.start_word_idx, entity.end_word_idx):
                if i == entity.start_word_idx:
                    word_labels[i] = self.BIO_TO_ID['b']
                else:
                    word_labels[i] = self.BIO_TO_ID['i']

        # === 3) Alignement labels mot → labels sub-token ===
        # word_ids[i] = index du mot d'origine pour le i-ème sub-token (None si spécial)
        word_ids = tok.word_ids(batch_index=0)

        token_labels = torch.full((self.max_len,), IGNORE_VALUE, dtype=torch.long)
        ignored = torch.ones(self.max_len, dtype=torch.bool)

        prev_word_idx = None
        for i, w_idx in enumerate(word_ids):
            # Donner le label uniquement au PREMIER sub-token de chaque mot
            if w_idx is not None and w_idx != prev_word_idx:
                token_labels[i] = word_labels[w_idx]
                ignored[i] = False
            prev_word_idx = w_idx

        # word_ids en tensor (avec IGNORE_VALUE à la place de None)
        word_ids_tensor = torch.tensor(
            [w if w is not None else IGNORE_VALUE for w in word_ids],
            dtype=torch.long,
        )

        return {
            'document_idx': row['document_idx'],
            'sentence_idx': row['sentence_idx'],
            'word_ids': word_ids_tensor,
            'input_ids': tok['input_ids'].flatten(),
            'attention_mask': tok['attention_mask'].flatten(),
            'ignored': ignored,
            'labels': token_labels,
        }
