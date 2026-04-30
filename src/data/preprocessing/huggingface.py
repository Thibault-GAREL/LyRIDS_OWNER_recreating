"""Preprocesseur générique pour datasets NER au format HuggingFace.

Format attendu : chaque ligne du dataset HF contient
    - une colonne `tokens`   : List[str]
    - une colonne `ner_tags` : List[int] avec ClassLabel ["O", "B-XXX", "I-XXX", ...]

Convention : 1 document = 1 phrase (la notion de document n'a pas vraiment
de sens en NER, et la plupart des datasets HF ne portent pas l'info de doc).
"""
import os

from datasets import load_dataset

from .base import BasePreprocessor
from ..schema import Dataset, Document, Entity, Metadata
from ..serialization import serialize_owner_dataset


class HuggingFaceNerPreprocessor(BasePreprocessor):
    """Convertit un dataset NER HuggingFace au format OWNER."""

    def __init__(
        self,
        dataset_name: str,
        tokens_column: str = 'tokens',
        ner_tags_column: str = 'ner_tags',
        train_split: str = 'train',
        dev_split: str = 'validation',
        test_split: str = 'test',
        cache_dir: str | None = None,
    ):
        self.dataset_name = dataset_name
        self.tokens_column = tokens_column
        self.ner_tags_column = ner_tags_column
        self.train_split = train_split
        self.dev_split = dev_split
        self.test_split = test_split
        self.cache_dir = cache_dir

    def _convert_split(self, hf_split) -> Dataset:
        """Convertit un split HF (Dataset HF) en Dataset OWNER.

        Supporte deux formats pour `ner_tags` :
            - Sequence(ClassLabel) : int + mapping `feature.names`
            - Sequence(Value('string')) : strings directes ("B-PER", "O", ...)
        """
        inner_feature = hf_split.features[self.ner_tags_column].feature
        if hasattr(inner_feature, 'names'):
            class_names = inner_feature.names
            def to_tag(t):
                return class_names[t]
        else:
            def to_tag(t):
                return t   # déjà une string

        documents: list[Document] = []
        all_types: set[str | int] = set()

        for idx, row in enumerate(hf_split):
            tokens: list[str] = row[self.tokens_column]
            tag_values = row[self.ner_tags_column]
            tags: list[str] = [to_tag(t) for t in tag_values]

            entities = self._bio_to_entities(tags)
            for e in entities:
                all_types.add(e.type)

            documents.append(Document(
                id=str(idx),
                sentences=[tokens],
                entities=entities,
            ))

        return Dataset(
            documents=documents,
            metadata=Metadata(entity_types=all_types),
        )

    @staticmethod
    def _bio_to_entities(tags: list[str]) -> list[Entity]:
        """Machine à états BIO → liste d'Entity (sur sentence_idx=0).

        Type d'entité en minuscules pour rester cohérent avec Conll2003Preprocessor.
        """
        entities: list[Entity] = []
        current: Entity | None = None

        for i, raw_tag in enumerate(tags):
            tag = raw_tag.lower()

            # Fermeture si on sort (O) ou si on commence une nouvelle (B-)
            if (tag == 'o' or tag.startswith('b-')) and current is not None:
                current.end_word_idx = i
                current = None

            if tag.startswith('b-'):
                current = Entity(
                    type=tag[2:],     # "b-org" -> "org"
                    sentence_idx=0,
                    start_word_idx=i,
                    end_word_idx=i,   # provisoire
                )
                entities.append(current)

        # Cas limite : entité encore ouverte à la fin
        if current is not None:
            current.end_word_idx = len(tags)

        return entities

    def preprocess_and_save(self, output_folder: str) -> None:
        os.makedirs(output_folder, exist_ok=True)
        print(f'Téléchargement de "{self.dataset_name}"...')
        hf_ds = load_dataset(self.dataset_name, cache_dir=self.cache_dir)

        split_mapping = [
            ('train', self.train_split),
            ('dev', self.dev_split),
            ('test', self.test_split),
        ]
        for owner_name, hf_name in split_mapping:
            if hf_name not in hf_ds:
                print(f'  [{owner_name}] split "{hf_name}" absent, skip')
                continue

            dataset = self._convert_split(hf_ds[hf_name])
            out_path = f'{output_folder}/{owner_name}.json'
            serialize_owner_dataset(dataset, out_path)

            n_docs = len(dataset.documents)
            n_ents = sum(len(d.entities) for d in dataset.documents)
            print(
                f'  [{owner_name}] {n_docs} phrases, {n_ents} entités, '
                f'types : {dataset.metadata.entity_types}'
            )
