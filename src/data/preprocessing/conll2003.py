"""Preprocesseur du dataset CoNLL-2003 (NER anglais en format BIO).

Format source : un mot par ligne, 4 colonnes séparées par des espaces :
    mot  POS  chunk-tag  NER-tag

avec :
    - lignes vides = séparation de phrases
    - `-DOCSTART- -X- -X- O` = début d'un nouveau document
    - NER-tag au format BIO : `O`, `B-XXX`, `I-XXX`
"""
import os

from .base import BasePreprocessor
from ..schema import Dataset, Document, Entity, Metadata
from ..serialization import serialize_owner_dataset


class Conll2003Preprocessor(BasePreprocessor):
    """Convertit CoNLL-2003 vers le format OWNER."""

    def __init__(self, train_path: str, dev_path: str, test_path: str):
        """
        Args:
            train_path: chemin du fichier train.txt CoNLL.
            dev_path: chemin du fichier valid.txt CoNLL.
            test_path: chemin du fichier test.txt CoNLL.
        """
        self.train_path = train_path
        self.dev_path = dev_path
        self.test_path = test_path

    def read_split(self, path: str) -> Dataset:
        """Lit un fichier CoNLL et retourne un Dataset OWNER.

        Algo : machine à états. À chaque ligne, on met à jour
        - la liste des documents
        - l'entité actuellement ouverte (si on est au milieu d'une entité multi-mots)
        """
        documents: list[Document] = []
        current_entity: Entity | None = None
        word_idx = 0   # index du mot dans la phrase courante

        with open(path, 'r', encoding='utf-8') as file:
            for raw_line in file:
                line = raw_line.strip()
                tsv = line.split(' ')
                document = documents[-1] if documents else None

                # === CAS 1 : début d'un nouveau document ===
                if line == '-DOCSTART- -X- -X- O':
                    documents.append(Document(
                        id=str(len(documents)),
                        sentences=[],
                        entities=[],
                    ))
                    current_entity = None

                # === CAS 2 : un token (4 colonnes) ===
                elif len(tsv) == 4:
                    word, _pos, _chunk, ner_tag = tsv
                    document.sentences[-1].append(word)
                    ner_tag = ner_tag.lower()

                    # Fermeture de l'entité précédente si on sort (O) ou si on
                    # commence une nouvelle entité (B-)
                    if (ner_tag == 'o' or ner_tag.startswith('b-')) and current_entity is not None:
                        current_entity.end_word_idx = word_idx
                        current_entity = None

                    # Ouverture d'une nouvelle entité
                    if ner_tag.startswith('b-'):
                        current_entity = Entity(
                            type=ner_tag[2:],   # "B-ORG" -> "ORG"
                            sentence_idx=len(document.sentences) - 1,
                            start_word_idx=word_idx,
                            end_word_idx=word_idx,   # provisoire, sera mis à jour à la fermeture
                        )
                        document.entities.append(current_entity)

                    word_idx += 1

                # === CAS 3 : ligne vide = fin de phrase ===
                elif len(tsv) == 1:
                    # Si une entité était ouverte, on la ferme à la fin de la phrase
                    if current_entity is not None:
                        current_entity.end_word_idx = len(document.sentences[-1])
                        current_entity = None
                    word_idx = 0
                    document.sentences.append([])

                else:
                    raise SyntaxError(f'Format de ligne inattendu : {line!r}')

        # Cas limite : entité encore ouverte à la fin du fichier
        if current_entity is not None:
            current_entity.end_word_idx = len(documents[-1].sentences[-1])

        # Ménage : retirer les phrases vides (créées par les lignes vides finales)
        for document in documents:
            document.sentences = [s for s in document.sentences if len(s) > 0]

        # Construire la metadata (set de tous les types vus)
        entity_types: set[str | int] = set()
        for document in documents:
            for entity in document.entities:
                entity_types.add(entity.type)

        return Dataset(
            documents=documents,
            metadata=Metadata(entity_types=entity_types),
        )

    def preprocess_and_save(self, output_folder: str) -> None:
        os.makedirs(output_folder, exist_ok=True)

        for split_name, path in [
            ('train', self.train_path),
            ('dev', self.dev_path),
            ('test', self.test_path),
        ]:
            dataset = self.read_split(path)
            serialize_owner_dataset(dataset, f'{output_folder}/{split_name}.json')
            print(f'  [{split_name}] {len(dataset.documents)} docs, '
                  f'{sum(len(d.entities) for d in dataset.documents)} entités, '
                  f'types : {dataset.metadata.entity_types}')
