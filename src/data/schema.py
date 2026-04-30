"""Modèles de données du format OWNER.

Représente la structure d'un dataset NER : entités, documents, dataset complet.
Les classes héritent de pydantic.BaseModel pour bénéficier de la validation
automatique des types et de la sérialisation JSON gratuite.
"""
from pydantic import BaseModel, Field


# Alias de lisibilité : une phrase est une liste de mots ("tokens").
Sentence = list[str]


class Entity(BaseModel):
    """Une entité nommée localisée dans un document.

    On ne stocke pas le texte de l'entité : il est dérivable depuis
    `sentence_idx`, `start_word_idx`, `end_word_idx` et la liste des phrases
    du document parent.

    Convention : `[start_word_idx, end_word_idx[` (end est exclusif), comme
    les slices Python.
    """

    # str pour les vrais types ("PER", "LOC", "ORG", ...)
    # int pour les types issus du clustering (0, 1, 2, ...)
    type: str | int

    sentence_idx: int       # index de la phrase dans le document
    start_word_idx: int     # index du 1er mot de l'entité (inclus)
    end_word_idx: int       # index du dernier mot + 1 (exclus)


class MiniDocument(BaseModel):
    """Document minimal — utilisé pour stocker les prédictions du modèle.

    Pas besoin du texte : on n'a que l'id du document et la liste
    des entités (prédites ou réelles).
    """

    id: str                                                # identifiant du document
    entities: list[Entity] = Field(default_factory=list)   # liste vide par défaut


class Document(MiniDocument):
    """Document complet — utilisé dans les datasets train/dev/test.

    Hérite de MiniDocument (donc a `id` et `entities`) et ajoute le texte
    sous forme de liste de phrases, chaque phrase étant une liste de mots.
    """

    sentences: list[Sentence]   # ex. [["Skai", "TV", "is", "Greek"], ["It", "is", "..."]]


class Metadata(BaseModel):
    """Métadonnées d'un dataset.

    Pour l'instant, on ne stocke que l'ensemble des types d'entités présents.
    Un set garantit l'unicité et un lookup en O(1).
    """

    entity_types: set[str | int] = Field(default_factory=set)


class Dataset(BaseModel):
    """Dataset complet — utilisé en entrée (train/dev/test)."""

    documents: list[Document]
    metadata: Metadata


class MiniDataset(BaseModel):
    """Dataset minimal — utilisé pour les prédictions du modèle.

    Mêmes champs que Dataset, mais avec des MiniDocument (sans le texte).
    """

    documents: list[MiniDocument]
    metadata: Metadata
