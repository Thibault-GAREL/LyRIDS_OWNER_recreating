"""Lecture et écriture des datasets au format OWNER (JSON).

Quatre fonctions seulement : lire/écrire un Dataset complet, idem pour MiniDataset.
Pydantic fait 90% du travail (parsing, validation, sérialisation).
"""
from pathlib import Path

from .schema import Dataset, MiniDataset


def serialize_owner_dataset(dataset: Dataset, output_file: str) -> None:
    """Sauvegarde un Dataset dans un fichier JSON. Crée le dossier parent si besoin.

    Args:
        dataset: l'objet Dataset à sauvegarder.
        output_file: chemin du fichier de sortie (sera écrasé s'il existe).
    """
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as file:
        file.write(dataset.model_dump_json())


def parse_owner_dataset(input_file: str) -> Dataset:
    """Lit un Dataset depuis un fichier JSON.

    Lève une ValidationError si le JSON ne respecte pas le schéma.

    Args:
        input_file: chemin du fichier JSON à lire.

    Returns:
        L'objet Dataset reconstruit, avec validation complète.
    """
    with open(input_file, 'r', encoding='utf-8') as file:
        return Dataset.model_validate_json(file.read())


def serialize_mini_owner_dataset(dataset: MiniDataset, output_file: str) -> None:
    """Sauvegarde un MiniDataset (typiquement les prédictions du modèle)."""
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as file:
        file.write(dataset.model_dump_json())


def parse_mini_owner_dataset(input_file: str) -> MiniDataset:
    """Lit un MiniDataset depuis un fichier JSON."""
    with open(input_file, 'r', encoding='utf-8') as file:
        return MiniDataset.model_validate_json(file.read())
