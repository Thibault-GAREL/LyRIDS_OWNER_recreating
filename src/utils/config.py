"""Chargement des fichiers de configuration YAML.

La config est un simple dict imbriqué (data/, mention_detection/, entity_typing/, output/).
On garde dict plutôt que dataclass pour rester proche du repo de référence OWNER
et faciliter le passage à mlflow.log_params (qui prend un dict).
"""
from pathlib import Path

import yaml


def load_config(path: str | Path) -> dict:
    """Charge un fichier YAML et retourne le dict correspondant."""
    with open(path, encoding='utf-8') as f:
        return yaml.safe_load(f)


def flatten_for_mlflow(config: dict, prefix: str = '') -> dict:
    """Aplatit un dict imbriqué en `section.sub_key: value` pour mlflow.log_params.

    Exemple : {'mention_detection': {'lr': 2e-5}} → {'mention_detection.lr': 2e-5}
    """
    flat = {}
    for key, value in config.items():
        full = f'{prefix}.{key}' if prefix else key
        if isinstance(value, dict):
            flat.update(flatten_for_mlflow(value, full))
        else:
            flat[full] = value
    return flat
