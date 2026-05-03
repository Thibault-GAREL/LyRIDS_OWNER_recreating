"""Helpers MLflow pour le pipeline OWNER.

Permet aux trainers (MD, ET, NER) d'ouvrir un run MLflow en mode "nested" si un
run parent est déjà actif (cas du pipeline NER end-to-end), ou en mode "root"
sinon (cas d'un trainer utilisé seul, ex. tests/test_md_training.py).
"""
from contextlib import contextmanager

import mlflow


@contextmanager
def mlflow_run(run_name: str):
    """Ouvre un run MLflow, nested automatiquement si un run est déjà actif."""
    nested = mlflow.active_run() is not None
    with mlflow.start_run(run_name=run_name, nested=nested) as run:
        yield run


def log_params_safe(params: dict) -> None:
    """Log des hyperparams seulement si un run MLflow est actif."""
    if mlflow.active_run() is None:
        return
    mlflow.log_params(params)


def log_metrics_safe(metrics: dict, step: int | None = None) -> None:
    """Log des métriques (filtrées sur les valeurs numériques) si un run est actif."""
    if mlflow.active_run() is None:
        return
    numeric = {k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))}
    mlflow.log_metrics(numeric, step=step)


def log_artifacts_safe(folder: str) -> None:
    """Log des artefacts d'un dossier si un run MLflow est actif."""
    if mlflow.active_run() is None:
        return
    mlflow.log_artifacts(folder)
