"""Utilitaires PyTorch partagés par tout le projet."""
import multiprocessing


# Convention PyTorch : nn.CrossEntropyLoss(ignore_index=-100) ignore par défaut
# les positions ayant cette valeur. On réutilise -100 pour les sub-tokens et
# tokens spéciaux qu'on ne veut pas inclure dans la loss.
IGNORE_VALUE = -100


def get_num_workers() -> int:
    """Nombre de workers raisonnable pour un DataLoader PyTorch.

    Bonne approximation : le nombre de cœurs CPU. Sur Windows, si tu
    rencontres des bugs de multiprocessing au lancement, force à 0.
    """
    return multiprocessing.cpu_count()
