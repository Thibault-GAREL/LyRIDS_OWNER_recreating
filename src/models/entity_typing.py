"""Modèles pour Entity Typing.

Deux composants :
    - EntityEncodingModel : PLM + dropout, extrait l'embedding du [MASK].
    - AutoKmeans         : KMeans qui choisit automatiquement k via le BIC.
"""
import numpy as np
import torch
from torch import nn
from transformers import AutoModel
from sklearn.cluster import KMeans


class EntityEncodingModel(nn.Module):
    """Encoder qui retourne l'embedding du token [MASK] dans un prompt.

    Le PLM (BERT/DeBERTa) sait prédire ce qui devrait remplacer [MASK].
    L'embedding interne au [MASK] capture donc une représentation type-aware
    de l'entité.
    """

    def __init__(self, plm_name: str):
        super().__init__()
        self.plm = AutoModel.from_pretrained(plm_name, return_dict=True)
        self.dropout = nn.Dropout(p=0.1)

    def forward(
        self,
        input_ids: torch.Tensor,        # [B, T]
        attention_mask: torch.Tensor,   # [B, T]
        mask_index: torch.Tensor,       # [B]   - position du [MASK] par instance
    ) -> torch.Tensor:                  # -> [B, hidden_size]
        token_embeddings = self.plm(
            input_ids=input_ids, attention_mask=attention_mask,
        ).last_hidden_state                                     # [B, T, hidden]

        # Pour chaque ligne du batch, prendre le token à la position mask_index[i]
        batch_size = token_embeddings.shape[0]
        entity_embedding = token_embeddings[
            torch.arange(batch_size, device=token_embeddings.device),
            mask_index.flatten(),
        ]                                                       # [B, hidden]

        return self.dropout(entity_embedding)


# ============================================================
#  AutoKmeans : K-means avec choix automatique de k via BIC
# ============================================================

def bic_kmeans(estimator: KMeans, X: np.ndarray) -> float:
    """BIC d'un KMeans dans le cas Gaussien.

    Formule : BIC = n * log(inertia / n) + log(n) * k
    Plus le BIC est petit, mieux c'est.
    Source : https://github.com/TankredO/pyckmeans
    """
    k = estimator.cluster_centers_.shape[0]
    n = X.shape[0]
    return n * np.log(estimator.inertia_ / n) + np.log(n) * k


class AutoKmeans:
    """K-means qui choisit automatiquement k dans [k_min, k_max] via BIC.

    Lance KMeans pour chaque k de la grille, calcule le BIC, garde le meilleur.
    """

    def __init__(self, k_min: int, k_max: int, k_step: int = 1, seed: int | None = None):
        if k_min < 1:
            raise ValueError(f"k_min doit être >= 1, reçu {k_min}")
        if k_min > k_max:
            raise ValueError(f"k_min ({k_min}) > k_max ({k_max})")

        self.k_min = k_min
        self.k_max = k_max
        self.k_step = k_step
        self.seed = seed

        self._best_kmeans: KMeans | None = None
        self._best_k: int | None = None
        self._best_bic: float = float('inf')
        self.all_results: list[tuple[int, float]] = []   # liste (k, bic) pour debug

    @property
    def k(self) -> int:
        """Nombre de clusters optimal trouvé."""
        if self._best_k is None:
            raise RuntimeError("Appeler fit() avant d'accéder à k.")
        return self._best_k

    def fit(self, embeddings: torch.Tensor) -> None:
        X = embeddings.cpu().numpy()

        for k in range(self.k_min, self.k_max + 1, self.k_step):
            km = KMeans(
                n_clusters=k,
                init='k-means++',
                n_init=10,
                random_state=self.seed,
            )
            km.fit(X)
            bic = bic_kmeans(km, X)
            self.all_results.append((k, bic))

            if bic < self._best_bic:
                self._best_bic = bic
                self._best_kmeans = km
                self._best_k = k

    def predict(self, embeddings: torch.Tensor) -> torch.Tensor:
        if self._best_kmeans is None:
            raise RuntimeError("Appeler fit() avant predict().")
        clusters = self._best_kmeans.predict(embeddings.cpu().numpy())
        return torch.from_numpy(clusters)
