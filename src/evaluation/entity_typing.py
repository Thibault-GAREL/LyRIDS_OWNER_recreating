"""Évaluation de Entity Typing : AMI et ARI entre clusters et vrais types.

Simplification vs original : on suppose que les entités prédites correspondent
exactement aux entités ground truth (cas où on évalue ET en isolation, sans
passer par MD). Pour la version end-to-end, l'évaluation doit gérer les
mismatches entre les deux ensembles d'entités.
"""
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score


def evaluate_entity_typing(
    true_type_ids: list,
    pred_cluster_ids: list,
) -> dict[str, float]:
    """Calcule AMI et ARI entre les vrais types et les clusters prédits.

    Args:
        true_type_ids:    [N] - vrai type de chaque entité (int, str, peu importe).
        pred_cluster_ids: [N] - cluster/type prédit (int, str, peu importe).
            sklearn accepte n'importe quel label hashable.

    Returns:
        dict avec 'ami' (Adjusted Mutual Info) et 'ari' (Adjusted Rand Index).
        Les deux dans [0, 1] : 0 = aléatoire, 1 = parfait.
    """
    if len(true_type_ids) != len(pred_cluster_ids):
        raise ValueError(
            f"Taille différente : truths={len(true_type_ids)}, "
            f"preds={len(pred_cluster_ids)}"
        )

    return {
        'ami': adjusted_mutual_info_score(true_type_ids, pred_cluster_ids),
        'ari': adjusted_rand_score(true_type_ids, pred_cluster_ids),
    }
