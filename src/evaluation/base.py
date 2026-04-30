"""Utilitaires d'évaluation partagés entre Mention Detection et Entity Typing."""
from ..data.schema import Dataset, Document, Entity, MiniDocument


def convert_document_to_entities(document: Document) -> MiniDocument:
    """Convertit un Document (avec texte + entités) en MiniDocument.

    Garde le vrai type de chaque entité — utile comme ground truth pour
    la comparaison avec les clusters prédits (Entity Typing).
    """
    return MiniDocument(
        id=document.id,
        entities=[
            Entity(
                type=e.type,
                sentence_idx=e.sentence_idx,
                start_word_idx=e.start_word_idx,
                end_word_idx=e.end_word_idx,
            )
            for e in document.entities
        ],
    )


def merge_dataset_with_predictions(
    dataset: Dataset, predictions: list[MiniDocument],
) -> Dataset:
    """Renvoie une copie du dataset où les entités sont remplacées par celles prédites.

    Permet d'enchaîner Mention Detection → Entity Typing : on remplace les
    entités ground truth par celles prédites par MD, puis on passe ce dataset
    à ET pour les clusteriser.
    """
    if len(dataset.documents) != len(predictions):
        raise ValueError(
            f"Tailles différentes : dataset={len(dataset.documents)}, "
            f"predictions={len(predictions)}"
        )

    new_dataset = dataset.model_copy(deep=True)
    pred_by_id = {p.id: p for p in predictions}
    new_entity_types: set[str | int] = set()

    for document in new_dataset.documents:
        pred_doc = pred_by_id.get(document.id)
        if pred_doc is None:
            document.entities = []
            continue

        document.entities = [
            Entity(
                type=str(e.type),                 # str pour compat ET (entity_type_to_id)
                sentence_idx=e.sentence_idx,
                start_word_idx=e.start_word_idx,
                end_word_idx=e.end_word_idx,
            )
            for e in pred_doc.entities
        ]
        new_entity_types.update(e.type for e in document.entities)

    new_dataset.metadata.entity_types = new_entity_types
    return new_dataset


def align_entities_by_position(
    truths: list[MiniDocument],
    preds: list[MiniDocument],
) -> tuple[list, list]:
    """Aligne les entités truth/pred par leur position pour eval end-to-end.

    Pour chaque document, on matche les entités par triplet (sentence_idx,
    start_word_idx, end_word_idx). Les entités non matchées reçoivent une
    étiquette spéciale (chaîne fixe pour truth manquante, int négatif pour pred
    manquante) — qui sera traitée comme un label distinct par AMI/ARI.

    Returns:
        (true_labels, pred_labels) : deux listes de même longueur, avec les
        types/clusters alignés. Compatible direct avec sklearn AMI/ARI.
    """
    truths_by_id = {d.id: d for d in truths}
    preds_by_id = {d.id: d for d in preds}
    all_ids = set(truths_by_id) | set(preds_by_id)

    true_labels: list = []
    pred_labels: list = []

    for doc_id in all_ids:
        true_doc = truths_by_id.get(doc_id, MiniDocument(id=doc_id))
        pred_doc = preds_by_id.get(doc_id, MiniDocument(id=doc_id))

        true_matched = [False] * len(true_doc.entities)
        pred_matched = [False] * len(pred_doc.entities)

        # TP : entités au même triplet (sent_idx, start, end)
        for i_t, t_ent in enumerate(true_doc.entities):
            for i_p, p_ent in enumerate(pred_doc.entities):
                if pred_matched[i_p]:
                    continue
                if (t_ent.sentence_idx == p_ent.sentence_idx
                        and t_ent.start_word_idx == p_ent.start_word_idx
                        and t_ent.end_word_idx == p_ent.end_word_idx):
                    true_labels.append(t_ent.type)
                    pred_labels.append(p_ent.type)
                    true_matched[i_t] = True
                    pred_matched[i_p] = True
                    break

        # FN : truth non matché → on met un label fantôme côté pred
        for i_t, matched in enumerate(true_matched):
            if not matched:
                true_labels.append(true_doc.entities[i_t].type)
                pred_labels.append('__no_pred__')

        # FP : pred non matché → label fantôme côté truth
        for i_p, matched in enumerate(pred_matched):
            if not matched:
                true_labels.append('__no_truth__')
                pred_labels.append(pred_doc.entities[i_p].type)

    return true_labels, pred_labels
