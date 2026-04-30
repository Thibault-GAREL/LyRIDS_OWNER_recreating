"""Évaluation de Mention Detection : conversion BIO → entités + P/R/F1."""
import torch

from ..data.schema import Document, Entity, MiniDocument


def convert_document_to_entities(document: Document) -> MiniDocument:
    """Convertit un Document (ground truth) en MiniDocument pour comparaison.

    On force le type à "entity" car en Mention Detection, on ne s'occupe pas
    du typage : seul compte l'emplacement de l'entité.
    """
    return MiniDocument(
        id=document.id,
        entities=[
            Entity(
                type="entity",
                sentence_idx=e.sentence_idx,
                start_word_idx=e.start_word_idx,
                end_word_idx=e.end_word_idx,
            )
            for e in document.entities
        ],
    )


def convert_bio_to_entities(
    bio_tags: torch.Tensor,
    word_ids: torch.Tensor,
    sentence_idx: int,
    document_id: str,
) -> MiniDocument:
    """Convertit une séquence de tags BIO (sub-tokens) en liste d'entités.

    Args:
        bio_tags: tensor 1D [length] avec valeurs 0=O, 1=B, 2=I.
        word_ids: tensor 1D [length] avec l'index du mot d'origine pour chaque sub-token.
        sentence_idx: index de la phrase dans le document.
        document_id: id du document.

    Returns:
        MiniDocument avec les entités extraites (au niveau mot).
    """
    bio_tags = bio_tags.cpu()
    word_ids = word_ids.cpu()

    entities: list[Entity] = []
    state = 0   # 0 = outside, 1 = inside
    start_pos = 0
    end_pos = 0

    for i, tag in enumerate(bio_tags.tolist()):
        if state == 0:
            if tag == 1:    # B → début d'entité
                state = 1
                start_pos = i
                end_pos = i
            # tag == 0 (O) ou 2 (I sans B avant, invalide) → on continue
        else:               # state == 1 (inside)
            if tag == 0:    # O → fin de l'entité courante
                entities.append(_make_entity(sentence_idx, start_pos, end_pos))
                state = 0
            elif tag == 1:  # B → fin de l'entité courante + début d'une nouvelle
                entities.append(_make_entity(sentence_idx, start_pos, end_pos))
                start_pos = i
                end_pos = i
            else:           # I → continuation
                end_pos = i

    # Cas limite : entité encore ouverte à la fin
    if state == 1:
        entities.append(_make_entity(sentence_idx, start_pos, end_pos))

    # Convertir indices token → indices mot via word_ids
    for entity in entities:
        entity.start_word_idx = int(word_ids[entity.start_word_idx].item())
        entity.end_word_idx = int(word_ids[entity.end_word_idx].item()) + 1   # +1 car end exclusif

    return MiniDocument(id=document_id, entities=entities)


def _make_entity(sentence_idx: int, start: int, end: int) -> Entity:
    return Entity(
        type="entity",
        sentence_idx=sentence_idx,
        start_word_idx=start,
        end_word_idx=end,
    )


def evaluate_mention_detection(
    truths: list[MiniDocument],
    preds: list[MiniDocument],
) -> dict[str, float]:
    """Calcule precision, recall, F1 pour Mention Detection.

    Une entité est correcte si elle a EXACTEMENT le même
    (sentence_idx, start_word_idx, end_word_idx). Le type est ignoré.
    """
    truths_by_id = {d.id: d for d in truths}
    preds_by_id = {d.id: d for d in preds}
    all_ids = set(truths_by_id.keys()) | set(preds_by_id.keys())

    tp = 0
    fp = 0
    fn = 0

    for doc_id in all_ids:
        true_doc = truths_by_id.get(doc_id, MiniDocument(id=doc_id, entities=[]))
        pred_doc = preds_by_id.get(doc_id, MiniDocument(id=doc_id, entities=[]))

        true_set = {(e.sentence_idx, e.start_word_idx, e.end_word_idx) for e in true_doc.entities}
        pred_set = {(e.sentence_idx, e.start_word_idx, e.end_word_idx) for e in pred_doc.entities}

        tp += len(true_set & pred_set)
        fp += len(pred_set - true_set)
        fn += len(true_set - pred_set)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0 else 0.0
    )

    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tp': tp,
        'fp': fp,
        'fn': fn,
    }
