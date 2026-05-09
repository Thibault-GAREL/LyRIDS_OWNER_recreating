"""Téléchargement + conversion de Universal-NER/Pile-NER-type au format OWNER.

Pile-NER est un dataset de NER open-world distillé de ChatGPT, au format
conversationnel. Il diffère fondamentalement de CoNLL :
    - PAS de colonnes `tokens` / `ner_tags` standard, donc HuggingFaceNerPreprocessor
      ne fonctionne PAS dessus → on a un parser dédié dans ce fichier.
    - PAS de split test natif (seulement `train`) → on en crée un à la main
      via un train/test split reproductible.
    - DATASET ÉNORME (~456 k phrases) → on sous-échantillonne par défaut pour
      tenir sur un GPU ~6 Go VRAM. Ajuste SAMPLE_SIZE si tu veux plus.
    - DES CENTAINES de types d'entités (open-world) → pense à augmenter
      `entity_typing.k_max` dans le YAML config.

Lancer depuis la racine du projet :
    python data/1-raw/pile_ner.py

Lien : https://huggingface.co/datasets/Universal-NER/Pile-NER-type
"""
import json
import os
import random
import re
import sys
from pathlib import Path

# Permet `from src...` quand on lance ce fichier en standalone
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets import load_dataset
from nltk import wordpunct_tokenize
from tqdm.auto import tqdm

from src.data.schema import Dataset, Document, Entity, Metadata
from src.data.serialization import serialize_owner_dataset


# ============================================================================
# Hyperparams ajustables
# ============================================================================
SAMPLE_SIZE = 10_000          # Nombre de phrases à garder (None = tout)
TEST_RATIO = 0.10             # Fraction du dataset pour le test split
SEED = 42                     # Reproductibilité du split
OUTPUT_DIR = 'data/2-processed/pile_ner'
CACHE_DIR = 'data/1-raw/.hf_cache'
ASCII_ONLY = True             # Heuristique simple : filtre les phrases majoritairement non-ASCII
ASCII_THRESHOLD = 0.90        # Garde si >= 90% des chars sont ASCII

# ============================================================================
# Parser Pile-NER
# ============================================================================

TYPE_EXTRACTOR = re.compile(r'^What describes (.+) in the text\?$')
LEFT_BOUNDARY = r'[ \(\)\[\]\{\}?,;:.!"\'*\-/\\|_><=&]'
RIGHT_BOUNDARY = rf'(s?{LEFT_BOUNDARY}|$)'


def is_mostly_ascii(text: str, threshold: float = 0.90) -> bool:
    """Heuristique simple pour filtrer les phrases non-anglaises sans langdetect."""
    if not text:
        return False
    n_ascii = sum(1 for c in text if ord(c) < 128)
    return n_ascii / len(text) >= threshold


def parse_document(document: dict) -> Document | None:
    """Parse un document Pile-NER au format conversation → Document OWNER.

    Le format est :
        conversations[0]['value'] = "Text: <la phrase>"
        conversations[1]['value'] = ack ("I've read this text")
        conversations[2k]['value']   = "What describes <type> in the text?"
        conversations[2k+1]['value'] = JSON list ["entity1", "entity2", ...]

    On retourne None si la phrase n'est pas exploitable (filtre langue, parsing).
    """
    document_id = str(document['id'])
    conversations = document['conversations']
    if not conversations:
        return None

    # 1. Extraire et tokeniser la phrase (premier message)
    raw_text = conversations[0]['value']
    if raw_text.startswith('Text: '):
        raw_text = raw_text[6:]                        # strip "Text: "

    if ASCII_ONLY and not is_mostly_ascii(raw_text, ASCII_THRESHOLD):
        return None

    tokens = wordpunct_tokenize(raw_text)
    if not tokens:
        return None

    # Mappe chaque token à sa position de mot dans la phrase tokenisée pour
    # retrouver les entités par recherche de sous-séquence.
    tokens_lower = [t.lower() for t in tokens]
    entities: list[Entity] = []

    # 2. Parcourir les paires (question, réponse) à partir de l'index 2 (l'ack est à l'index 1)
    for i in range(2, len(conversations) - 1, 2):
        question = conversations[i].get('value', '')
        answer = conversations[i + 1].get('value', '')

        match = TYPE_EXTRACTOR.match(question)
        if not match:
            continue
        entity_type = match.group(1).lower().strip()

        try:
            entity_strings = json.loads(answer)
        except (json.JSONDecodeError, ValueError):
            continue
        if not isinstance(entity_strings, list):
            continue

        for entity_str in entity_strings:
            if not isinstance(entity_str, str) or not entity_str.strip():
                continue
            entity_tokens = [t.lower() for t in wordpunct_tokenize(entity_str)]
            if not entity_tokens:
                continue
            # Recherche de la sous-séquence dans les tokens (insensible à la casse)
            for start in _find_subsequence(tokens_lower, entity_tokens):
                entities.append(Entity(
                    type=entity_type,
                    sentence_idx=0,
                    start_word_idx=start,
                    end_word_idx=start + len(entity_tokens),
                ))

    return Document(id=document_id, sentences=[tokens], entities=entities)


def _find_subsequence(haystack: list[str], needle: list[str]) -> list[int]:
    """Renvoie tous les indices de départ où `needle` apparaît dans `haystack`."""
    if not needle or len(needle) > len(haystack):
        return []
    n = len(needle)
    return [i for i in range(len(haystack) - n + 1) if haystack[i:i + n] == needle]


def filter_nested(documents: list[Document]) -> list[Document]:
    """Si deux entités se chevauchent, on garde la plus longue (in-place)."""
    for doc in documents:
        keep = [True] * len(doc.entities)
        for i, e1 in enumerate(doc.entities):
            for j, e2 in enumerate(doc.entities):
                if i >= j or not (keep[i] and keep[j]):
                    continue
                if e1.sentence_idx != e2.sentence_idx:
                    continue
                # Chevauchement
                if e1.start_word_idx < e2.end_word_idx and e2.start_word_idx < e1.end_word_idx:
                    span1 = e1.end_word_idx - e1.start_word_idx
                    span2 = e2.end_word_idx - e2.start_word_idx
                    if span1 >= span2:
                        keep[j] = False
                    else:
                        keep[i] = False
        doc.entities = [e for e, k in zip(doc.entities, keep) if k]
    return documents


def build_dataset(documents: list[Document]) -> Dataset:
    types: set[str] = set()
    for d in documents:
        for e in d.entities:
            types.add(e.type)
    return Dataset(documents=documents, metadata=Metadata(entity_types=types))


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f'Téléchargement de "Universal-NER/Pile-NER-type" (cache: {CACHE_DIR})...')
    hf_ds = load_dataset('Universal-NER/Pile-NER-type', cache_dir=CACHE_DIR)
    train = hf_ds['train']

    if SAMPLE_SIZE is not None and SAMPLE_SIZE < len(train):
        print(f'Sous-échantillonnage : {len(train)} → {SAMPLE_SIZE} (seed={SEED})')
        train = train.shuffle(seed=SEED).select(range(SAMPLE_SIZE))

    print(f'Parsing de {len(train)} documents...')
    documents: list[Document] = []
    for row in tqdm(train):
        doc = parse_document(row)
        if doc is not None:
            documents.append(doc)
    print(f'  → {len(documents)} documents valides après filtrage')

    print('Filtrage des entités imbriquées (on garde la plus longue)...')
    documents = filter_nested(documents)

    # Train/test split reproductible
    rng = random.Random(SEED)
    rng.shuffle(documents)
    n_test = max(1, int(len(documents) * TEST_RATIO))
    test_docs = documents[:n_test]
    train_docs = documents[n_test:]

    splits = [('train', train_docs), ('test', test_docs)]
    for name, docs in splits:
        dataset = build_dataset(docs)
        out_path = f'{OUTPUT_DIR}/{name}.json'
        serialize_owner_dataset(dataset, out_path)
        n_ents = sum(len(d.entities) for d in dataset.documents)
        n_types = len(dataset.metadata.entity_types)
        print(f'  [{name}] {len(docs)} phrases, {n_ents} entités, {n_types} types distincts')


if __name__ == '__main__':
    main()
