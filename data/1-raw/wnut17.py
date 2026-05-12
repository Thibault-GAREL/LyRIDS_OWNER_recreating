"""Téléchargement + conversion de WNUT 17 au format OWNER.

Source : https://github.com/leondz/emerging_entities_17 (les loading scripts HF
de leondz/wnut_17 et wnut_17 sont dépréciés, donc on tire les .conll depuis le
repo source de l'auteur).

Format des fichiers : `word \\t tag` (BIO), lignes vides entre phrases.

Lancer depuis la racine du projet :
    python data/1-raw/wnut17.py
"""
import sys
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.schema import Dataset, Document, Entity, Metadata
from src.data.serialization import serialize_owner_dataset


BASE_URL = 'https://raw.githubusercontent.com/leondz/emerging_entities_17/master'
RAW_DIR = Path('data/1-raw/wnut17_files')
OUTPUT_DIR = Path('data/2-processed/wnut17')

SPLITS = [
    ('train', 'wnut17train.conll'),
    ('dev',   'emerging.dev.conll'),
    ('test',  'emerging.test.annotated'),
]


def download(url: str, dest: Path) -> None:
    if dest.exists():
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f'  ↓ {url}')
    urllib.request.urlretrieve(url, dest)


def parse_tsv(path: Path) -> Dataset:
    """Parse fichier CoNLL `word TAB tag` en Dataset OWNER."""
    documents: list[Document] = []
    sentence: list[str] = []
    entities: list[Entity] = []
    current: Entity | None = None
    i = 0

    def flush():
        nonlocal sentence, entities, current, i
        if current is not None:
            current.end_word_idx = len(sentence)
            current = None
        if sentence:
            documents.append(Document(
                id=str(len(documents)), sentences=[sentence], entities=entities,
            ))
        sentence, entities = [], []
        i = 0

    with open(path, encoding='utf-8') as f:
        for line in f:
            line = line.rstrip()
            if not line:
                flush()
                continue
            parts = line.split('\t') if '\t' in line else line.split()
            if len(parts) < 2:
                continue
            word, tag = parts[0], parts[1].lower()
            sentence.append(word)
            if (tag == 'o' or tag.startswith('b-')) and current is not None:
                current.end_word_idx = i
                current = None
            if tag.startswith('b-'):
                current = Entity(type=tag[2:], sentence_idx=0, start_word_idx=i, end_word_idx=i + 1)
                entities.append(current)
            elif tag.startswith('i-') and current is not None:
                current.end_word_idx = i + 1
            i += 1
    flush()

    types: set[str] = set()
    for d in documents:
        for e in d.entities:
            types.add(e.type)
    return Dataset(documents=documents, metadata=Metadata(entity_types=types))


def main():
    for owner_name, fname in SPLITS:
        url = f'{BASE_URL}/{fname}'
        raw_path = RAW_DIR / fname
        try:
            download(url, raw_path)
        except Exception as e:
            print(f'  ❌ download échoué ({fname}) : {e}')
            continue
        dataset = parse_tsv(raw_path)
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        serialize_owner_dataset(dataset, str(OUTPUT_DIR / f'{owner_name}.json'))
        n_ents = sum(len(d.entities) for d in dataset.documents)
        print(f'  [{owner_name}] {len(dataset.documents)} phrases, {n_ents} entités, {len(dataset.metadata.entity_types)} types')


if __name__ == '__main__':
    main()
