"""Téléchargement + conversion des 5 sous-datasets CrossNER au format OWNER.

Source officielle : https://github.com/zliucr/CrossNER (les loading scripts HF
de ce dataset sont dépréciés, donc on tire les TSV directement depuis github).

Domaines : AI, Literature, Music, Politics, Science.
Format des fichiers : `word \\t tag` (tag style BIO), lignes vides entre phrases.

Lancer depuis la racine du projet :
    python data/1-raw/crossner.py
"""
import sys
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.schema import Dataset, Document, Entity, Metadata
from src.data.serialization import serialize_owner_dataset


DOMAINS = ['ai', 'literature', 'music', 'politics', 'science']
SPLITS = [('train', 'train.txt'), ('dev', 'dev.txt'), ('test', 'test.txt')]
BASE_URL = 'https://raw.githubusercontent.com/zliucr/CrossNER/main/ner_data'
RAW_DIR = Path('data/1-raw/crossner_files')


def download(url: str, dest: Path) -> None:
    """Télécharge `url` vers `dest` si pas déjà présent."""
    if dest.exists():
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f'  ↓ {url}')
    urllib.request.urlretrieve(url, dest)


def parse_tsv(path: Path, word_col: int = 0, tag_col: int = 1) -> Dataset:
    """Parse un fichier CoNLL-like (word TAB tag) → Dataset OWNER."""
    documents: list[Document] = []
    sentence: list[str] = []
    entities: list[Entity] = []
    current: Entity | None = None
    i = 0

    def flush_sentence():
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
                flush_sentence()
                continue
            parts = line.split('\t') if '\t' in line else line.split()
            if len(parts) < 2:
                continue
            word = parts[word_col]
            tag = parts[tag_col].lower()
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
    flush_sentence()

    types: set[str] = set()
    for d in documents:
        for e in d.entities:
            types.add(e.type)
    return Dataset(documents=documents, metadata=Metadata(entity_types=types))


def main():
    for domain in DOMAINS:
        print(f'\n========== CrossNER-{domain} ==========')
        for owner_name, fname in SPLITS:
            url = f'{BASE_URL}/{domain}/{fname}'
            raw_path = RAW_DIR / domain / fname
            try:
                download(url, raw_path)
            except Exception as e:
                print(f'  ❌ download échoué ({fname}) : {e}')
                continue
            dataset = parse_tsv(raw_path)
            out_dir = Path(f'data/2-processed/crossner_{domain}')
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f'{owner_name}.json'
            serialize_owner_dataset(dataset, str(out_path))
            n_ents = sum(len(d.entities) for d in dataset.documents)
            print(f'  [{owner_name}] {len(dataset.documents)} phrases, {n_ents} entités, {len(dataset.metadata.entity_types)} types')


if __name__ == '__main__':
    main()
