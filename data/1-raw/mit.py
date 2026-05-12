"""Téléchargement + conversion des datasets MIT Movie + MIT Restaurant.

Source : https://groups.csail.mit.edu/sls/downloads/ (les loading scripts HF
de tner/mit_* sont dépréciés, donc on tire les .bio depuis MIT SLS).

⚠️ Format des fichiers MIT : `tag \\t word` (label en PREMIER, contrairement à
CoNLL/CrossNER). On utilise donc word_col=1, tag_col=0 dans le parser.

Lancer depuis la racine du projet :
    python data/1-raw/mit.py
"""
import sys
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.schema import Dataset, Document, Entity, Metadata
from src.data.serialization import serialize_owner_dataset


BASE_URL = 'https://groups.csail.mit.edu/sls/downloads'
RAW_DIR = Path('data/1-raw/mit_files')

# (output_name, download_path, list of (split_owner_name, filename))
DATASETS = [
    ('mit_movie', 'movie', [
        ('train', 'trivia10k13train.bio'),
        ('test',  'trivia10k13test.bio'),
    ]),
    ('mit_restaurant', 'restaurant', [
        ('train', 'restauranttrain.bio'),
        ('test',  'restauranttest.bio'),
    ]),
]


def download(url: str, dest: Path) -> None:
    if dest.exists():
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f'  ↓ {url}')
    urllib.request.urlretrieve(url, dest)


def parse_tsv(path: Path, word_col: int, tag_col: int) -> Dataset:
    """Parse fichier CoNLL-like (TSV) en Dataset OWNER."""
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
    flush()

    types: set[str] = set()
    for d in documents:
        for e in d.entities:
            types.add(e.type)
    return Dataset(documents=documents, metadata=Metadata(entity_types=types))


def main():
    for out_name, subdir, splits in DATASETS:
        print(f'\n========== {out_name} ==========')
        for owner_name, fname in splits:
            url = f'{BASE_URL}/{subdir}/{fname}'
            raw_path = RAW_DIR / subdir / fname
            try:
                download(url, raw_path)
            except Exception as e:
                print(f'  ❌ download échoué ({fname}) : {e}')
                continue
            # MIT : label en premier, word en deuxième
            dataset = parse_tsv(raw_path, word_col=1, tag_col=0)
            out_dir = Path(f'data/2-processed/{out_name}')
            out_dir.mkdir(parents=True, exist_ok=True)
            serialize_owner_dataset(dataset, str(out_dir / f'{owner_name}.json'))
            n_ents = sum(len(d.entities) for d in dataset.documents)
            print(f'  [{owner_name}] {len(dataset.documents)} phrases, {n_ents} entités, {len(dataset.metadata.entity_types)} types')


if __name__ == '__main__':
    main()
