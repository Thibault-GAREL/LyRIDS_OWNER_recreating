"""Téléchargement + conversion de FabNER au format OWNER.

FabNER utilise un schéma **BIOES** (B/I/O/E/S) au lieu du BIO standard. Le
HuggingFaceNerPreprocessor du projet ne gère que BIO → on parse les tags
ici dans un script autonome.

Format HF : `DFKI-SLT/fabner` avec `tokens` + `ner_tags` (ClassLabel BIOES).

Lancer depuis la racine du projet :
    python data/1-raw/fabner.py

Lien HF : https://huggingface.co/datasets/DFKI-SLT/fabner
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets import load_dataset

from src.data.schema import Dataset, Document, Entity, Metadata
from src.data.serialization import serialize_owner_dataset


CACHE_DIR = 'data/1-raw/.hf_cache'
OUTPUT_DIR = 'data/2-processed/fabner'


def parse_bioes(tags: list[str], tokens: list[str]) -> list[Entity]:
    """Machine à états BIOES → liste d'Entity (sentence_idx=0)."""
    entities: list[Entity] = []
    current: Entity | None = None
    for i, raw in enumerate(tags):
        tag = raw.lower()
        prefix = tag.split('-')[0] if '-' in tag else tag
        etype = tag.split('-', 1)[1] if '-' in tag else None

        if prefix == 'b':
            if current is not None:
                current.end_word_idx = i
            current = Entity(type=etype, sentence_idx=0, start_word_idx=i, end_word_idx=i + 1)
            entities.append(current)
        elif prefix == 'i' and current is not None and current.type == etype:
            current.end_word_idx = i + 1
        elif prefix == 'e' and current is not None and current.type == etype:
            current.end_word_idx = i + 1
            current = None
        elif prefix == 's':
            if current is not None:
                current.end_word_idx = i
                current = None
            entities.append(Entity(type=etype, sentence_idx=0, start_word_idx=i, end_word_idx=i + 1))
        else:  # 'o' ou tag incohérent → on ferme l'entité en cours si besoin
            if current is not None:
                current.end_word_idx = i
                current = None
    if current is not None and current.end_word_idx <= current.start_word_idx:
        current.end_word_idx = len(tokens)
    return entities


def convert_split(hf_split) -> Dataset:
    class_names = hf_split.features['ner_tags'].feature.names
    documents: list[Document] = []
    all_types: set[str] = set()
    for idx, row in enumerate(hf_split):
        tokens = row['tokens']
        tags = [class_names[t] for t in row['ner_tags']]
        entities = parse_bioes(tags, tokens)
        for e in entities:
            all_types.add(e.type)
        documents.append(Document(id=str(idx), sentences=[tokens], entities=entities))
    return Dataset(documents=documents, metadata=Metadata(entity_types=all_types))


def main():
    print('Téléchargement de "DFKI-SLT/fabner"...')
    hf_ds = load_dataset('DFKI-SLT/fabner', cache_dir=CACHE_DIR)

    for owner_name, hf_name in [('train', 'train'), ('dev', 'validation'), ('test', 'test')]:
        if hf_name not in hf_ds:
            print(f'  [{owner_name}] split absent, skip')
            continue
        dataset = convert_split(hf_ds[hf_name])
        out_path = f'{OUTPUT_DIR}/{owner_name}.json'
        serialize_owner_dataset(dataset, out_path)
        n_ents = sum(len(d.entities) for d in dataset.documents)
        print(f'  [{owner_name}] {len(dataset.documents)} phrases, {n_ents} entités, {len(dataset.metadata.entity_types)} types')


if __name__ == '__main__':
    main()
