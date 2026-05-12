"""Téléchargement + conversion de GENIA (biomédical, PubMed) au format OWNER.

Format HF : `Rosenberg/genia` — chaque ligne est UNE phrase avec :
    - tokens : List[str]
    - entities : List[{'type': str, 'start': int, 'end': int}]
    - org_id : str — identifiant du document d'origine (plusieurs phrases par doc)

On regroupe les phrases du même `org_id` en un Document avec sentences[i] = phrase i.

Lancer depuis la racine du projet :
    python data/1-raw/genia.py

Lien HF : https://huggingface.co/datasets/Rosenberg/genia
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
OUTPUT_DIR = 'data/2-processed/genia'


def convert_split(hf_split) -> Dataset:
    documents: list[Document] = []
    current_id = None
    current_sentence = 0
    for row in hf_split:
        org_id = row['org_id']
        if org_id != current_id:
            current_id = org_id
            documents.append(Document(id=str(org_id), sentences=[], entities=[]))
            current_sentence = 0
        doc = documents[-1]
        doc.sentences.append(row['tokens'])
        for ent in row['entities']:
            doc.entities.append(Entity(
                type=ent['type'].lower(),
                sentence_idx=current_sentence,
                start_word_idx=ent['start'],
                end_word_idx=ent['end'],
            ))
        current_sentence += 1

    all_types: set[str] = set()
    for d in documents:
        for e in d.entities:
            all_types.add(e.type)
    return Dataset(documents=documents, metadata=Metadata(entity_types=all_types))


def main():
    print('Téléchargement de "Rosenberg/genia"...')
    hf_ds = load_dataset('Rosenberg/genia', cache_dir=CACHE_DIR)

    for owner_name, hf_name in [('train', 'train'), ('dev', 'validation'), ('test', 'test')]:
        if hf_name not in hf_ds:
            print(f'  [{owner_name}] split absent, skip')
            continue
        dataset = convert_split(hf_ds[hf_name])
        out_path = f'{OUTPUT_DIR}/{owner_name}.json'
        serialize_owner_dataset(dataset, out_path)
        n_ents = sum(len(d.entities) for d in dataset.documents)
        print(f'  [{owner_name}] {len(dataset.documents)} documents, {n_ents} entités, {len(dataset.metadata.entity_types)} types')


if __name__ == '__main__':
    main()
