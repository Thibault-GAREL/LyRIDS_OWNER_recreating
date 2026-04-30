from src.data.schema import Entity, Document, Dataset, Metadata
from src.data.serialization import serialize_owner_dataset, parse_owner_dataset

# 1) Construire un dataset factice
doc1 = Document(
    id="doc_1",
    sentences=[
        ["Skai", "TV", "is", "a", "Greek", "network", "."],
        ["It", "is", "based", "in", "Piraeus", "."],
    ],
    entities=[
        Entity(type="ORG", sentence_idx=0, start_word_idx=0, end_word_idx=2),
        Entity(type="MISC", sentence_idx=0, start_word_idx=4, end_word_idx=5),
        Entity(type="LOC", sentence_idx=1, start_word_idx=4, end_word_idx=5),
    ],
)

doc2 = Document(
    id="doc_2",
    sentences=[["Paris", "is", "the", "capital", "of", "France", "."]],
    entities=[
        Entity(type="LOC", sentence_idx=0, start_word_idx=0, end_word_idx=1),
        Entity(type="LOC", sentence_idx=0, start_word_idx=5, end_word_idx=6),
    ],
)

ds = Dataset(
    documents=[doc1, doc2],
    metadata=Metadata(entity_types={"ORG", "MISC", "LOC"}),
)

# 2) Sauvegarder sur disque
serialize_owner_dataset(ds, "data/2-processed/demo.json")

# 3) Recharger depuis disque
ds_reloaded = parse_owner_dataset("data/2-processed/demo.json")

# 4) Vérifier l'égalité (Pydantic compare champ à champ)
assert ds.documents[0].id == ds_reloaded.documents[0].id
assert len(ds.documents) == len(ds_reloaded.documents)
print("OK !")
print(f"Nombre de docs : {len(ds_reloaded.documents)}")
print(f"Types d'entités : {ds_reloaded.metadata.entity_types}")
print(f"1ère entité du 1er doc : {ds_reloaded.documents[0].entities[0]}")
