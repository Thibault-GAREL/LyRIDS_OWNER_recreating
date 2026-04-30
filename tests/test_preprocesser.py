import os
from src.data.preprocessing.conll2003 import Conll2003Preprocessor

# 1) Créer un mini fichier CoNLL factice
mini_conll = """-DOCSTART- -X- -X- O

EU NNP B-NP B-ORG
rejects VBZ B-VP O
German JJ B-NP B-MISC
call NN I-NP O
. . O O

Peter NNP B-NP B-PER
Blackburn NNP I-NP I-PER
visits VBZ B-VP O
Paris NNP B-NP B-LOC
. . O O

-DOCSTART- -X- -X- O

The DT B-NP O
United NNP I-NP B-LOC
States NNP I-NP I-LOC
won VBD B-VP O
. . O O
"""

os.makedirs("data/1-raw/conll2003_mini", exist_ok=True)
for split in ("train", "valid", "test"):
    with open(f"data/1-raw/conll2003_mini/{split}.txt", "w", encoding="utf-8") as f:
        f.write(mini_conll)

# 2) Lancer le preprocesseur
prep = Conll2003Preprocessor(
    train_path="data/1-raw/conll2003_mini/train.txt",
    dev_path="data/1-raw/conll2003_mini/valid.txt",
    test_path="data/1-raw/conll2003_mini/test.txt",
)
prep.preprocess_and_save("data/2-processed/conll2003_mini")

# 3) Recharger et inspecter le résultat
from src.data.serialization import parse_owner_dataset

ds = parse_owner_dataset("data/2-processed/conll2003_mini/train.json")

for doc in ds.documents:
    print(f"\n--- Document {doc.id} ---")
    for i, sent in enumerate(doc.sentences):
        print(f"  Phrase {i}: {sent}")
    for ent in doc.entities:
        words = doc.sentences[ent.sentence_idx][ent.start_word_idx : ent.end_word_idx]
        print(
            f"  Entité [{ent.type}]: {words} (pos={ent.sentence_idx},{ent.start_word_idx}-{ent.end_word_idx})"
        )

print(f"\nTypes d'entités : {ds.metadata.entity_types}")
