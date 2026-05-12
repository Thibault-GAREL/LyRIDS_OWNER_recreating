"""Instructions pour préparer i2b2, GUM et GENTLE — datasets restreints ou avec
format custom qui demandent une intervention manuelle.

⚠️ Ce script ne fait PAS le téléchargement automatique : il imprime les étapes
à faire manuellement. À adapter en preprocesseurs complets si tu as le temps
de coder le parsing CoNLL multi-fichier (GUM/GENTLE) ou XML (i2b2).

Référence : `OWNER/owner/data/preprocessing/{gum,i2b2}.py` contient les parsers
complets si tu veux les porter (BeautifulSoup + nltk pour i2b2).

Lancer :
    python data/1-raw/restricted_datasets.py
"""
INSTRUCTIONS = {
    'i2b2': """
    📋 i2b2 (de-identification challenge, biomédical clinique)
    ────────────────────────────────────────────────────────
    1. Demande l'accès sur https://www.i2b2.org/NLP/DataSets/Main.php (license recherche)
    2. Télécharge le corpus 2014 ou 2016 (XML)
    3. Place les fichiers XML dans : data/1-raw/i2b2/{train,dev,test}/*.xml
    4. Port le parser depuis OWNER/owner/data/preprocessing/i2b2.py :
       - Lit chaque XML via BeautifulSoup, extrait <TEXT> et <TAGS>
       - Tokenize avec nltk (sent_tokenize + wordpunct_tokenize)
       - Aligne les offsets char → (sentence_idx, word_idx)
       - Sortie : data/2-processed/i2b2/{train,test}.json
    """,
    'gum': """
    📋 GUM (Georgetown University Multilayer corpus)
    ────────────────────────────────────────────────
    1. Clone https://github.com/amir-zeldes/gum
    2. Dossier pertinent : `gum/dep/` (CoNLL-U format)
    3. Port le parser depuis OWNER/owner/data/preprocessing/gum.py :
       - Lit les fichiers CoNLL-U un par un
       - Extrait les entités via regex sur la colonne entity_types
       - Filtre les entités nested (garde la plus petite, sauf "abstract")
       - Sortie : data/2-processed/gum/test.json (test seulement)
    """,
    'gentle': """
    📋 GENTLE (dérivé de GUM, texte non-conventionnel)
    ──────────────────────────────────────────────────
    1. Clone https://github.com/gucorpling/gentle
    2. Même format CoNLL-U que GUM → même parser
    3. Réutilise le parser de gum.py en pointant sur les fichiers GENTLE
    4. Sortie : data/2-processed/gentle/test.json
    """,
}

for name, instructions in INSTRUCTIONS.items():
    print(instructions)

print("\n💡 Quand un de ces datasets sera prêt dans data/2-processed/{name}/test.json,")
print("   le script test_cross_domain_eval.py l'utilisera automatiquement.")
print("   Tant qu'il manque, la ligne correspondante du rapport sera marquée 'N/A'.")
