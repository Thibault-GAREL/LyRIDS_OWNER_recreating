"""Téléchargement + conversion de Davlan/conll2003_noMISC au format OWNER.

À lancer depuis la racine du projet :
    python data/1-raw/conll2003_noMISC.py

Lien : https://huggingface.co/datasets/Davlan/conll2003_noMISC
"""
import sys
from pathlib import Path

# Permet les imports `from src...` quand on lance ce fichier en standalone
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.preprocessing.huggingface import HuggingFaceNerPreprocessor


prep = HuggingFaceNerPreprocessor(
    dataset_name='Davlan/conll2003_noMISC',
    cache_dir='data/1-raw/.hf_cache',   # cache HF local au projet
)
prep.preprocess_and_save('data/2-processed/conll2003_noMISC')
