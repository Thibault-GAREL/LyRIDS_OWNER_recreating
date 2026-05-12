"""Entraîne UNIQUEMENT Entity Typing en réutilisant un checkpoint MD pré-existant.

Cas d'usage : tu as déjà entraîné MD (ex. sur CoNLL) et tu veux entraîner ET sur
un autre dataset (Pile-NER) sans repasser par l'entraînement MD coûteux.

La config doit définir une section `pretrained.mention_detection_dir` pointant
vers le dossier contenant `mention_detection.pt` à recharger.

Lancement :
    python -m tests.test_ner_training_et_only configs/pile_ner_et_only.yaml
"""

# python -m tests.test_ner_training_et_only configs/pile_ner_et_only.yaml

import shutil
import sys
from pathlib import Path

import mlflow

from src.training.entity_typing import EntityTypingTrainer
from src.training.mention_detection import MentionDetectionTrainer
from src.training.ner import NerTrainer
from src.utils.config import flatten_for_mlflow, load_config

config_path = sys.argv[1] if len(sys.argv) > 1 else 'configs/pile_ner_et_only.yaml'
config = load_config(config_path)

if 'pretrained' not in config or 'mention_detection_dir' not in config['pretrained']:
    raise ValueError(
        "Config invalide : la section `pretrained.mention_detection_dir` est "
        "requise pour ce script (sinon utilise tests/test_ner_training.py)."
    )

# 1. MD : instancier + recharger les poids déjà entraînés (pas de train)
md = MentionDetectionTrainer(
    train_path=config['data']['train_path'],
    test_path=config['data']['test_path'],
    **config['mention_detection'],
)
pretrained_md_dir = config['pretrained']['mention_detection_dir']
md_weights_path = Path(pretrained_md_dir) / 'mention_detection.pt'
if not md_weights_path.exists():
    raise FileNotFoundError(
        f"Pas de checkpoint MD trouvé à {md_weights_path}. "
        f"Vérifie `pretrained.mention_detection_dir` dans la config."
    )
print(f'Chargement du MD pré-entraîné depuis : {pretrained_md_dir}')
md.load_model(pretrained_md_dir)

# 2. ET : à entraîner
et = EntityTypingTrainer(
    train_path=config['data']['train_path'],
    test_path=config['data']['test_path'],
    **config['entity_typing'],
)

mlflow.set_experiment(config['output']['mlflow_experiment'])
with mlflow.start_run(run_name=config['output']['mlflow_run_name']):
    mlflow.log_params(flatten_for_mlflow(config))
    mlflow.log_artifact(config_path)

    ner = NerTrainer(md, et)

    # MD n'a besoin que du test set (pour l'éval end-to-end finale).
    # ET a besoin du train + test pour s'entraîner.
    md.load_data(training=False)
    et.load_data(training=True)

    checkpoint_dir = config['output']['checkpoint_dir']
    # Entraîne UNIQUEMENT ET, avec best-checkpoint sur l'AMI
    et.train(checkpoint_dir=checkpoint_dir)

    metrics = ner.evaluate()
    print("\n========== NER end-to-end ==========")
    print(f'AMI={metrics["ami"]:.4f}  ARI={metrics["ari"]:.4f}')
    print(f'n_truth={metrics["n_truth"]}  n_pred_md={metrics["n_pred"]}')

    # Save : MD est identique au pré-entraîné (dupliqué dans le nouveau dossier
    # pour que le checkpoint soit auto-suffisant), ET correspond au best epoch.
    ner.save_model(checkpoint_dir)
    shutil.copy(config_path, Path(checkpoint_dir) / 'config.yaml')
    print(f"\n✓ MD (réutilisé) + ET (nouveau) + config sauvegardés dans {checkpoint_dir}/")
