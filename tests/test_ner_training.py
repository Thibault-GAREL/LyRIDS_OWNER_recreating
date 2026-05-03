"""Test pipeline NER end-to-end (Phase 5 A) sur le vrai CoNLL-2003 noMISC.

La config est lue depuis configs/ner_default.yaml.
Hyperparams adaptés à un GPU ~6 Go VRAM (GTX 1660 Ti).
Si CUDA out of memory, baisse batch_size dans le YAML.
"""

# python -m tests.test_ner_training
# python -m tests.test_ner_training configs/ner_default.yaml

import sys

import mlflow

from src.training.mention_detection import MentionDetectionTrainer
from src.training.entity_typing import EntityTypingTrainer
from src.training.ner import NerTrainer
from src.utils.config import flatten_for_mlflow, load_config

config_path = sys.argv[1] if len(sys.argv) > 1 else 'configs/ner_default.yaml'
config = load_config(config_path)

md = MentionDetectionTrainer(
    train_path=config['data']['train_path'],
    test_path=config['data']['test_path'],
    **config['mention_detection'],
)

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
    ner.load_data(training=True)
    ner.train()

    metrics = ner.evaluate()
    print("\n========== NER end-to-end ==========")
    print(f'AMI={metrics["ami"]:.4f}  ARI={metrics["ari"]:.4f}')
    print(f'n_truth={metrics["n_truth"]}  n_pred_md={metrics["n_pred"]}')

    ner.save_model(config['output']['checkpoint_dir'])
    print(f"\n✓ Modèles sauvegardés dans {config['output']['checkpoint_dir']}/")
