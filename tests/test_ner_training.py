"""Test pipeline NER end-to-end (Phase 5 A) sur le vrai CoNLL-2003 noMISC.

Hyperparams adaptés à un GPU ~6 Go VRAM (GTX 1660 Ti).
Si CUDA out of memory, baisse batch_size à 4 ou 2.
"""

from src.training.mention_detection import MentionDetectionTrainer
from src.training.entity_typing import EntityTypingTrainer
from src.training.ner import NerTrainer

md = MentionDetectionTrainer(
    train_path="data/2-processed/conll2003_noMISC/train.json",
    test_path="data/2-processed/conll2003_noMISC/test.json",
    plm_name="distilbert-base-uncased",   # ~250 Mo, OK pour 6 Go VRAM
    max_len=128,
    batch_size=8,
    num_epochs=2,                          # 2 epochs suffisent sur 14k phrases
    learning_rate=2e-5,
)

et = EntityTypingTrainer(
    train_path="data/2-processed/conll2003_noMISC/train.json",
    test_path="data/2-processed/conll2003_noMISC/test.json",
    plm_name="bert-base-uncased",          # ~440 Mo, BERT a un [MASK]
    max_len=128,
    batch_size=8,
    num_epochs=2,
    learning_rate=2e-5,
    margin=1.0,
    k_min=2,
    k_max=10,                              # vrai k = 3 (loc/org/per), on cherche dans [2, 10]
    k_step=1,
    seed=42,
)

ner = NerTrainer(md, et)
ner.load_data(training=True)
ner.train()

metrics = ner.evaluate()
print("\n========== NER end-to-end ==========")
print(f'AMI={metrics["ami"]:.4f}  ARI={metrics["ari"]:.4f}')
print(f'n_truth={metrics["n_truth"]}  n_pred_md={metrics["n_pred"]}')

ner.save_model('outputs/models/ner_checkpoint')
print("\n✓ Modèles sauvegardés dans outputs/models/ner_checkpoint/")
