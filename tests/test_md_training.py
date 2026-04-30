"""Test de bout en bout de Phase 3 : entraîner et évaluer Mention Detection."""

# python -m tests.test_md_training

from src.training.mention_detection import MentionDetectionTrainer

trainer = MentionDetectionTrainer(
    train_path="data/2-processed/conll2003_mini/train.json",
    test_path="data/2-processed/conll2003_mini/test.json",
    plm_name="distilbert-base-uncased",  # plus léger que deberta pour ton GTX 1660 Ti
    max_len=64,
    batch_size=4,
    num_epochs=10,  # mini dataset → faut sur-fitter pour voir un signal
    learning_rate=5e-5,
)

trainer.load_data(training=True)
print(f"Train : {len(trainer.train_dataset)} phrases")
print(f"Test  : {len(trainer.test_dataset)} phrases")

trainer.train()

# Évaluation finale
final = trainer.evaluate()
print(
    f'\nFinal : F1={final["f1"]:.4f} P={final["precision"]:.4f} R={final["recall"]:.4f}'
)
