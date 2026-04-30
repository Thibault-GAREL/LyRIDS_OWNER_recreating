"""Test de bout en bout de Phase 4 : entraîner et évaluer Entity Typing."""

from src.training.entity_typing import EntityTypingTrainer

trainer = EntityTypingTrainer(
    train_path="data/2-processed/conll2003_mini/train.json",
    test_path="data/2-processed/conll2003_mini/test.json",
    plm_name="bert-base-uncased",  # BERT a un [MASK], distilbert aussi
    max_len=64,
    batch_size=4,
    num_epochs=10,
    learning_rate=5e-5,
    margin=1.0,
    k_min=2,
    k_max=4,  # mini-dataset = 4 types max (ORG, MISC, PER, LOC)
    k_step=1,
    seed=42,
)

trainer.load_data(training=True)
print(f"Train : {len(trainer.train_dataset)} entités")
print(f"Test  : {len(trainer.test_dataset)} entités")
print(f"Types : {trainer.train_dataset.entity_type_to_id}")

trainer.train()

final = trainer.evaluate()
print(
    f'\nFinal : AMI={final["ami"]:.4f} ARI={final["ari"]:.4f} '
    f'k_trouvé={final["k"]} (vrai={final["true_k"]})'
)
