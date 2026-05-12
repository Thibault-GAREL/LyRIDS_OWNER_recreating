"""Évaluation cross-domain : applique 1+ checkpoints OWNER sur les 13 datasets
de comparaison du paper, et produit un rapport markdown sous forme de tableau
(datasets en lignes, modèles en colonnes).

Pré-requis :
  - Chaque checkpoint a un `config.yaml` à côté de ses .pt
    (généré par test_ner_training.py / test_ner_training_et_only.py).
    Pour les checkpoints anciens, copie la config à la main.
  - Chaque dataset cible a son test.json dans data/2-processed/{name}/
    (généré par les scripts data/1-raw/{name}.py).
  - Les datasets manquants apparaissent comme "N/A" dans le tableau.

Lancement :
    python -m tests.test_cross_domain_eval
"""
from datetime import date
from pathlib import Path

from src.training.entity_typing import EntityTypingTrainer
from src.training.mention_detection import MentionDetectionTrainer
from src.training.ner import NerTrainer
from src.utils.config import load_config


# Checkpoints à évaluer — nom affiché + dossier
CHECKPOINTS: list[tuple[str, str]] = [
    ('CoNLL',             'outputs/models/ner_checkpoint'),
    ('Pile-NER ET-only',  'outputs/models/pile_ner_et_only_checkpoint'),
]

# Datasets de comparaison du paper — nom affiché + dossier OWNER format
DATASETS: dict[str, str] = {
    'CrossNER-AI':        'data/2-processed/crossner_ai',
    'CrossNER-Literature':'data/2-processed/crossner_literature',
    'CrossNER-Music':     'data/2-processed/crossner_music',
    'CrossNER-Politics':  'data/2-processed/crossner_politics',
    'CrossNER-Science':   'data/2-processed/crossner_science',
    'MIT Movie':          'data/2-processed/mit_movie',
    'MIT Restaurant':     'data/2-processed/mit_restaurant',
    'FabNER':             'data/2-processed/fabner',
    'GENIA':              'data/2-processed/genia',
    'i2b2':               'data/2-processed/i2b2',
    'GENTLE':             'data/2-processed/gentle',
    'GUM':                'data/2-processed/gum',
    'WNUT 17':            'data/2-processed/wnut17',
}

# Surcharge AutoKmeans pour l'éval : le k_max d'entraînement est tuné pour le
# dataset source (3 types pour CoNLL → k_max=10), trop bas pour des datasets
# cibles avec 5-15 types. On élargit l'intervalle pour donner sa chance au
# clustering. Mets ces 3 valeurs à None pour conserver la config d'entraînement.
ET_K_MIN_OVERRIDE: int | None = 2
ET_K_MAX_OVERRIDE: int | None = 20
thibaET_K_STEP_OVERRIDE: int | None = 2


def evaluate_one(checkpoint_dir: str, test_path: str) -> dict:
    """Charge un checkpoint et évalue sur un test.json OWNER format."""
    config_path = Path(checkpoint_dir) / 'config.yaml'
    if not config_path.exists():
        raise FileNotFoundError(
            f"Pas de config.yaml dans {checkpoint_dir}. Pour un checkpoint "
            f"ancien, copie le YAML d'entraînement correspondant ici "
            f"(ex: cp configs/ner_default.yaml {checkpoint_dir}/config.yaml)."
        )
    config = load_config(str(config_path))

    md = MentionDetectionTrainer(
        train_path=config['data']['train_path'],   # ignoré (training=False)
        test_path=test_path,
        **config['mention_detection'],
    )

    et_kwargs = dict(config['entity_typing'])
    if ET_K_MIN_OVERRIDE is not None:
        et_kwargs['k_min'] = ET_K_MIN_OVERRIDE
    if ET_K_MAX_OVERRIDE is not None:
        et_kwargs['k_max'] = ET_K_MAX_OVERRIDE
    if ET_K_STEP_OVERRIDE is not None:
        et_kwargs['k_step'] = ET_K_STEP_OVERRIDE
    et = EntityTypingTrainer(
        train_path=config['data']['train_path'],
        test_path=test_path,
        **et_kwargs,
    )

    ner = NerTrainer(md, et)
    ner.load_model(checkpoint_dir)
    md.load_data(training=False)
    et.load_data(training=False)
    return ner.evaluate()


def write_report(path: Path, results: dict, checkpoint_names: list[str]) -> None:
    """Écrit un tableau markdown : datasets en lignes, modèles × {AMI, ARI} en colonnes."""
    lines = [
        '# Évaluation cross-domain — modèles OWNER vs datasets du paper',
        '',
        f'Date : {date.today().isoformat()}',
        '',
        'Métriques :',
        '- **AMI** (Adjusted Mutual Information) — information mutuelle entre clusters prédits et vraies classes (corrigée pour le hasard).',
        '- **ARI** (Adjusted Rand Index) — accord entre les deux partitions (corrigé pour le hasard).',
        '',
        f'AutoKmeans (eval-only) : k_min={ET_K_MIN_OVERRIDE}, k_max={ET_K_MAX_OVERRIDE}, k_step={ET_K_STEP_OVERRIDE}',
        '',
    ]
    header_cells = ['Dataset']
    for c in checkpoint_names:
        header_cells.append(f'{c} AMI')
        header_cells.append(f'{c} ARI')
    lines.append('| ' + ' | '.join(header_cells) + ' |')
    lines.append('|' + '|'.join(['---'] * len(header_cells)) + '|')

    for ds_name in DATASETS:
        row = [ds_name]
        for c in checkpoint_names:
            m = results[c].get(ds_name)
            if m is None:
                row.extend(['N/A', 'N/A'])
            else:
                row.append(f'{m["ami"]:.4f}')
                row.append(f'{m["ari"]:.4f}')
        lines.append('| ' + ' | '.join(row) + ' |')

    lines.append('')
    lines.append('### Détails')
    for c in checkpoint_names:
        n_ok = sum(1 for v in results[c].values() if v is not None)
        n_total = len(DATASETS)
        lines.append(f'- **{c}** : {n_ok}/{n_total} datasets évalués (les "N/A" indiquent des datasets non préparés dans `data/2-processed/`).')

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main():
    results: dict[str, dict[str, dict | None]] = {name: {} for name, _ in CHECKPOINTS}

    for ckpt_name, ckpt_dir in CHECKPOINTS:
        if not Path(ckpt_dir).exists():
            print(f'⚠️  Checkpoint manquant : {ckpt_dir} → skip {ckpt_name}')
            for ds_name in DATASETS:
                results[ckpt_name][ds_name] = None
            continue

        for ds_name, ds_dir in DATASETS.items():
            test_path = f'{ds_dir}/test.json'
            if not Path(test_path).exists():
                print(f'  [{ckpt_name}] {ds_name} : pas de test.json à {test_path} → N/A')
                results[ckpt_name][ds_name] = None
                continue

            print(f'\n========== {ckpt_name} on {ds_name} ==========')
            try:
                metrics = evaluate_one(ckpt_dir, test_path)
                results[ckpt_name][ds_name] = metrics
                print(f'  AMI={metrics["ami"]:.4f}  ARI={metrics["ari"]:.4f}  '
                      f'(n_truth={metrics.get("n_truth", "?")}, n_pred={metrics.get("n_pred", "?")})')
            except Exception as e:
                print(f'  ❌ Erreur : {type(e).__name__}: {e}')
                results[ckpt_name][ds_name] = None

    out_path = Path(f'outputs/results/{date.today().isoformat()}-cross_domain_eval.md')
    write_report(out_path, results, [name for name, _ in CHECKPOINTS])
    print(f'\n✓ Rapport écrit dans {out_path}')


if __name__ == '__main__':
    main()
