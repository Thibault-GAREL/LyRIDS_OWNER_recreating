# CLAUDE.md — LyRIDS OWNER

Document de pilotage du projet pour Claude Code. À lire **avant** toute intervention sur le repo.

---

## 🎯 Vue d'ensemble

Réimplémentation pédagogique du pipeline **OWNER** (Toward Unsupervised Open-World Named Entity Recognition).

Pipeline en deux étages :
1. **Mention Detection (MD)** — DistilBERT + tagging BIO pour détecter les positions des entités.
2. **Entity Typing (ET)** — BERT + prompt à `[MASK]` → embeddings, raffinés par Triplet Margin Loss, puis clusterisés via K-means + sélection auto de `k` par BIC.

Évaluation end-to-end via **AMI / ARI** entre les types prédits (clusters) et la vérité terrain.

Référence papier : [`peportier.me/publications/2025_GENEST_OWNER`](https://peportier.me/publications/2025_GENEST_OWNER__Toward_Unsupervised_Open-World_Named_Entity_Recognition.pdf)
Repo de référence (officiel) : [`alteca/OWNER`](https://github.com/alteca/OWNER) — utile comme source d'inspiration mais à **ne pas recopier** tel quel.

---

## 🛠️ Conventions de travail

### Environnement Python

- **Venv obligatoire** : `pytorch_cuda_env` (PyTorch 2.5.1+cu121, transformers, mlflow 3.10.1, pyyaml 6.0.3).
  ```powershell
  & c:\0-Code_py_temp\pytorch_cuda_env\Scripts\Activate.ps1
  ```
- Ne **jamais** `pip install` sans demande explicite de l'utilisateur. Les libs déjà présentes sont listées dans le skill `thibault-pc-config`.
- `langdetect` n'est **pas** installé → préférer une heuristique simple (cf. `ASCII_ONLY` dans `data/1-raw/pile_ner.py`).

### CUDA

GPU ~6 Go VRAM (GTX 1660 Ti Max-Q). Toujours :
```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```
Si CUDA OOM : baisser `batch_size` dans le YAML (jamais en hardcodant dans le `.py`).

### Configuration

- **Tous les hyperparams** vivent dans `configs/*.yaml`. Ne **jamais** hardcoder dans les trainers ou les tests.
- Un dataset = une config dédiée (ex. `ner_default.yaml` pour CoNLL, `pile_ner.yaml` pour Pile-NER).
- Loader : [`src/utils/config.py`](src/utils/config.py) (`load_config`, `flatten_for_mlflow`).

### MLflow

- **Tout** entraînement passe par MLflow. Tracking : file backend par défaut (`mlruns/`).
- Architecture des runs : 1 parent (`owner_ner_full` ou équivalent) + 2 nested (`mention_detection`, `entity_typing`).
- Helper : [`src/utils/mlflow_helpers.py`](src/utils/mlflow_helpers.py). Toujours utiliser `mlflow_run()` (gère le nesting auto) + les variantes `*_safe()` (no-op si pas de run actif → reste utilisable hors pipeline complet).
- Visualisation : `mlflow ui` puis [`http://localhost:5000`](http://localhost:5000).
- ⚠️ Le file backend est déprécié (warning de mlflow 3.10). Migration SQLite à prévoir (cf. roadmap).

### Git

Avant toute modification structurelle (refactor, suppression de fichiers, intégration majeure) :
```powershell
git add -A
git commit -m "checkpoint avant <description>"
git push    # uniquement si demandé explicitement
```
Ne jamais skipper hooks (`--no-verify`) ni amend les commits passés.

### Lancement standard

```powershell
# CoNLL-2003 (par défaut)
python -m tests.test_ner_training

# Avec une config custom
python -m tests.test_ner_training configs/pile_ner.yaml
```

---

## ✅ Ce qui est fait

### Phase 5A — Training + saving (commit `a5d9678`)
- [`src/training/mention_detection.py`](src/training/mention_detection.py), [`entity_typing.py`](src/training/entity_typing.py), [`ner.py`](src/training/ner.py) : trainers complets avec `train`/`evaluate`/`save_model`/`load_model`.
- Modèles : `MdBioModel` (DistilBERT + classif BIO), `EntityEncodingModel` (BERT + `[MASK]` prompt), `AutoKmeans` (BIC).
- Loss ET : `BatchTripletMarginLoss` (mining batch-wise).
- Test end-to-end : [`tests/test_ner_training.py`](tests/test_ner_training.py).

### Phase 5C — MLflow tracking (commit `caf995d`)
- [`src/utils/mlflow_helpers.py`](src/utils/mlflow_helpers.py) : `mlflow_run` (nested auto), `log_params_safe`, `log_metrics_safe`, `log_artifacts_safe`.
- MD/ET trainers : log des hyperparams, batch loss, epoch loss, métriques de validation. Checkpoints loggés via `save_model`.
- NER trainer : log des métriques end-to-end (AMI/ARI/n_truth/n_pred) dans le run parent.
- Smoke test validé : 3 runs (1 parent + 2 nested) créés correctement.

### Phase 5D — Config YAML (commit `caf995d`)
- [`configs/ner_default.yaml`](configs/ner_default.yaml) : config par défaut CoNLL.
- [`src/utils/config.py`](src/utils/config.py) : loader + flatten.
- [`tests/test_ner_training.py`](tests/test_ner_training.py) : lit la config (CLI arg ou défaut), log la config flattenée + le YAML lui-même comme artefact.

### Phase 5E — Polish (commit `caf995d`)
- README mis à jour : features, structure, install (mlflow + pyyaml), `mlflow ui`, statut.
- `.gitkeep` restaurés dans `outputs/models/` et `outputs/results/`.

### Pile-NER — Préparation (non commité)
- Script standalone : [`data/1-raw/pile_ner.py`](data/1-raw/pile_ner.py)
  - Download `Universal-NER/Pile-NER-type` depuis HF.
  - Parse format conversationnel → format OWNER.
  - Sous-échantillonnage configurable (`SAMPLE_SIZE = 10_000` par défaut).
  - Train/test split reproductible (seed=42, ratio=0.10).
  - Heuristique ASCII pour filter EN sans `langdetect`.
- Config dédiée : [`configs/pile_ner.yaml`](configs/pile_ner.yaml) (`k_max=100`, `k_step=5`).
- Smoke test parser validé sur exemple synthétique.
- ⚠️ **Pas encore validé en réel** (download + entraînement complets).

---

## 🚧 Ce qui reste à faire

### Court terme

1. **Valider Pile-NER end-to-end**
   ```powershell
   python data/1-raw/pile_ner.py
   python -m tests.test_ner_training configs/pile_ner.yaml
   mlflow ui
   ```
   Ajuster `SAMPLE_SIZE` / `k_max` selon les premiers résultats.

2. **Commit Pile-NER** (script + config) une fois la validation OK.

3. **Push GitHub** des phases 5C/5D/5E + Pile-NER.

### Moyen terme — Évaluation cross-domain (objectif principal)

Reproduire la **section évaluation** du paper OWNER : entraîner sur CoNLL et/ou Pile-NER, puis **évaluer** (sans réentraîner) sur 13 datasets domain-specific.

**Datasets cibles du paper** :

| Dataset      | Domaine                                  | Source              | Statut |
|--------------|------------------------------------------|---------------------|--------|
| CrossNER × 5 | AI, Literature, Music, Politics, Science | HF                  | TODO   |
| MIT Movie    | Reviews                                  | HF                  | TODO   |
| MIT Restaurant | Search queries                         | HF                  | TODO   |
| FabNER       | Physique / chimie                        | HF                  | TODO   |
| GENIA        | Biomédical (PubMed)                      | non-HF / restricted | TODO   |
| i2b2         | Biomédical clinique                      | licence requise ⚠️  | TODO   |
| GENTLE       | Texte non-conventionnel                  | HF / ext            | TODO   |
| GUM          | Texte non-conventionnel                  | HF                  | TODO   |
| WNUT 17      | Réseaux sociaux                          | HF                  | TODO   |

**Architecture suggérée** :

1. **Nouveau script de test** : `tests/test_cross_domain_eval.py`
   - Charge un checkpoint (`ner.load_model(folder)`).
   - Pour chaque dataset cible : load `test.json` au format OWNER, run `ner.evaluate()`.
   - Log les résultats dans MLflow (1 run par couple `(checkpoint, dataset_eval)`).
   - Génère un tableau récapitulatif AMI/ARI.

2. **Preprocessors dataset par dataset** dans `data/1-raw/{name}.py` :
   - Pour les datasets HF avec format `tokens` + `ner_tags` standard (CrossNER, WNUT, MIT) : réutiliser [`HuggingFaceNerPreprocessor`](src/data/preprocessing/huggingface.py).
   - Pour les formats spéciaux (FabNER ?, GUM, GENTLE) : preprocessor custom dans le script (pattern `pile_ner.py`).
   - Pour les datasets restreints (i2b2) : noter la procédure d'accès dans le script, ne pas committer les data.

3. **Configs d'évaluation** : `configs/eval_{dataset}.yaml` (uniquement la section `data` + `output`, pas besoin de `mention_detection`/`entity_typing` puisqu'on n'entraîne pas).

4. **Tableau de résultats** comparant aux chiffres du paper (App. E).

### Long terme

- **Migration MLflow vers SQLite** (file backend déprécié en 02/2026) :
  ```python
  mlflow.set_tracking_uri('sqlite:///mlflow.db')
  ```
- **Hyperparameter sweep** (Optuna ?) pour `k_max`, `margin`, `learning_rate`.
- **Tester d'autres PLMs** : DeBERTa-v3 (déjà mentionné comme défaut dans `MentionDetectionTrainer`), RoBERTa.
- **Documentation** : tutoriel notebook qui montre le pipeline étape par étape.

---

## 📂 Structure du repo

```
LyRIDS_OWNER/
├── src/
│   ├── data/                  # schemas, datasets PyTorch, preprocessing, sérialisation
│   ├── models/                # MdBioModel, EntityEncodingModel, AutoKmeans
│   ├── training/              # MD / ET / NER trainers
│   ├── evaluation/            # AMI/ARI, alignement entités
│   └── utils/
│       ├── config.py          # YAML loader
│       ├── mlflow_helpers.py  # nested-run + safe logs
│       └── pytorch.py
├── configs/
│   ├── ner_default.yaml       # CoNLL-2003
│   └── pile_ner.yaml          # Pile-NER
├── data/
│   ├── 1-raw/                 # scripts download/preprocess
│   ├── 2-processed/           # format OWNER (.json)
│   └── 3-external/
├── outputs/
│   ├── models/                # checkpoints
│   ├── logs/
│   └── results/               # logs textuels
├── tests/
│   └── test_ner_training.py
├── mlruns/                    # MLflow (git-ignored)
├── README.md
└── CLAUDE.md                  # ← ce fichier
```

---

## 🧠 Décisions de design à connaître

- **Pas de validation split** dans le pipeline actuel : on valide directement sur le test à chaque epoch (acceptable vu la taille des datasets, à reconsidérer pour l'évaluation cross-domain).
- **`save_model` log les artefacts dans le run actif** : si appelé après que les sous-runs MD/ET sont fermés, ça atterrit dans le parent — c'est intentionnel.
- **Triplet Loss en mode "all valid triplets"** : pas de hard mining. Suffisant en pratique sur CoNLL ; à revoir pour Pile-NER (plus de types = plus de triplets).
- **AutoKmeans utilise BIC**, pas la silhouette ni le coude. Le `k_step` contrôle la granularité (et donc le coût).

---

## ⚠️ Pièges connus

- **Inner vs outer git repo** : ce dossier (`LyRIDS_OWNER/LyRIDS_OWNER/`) a son propre `.git` (remote `LyRIDS_OWNER_recreation`). Le dossier parent est aussi un repo. Toujours vérifier `pwd` avant un commit.
- **Pylance signale `mlflow` import error** : faux positif, l'IDE n'utilise pas `pytorch_cuda_env`. À ignorer.
- **CRLF warnings** : Windows + Git → normal, pas un problème.
- **`outputs/logs/`** : présent pour rétro-compat mais inutilisé (MLflow écrit dans `mlruns/`).
