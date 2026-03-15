# AGENTS.md

## Identity

**lepus-classifier** -- A CNN-based image classifier that distinguishes between two rabbit species (Lepus genus) using ~85 training images, with stratified K-fold cross-validation and Weights & Biases experiment tracking.

## Architecture

| Component | Purpose |
|-----------|---------|
| `src/bootstrap.py` | Entry point. Configures logging, data, model, trainer, and K-fold loop; calls `trainer.fit()`. |
| `src/data/data_handler.py` | Core training infrastructure: `LepusStratifiedKFoldDataModule`, `StratifiedKFoldLoop`, `BaseModel`, `EnsembleVotingModel`, `MetricsCallback`. |
| `src/data/data_extractor.py` | Downloads images from URLs and loads them from disk. |
| `src/data/data_processing.py` | Image encoding/resizing and target label encoding. |
| `src/data/data_types.py` | Type aliases (`Image`, `Label`, `FeaturesEncoder`, `TargetEncoder`). |
| `src/utils/image_check.py` | Image validation utilities. |

The pipeline: `bootstrap()` creates a `LepusStratifiedKFoldDataModule` from a CSV manifest, wraps the Lightning trainer's fit loop with a `StratifiedKFoldLoop`, trains a CNN (`BasicModel` or user-supplied model) across K folds, and ensembles the fold checkpoints via `EnsembleVotingModel`. All metrics are logged to W&B.

## Key Files

| File | Description |
|------|-------------|
| `src/bootstrap.py` | Main entry point and default `BasicModel` CNN definition. |
| `src/data/data_handler.py` | Data modules, K-fold loop, base model, ensemble, and metrics callback. |
| `src/data/data_processing.py` | Image and label preprocessing. |
| `src/data/data_types.py` | Shared type definitions. |
| `resources/data.csv` | Dataset manifest (image URLs and labels). |
| `notebooks/` | Experiment notebooks (e.g. Colab bootstrap example). |
| `pyproject.toml` | Pylint configuration. |
| `requirements.txt` | Top-level requirements (prod + dev). |

## Commands

```bash
pip install -r requirements.txt   # install dependencies
wandb login                       # authenticate W&B
python -m src.bootstrap           # train the default model
```

## Code Style

- **Linting**: `pylint` (configured in `pyproject.toml`, max line length 88).
- **Formatting**: `black` (listed in dev requirements).
- **Commit convention**: Angular conventional commits (see `sr.yaml`).
- **Experiment tracking**: Weights & Biases.
