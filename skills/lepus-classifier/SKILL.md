# Skill: lepus-classifier

## Description

Work with lepus-classifier -- a CNN-based image classifier for rabbit species (Lepus genus) using PyTorch Lightning with stratified K-fold cross-validation and W&B tracking.

## When to Use

- Defining new CNN architectures by subclassing `BaseModel`
- Modifying the data pipeline (image loading, encoding, augmentation)
- Adjusting K-fold training or ensemble voting logic
- Working with the Weights & Biases integration and metrics callbacks
- Running experiments via `bootstrap()` or Colab notebooks

## Context

- **Language**: Python 3.8+
- **Framework**: PyTorch Lightning 1.6, scikit-learn, W&B
- **Entry point**: `src/bootstrap.py` -- `bootstrap()` function configures everything and trains
- **Model base**: `BaseModel` in `src/data/data_handler.py` (abstract `LightningModule` with NLL loss)
- **Default model**: `BasicModel` in `src/bootstrap.py` (Conv2d -> MaxPool -> ReLU -> Flatten -> Linear -> LogSoftmax)
- **Data flow**: CSV manifest (`resources/data.csv`) -> download images -> `LepusDataset` -> `LepusStratifiedKFoldDataModule` -> `StratifiedKFoldLoop`
- **Ensemble**: `EnsembleVotingModel` averages logits from fold checkpoints
- **Metrics**: `MetricsCallback` tracks accuracy, precision, recall, F1, confusion matrix via torchmetrics + W&B

## Key Commands

```bash
pip install -r requirements.txt   # install dependencies
wandb login                       # authenticate W&B
python -m src.bootstrap           # train default model
```

## Conventions

- Images are resized to 200x200 grayscale by default
- 3-fold cross-validation with train_size=0.8 holdout
- All experiments logged to W&B project `rabbit-classifier`
- Seed set to 42 for reproducibility (`seed_everything`)
- Conventional commits required (see `sr.yaml`)
- Linting via pylint (max line length 88), formatting via black
