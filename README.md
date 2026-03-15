# Lepus Classifier

<img src="./assets/mascot.webp" alt="mascot" width="200">

A CNN-based image classifier that distinguishes between two rabbit species (*Lepus* genus) using only ~85 training images. Built to explore how well small datasets can work with modern deep learning techniques.

## Quickstart

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Login to [Weights & Biases](https://wandb.ai) for experiment tracking:

```bash
wandb login
```

3. Train a model:

```python
from src.bootstrap import bootstrap

bootstrap()
```

Or define a custom architecture:

```python
from src.bootstrap import bootstrap
from src.data.model import BaseModel
import torch

class CustomModel(BaseModel):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.Sequential(...)
        super().__post_init__()

    def forward(self, x):
        return self.layers(x)

bootstrap(model=CustomModel())
```

See [notebooks/example_bootstrap.ipynb](notebooks/example_bootstrap.ipynb) for a Google Colab example.

## Project Structure

| Path | Description |
|------|-------------|
| [`src/`](src/) | Training pipeline, data loading, and model definitions |
| [`notebooks/`](notebooks/) | Experiment notebooks and usage examples |
| [`resources/data.csv`](resources/data.csv) | Dataset manifest (image URLs and labels) |
| [`docs/report-docs/lepus-classifier-report.pdf`](docs/report-docs/lepus-classifier-report.pdf) | Final research report |
| [`docs/poster.pdf`](docs/poster.pdf) | Project poster |
| [`docs/proposals/`](docs/proposals/) | Initial project proposals |

## Experiment Logs

Tracked with Weights & Biases: [csci-4155-rabbit-classifier](https://wandb.ai/csci-4155-rabbit-classifier)

## License

[Apache 2.0](LICENSE)
