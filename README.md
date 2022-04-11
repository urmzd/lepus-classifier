# Lepus Classifier

This project aims to distinguish between the Eurpoean Hare and the Eastern Cotton-tail Rabbit class
using a relatively small set of images retrieved online.

![Mascot](./assets/mascot.webp)

## Table of Contents

- [Data](#data)
- [Quickstart](#quickstart)
- [Proposals](#proposals)
- [Report](#report)
- [Poster](#poster)

## Data

Data was retrieved and can be found under [resources/data.txt](resources/data.txt)

## Proposals

All the initial proposals can be found under [resources/proposals](docs/proposals/).

## Report

The final report can be found under [docs/report-docs/lepus-classifier.pdf](docs/report-docs/lepus-classifier_report.pdf)

## Quickstart

1. Install dependencies.

```bash
  pip install -r requirements.txt
```

2. Login to your Weights and Biases account using your API key.

```bash
  wandb login
```

3. Use the bootstrap function to start training your custom model.

```python
import src.bootstrap import bootstrap

class CustomModel(BaseModel):
  def __init__(self):
    super().__init__()
    self.layers = torch.nn.Sequential(
      ...
    )
    super().__post_init__()

  def forward(self, x)

bootstrap(model=CustomModel())
```

### Google Collab

Take a look at [notebooks/example_bootstrap.ipynb](notebooks/example_bootstrap.ipynb)
