# Lepus Classifier

<img src="./assets/mascot.webp" alt="mascot" width="200" style="float:left">

> State-of-the-art image classifiers are typically training on hundreds of thousands of images and require extensive computing power. In this report, we examine methods to improve performance of a CNN without the need for large data sets and specialized hardware. Using 85 images of two species from the Lepus genus, we demonstrate that optimal image classifier architectures are still limited by the quantity of data they are trained with, especially when images have highly complex feature sets.

## Table of Contents

- [Quickstart](#quickstart)
- [Proposals](#proposals)
- [Report](#report)
- [Poster](#poster)
- [Data](#data)
- [Interactive Notebooks](#interactive-notebooks)
- [Experiment Tracking & Logs](#experiement-tracking-and-logs)

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

## Proposals

All the initial proposals can be found under [resources/proposals](docs/proposals/).

## Report

The final report can be found under [docs/report-docs/lepus-classifier.pdf](docs/report-docs/lepus-classifier_report.pdf)

## Poster

The final project poster can be found under [docs/poster.pdf](docs/poster.pdf)

## Data

Data was retrieved and can be found under [resources/data.txt](resources/data.txt)

## Interactive Notebooks

All interactie notebooks, including experiments done can be found under the [notebooks folder](/notebooks/).

## Experiment Tracking and Logs

Our experimental logs can be found at our [Weights & Biases project](https://wandb.ai/csci-4155-rabbit-classifier).

## Dependencies

All dependencies can be installed using the following snippet:

```bash
pip install -r requirements.txt
```

