#!/usr/bin/env python

import sys
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import List, Optional, Union

from loguru import logger
from pytorch_lightning import Callback, Trainer, seed_everything
from pytorch_lightning.loggers import WandbLogger
import torch

import wandb
from src.data.data_handler import (
    BaseModel,
    LepusStratifiedKFoldDataModule,
    MetricsCallback,
    StratifiedKFoldLoop,
)
from src.data.data_processing import get_image_encoder
import math

class AdaGradModel(BaseModel):
  def __init__(self, n_targets=2, learning_rate=0.02):
      super().__init__(n_targets=n_targets, learning_rate=learning_rate)
      self.layer1 = torch.nn.Conv2d(1, 20, 2, 2)
      self.layer2= torch.nn.MaxPool2d(2, 2)
      self.layer3=torch.nn.Conv2d(20, 200, 2)
      self.layer4=torch.nn.MaxPool2d(2, 2)
      self.layer5=torch.nn.Flatten(start_dim=1)
      self.layer6=torch.nn.ReLU()
      self.layer7=torch.nn.Linear(int(math.pow(24,2)) * 200, 2)
      self.layer8=torch.nn.LogSoftmax()
      super().__post_init__()

  def forward(self, x):
      res_1 = self.layer1(x)
      res_2 = self.layer2(res_1)
      res_3 = self.layer3(res_2)
      res_4 = self.layer4(res_3)
      res_5 = self.layer5(res_4)
      res_6 = self.layer6(res_5)
      res_7 = self.layer7(res_6)
      res_8 = self.layer8(res_7)
      return res_8

  def configure_optimizers(self):
      return torch.optim.Adagrad(self.parameters(), self.learning_rate)

LOG_LEVEL = "INFO"
DATA_MANFIEST_PATH = Path("./resources/data.csv")
IMAGE_FOLDER_PATH = Path("/tmp/images")
HEIGHT = 200
WIDTH = 200
SCALE_HEIGHT = False
BATCH_SIZE = 2
NUM_FOLDS = 5
EXPORT_PATH = Path("model_checkpoints")
LEARNING_RATE = 0.02
N_CLASSES = 2
SEED_NO: Optional[int] = 42
PROJECT_NAME = "rabbit-classifier"
RUN_NAME = None


def get_default_callbacks() -> List[Callback]:
    return [MetricsCallback(n_targets=N_CLASSES)]


@dataclass
class TrainerFactory:
    logger: WandbLogger = WandbLogger(project=PROJECT_NAME, log_model="all")
    callbacks: List[Callback] = field(default_factory=get_default_callbacks)
    strategy: str = "ddp"
    max_epochs: int = 10
    devices: Union[List[int], str, None] = "auto"
    deterministic: bool = True
    project_name = PROJECT_NAME
    run_name: Optional[str] = RUN_NAME

    def get_trainer(self, **kwargs):
        trainer = Trainer(
            max_epochs=self.max_epochs,
            limit_train_batches=None,
            limit_val_batches=None,
            limit_test_batches=None,
            num_sanity_val_steps=0,
            devices=self.devices,
            accelerator="auto",
            strategy=self.strategy,
            logger=self.logger,
            callbacks=self.callbacks,
            deterministic=self.deterministic,
            **kwargs
        )
        return trainer


def bootstrap(
    model=AdaGradModel(learning_rate=LEARNING_RATE),
    log_level=LOG_LEVEL,
    data_manfiest_path=DATA_MANFIEST_PATH,
    image_folder_path=IMAGE_FOLDER_PATH,
    height=HEIGHT,
    width=WIDTH,
    scale_height=SCALE_HEIGHT,
    batch_size=BATCH_SIZE,
    num_folds=NUM_FOLDS,
    export_path=EXPORT_PATH,
    seed_no=SEED_NO,
    trainer_factory: TrainerFactory = TrainerFactory(),
):
    export_path.mkdir(exist_ok=True, parents=True)
    image_folder_path.mkdir(exist_ok=True, parents=True)

    if seed_no:
        seed_everything(seed_no, workers=True)

    # System logger.
    logger.remove()
    logger.add(sys.stderr, level=log_level)

    x_encoder = partial(
        get_image_encoder(
            desired_height=height, desired_width=width, scale_height=scale_height
        )
    )

    datamodule = LepusStratifiedKFoldDataModule(
        data_manifest_path=data_manfiest_path,
        image_folder_path=image_folder_path,
        transform_features=x_encoder,
        batch_size=batch_size,
        n_splits=num_folds,
    )

    trainer = trainer_factory.get_trainer()

    internal_fit_loop = trainer.fit_loop
    trainer.fit_loop = StratifiedKFoldLoop(num_folds, export_path=export_path)
    trainer.fit_loop.connect(internal_fit_loop)
    trainer.fit(model, datamodule)

    trainer.logger.finalize("success")


if __name__ == "__main__":
    bootstrap()
