import sys
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import lightning as L
import lightning.pytorch.strategies as strategies
import torch
from lightning.pytorch.loggers import WandbLogger
from lightning_kfold import KFoldTrainer
from loguru import logger
from torch.nn import functional as F

import wandb
from src.data.data_handler import BaseModel, LepusStratifiedKFoldDataModule, MetricsCallback
from src.data.data_processing import get_image_encoder

LOG_LEVEL = "INFO"
DATA_MANIFEST_PATH = Path("./resources/data.csv")
IMAGE_FOLDER_PATH = Path("/tmp/images")
HEIGHT = 200
WIDTH = 200
SCALE_HEIGHT = False
BATCH_SIZE = 2
NUM_FOLDS = 3
EXPORT_PATH = Path("model_checkpoints")
LEARNING_RATE = 1e-5
N_CLASSES = 2
SEED_NO: Optional[int] = 42
PROJECT_NAME = "rabbit-classifier"
RUN_NAME = "BasicModel"


def get_default_callbacks() -> List[L.Callback]:
    return [MetricsCallback(n_targets=N_CLASSES)]


@dataclass
class TrainerFactory:
    logger: Optional[WandbLogger] = None
    callbacks: List[L.Callback] = field(default_factory=get_default_callbacks)
    strategy: Optional[Union[str, strategies.Strategy]] = "single_device"
    max_epochs: int = 10
    devices: Union[List[int], str, None] = "auto"
    deterministic: bool = True
    project_name = PROJECT_NAME
    run_name: Optional[str] = RUN_NAME

    def get_trainer_kwargs(
        self, logger_kwargs: Dict[str, Any], trainer_kwargs: Dict[str, Any]
    ) -> Dict[str, Any]:
        self.logger = WandbLogger(
            **dict(project=PROJECT_NAME, log_model="all", **logger_kwargs)
        )
        return dict(
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
            **trainer_kwargs,
        )


class BasicModel(BaseModel):
    def __init__(self, n_targets=2, learning_rate=0.02) -> None:
        super().__init__(n_targets=n_targets, learning_rate=learning_rate)

        self.layer_1 = torch.nn.Conv2d(1, 15, 2, 2)
        self.layer_2 = torch.nn.MaxPool2d(2, 2)
        self.layer_3 = torch.nn.ReLU()
        self.layer_4 = torch.nn.Flatten(1, -1)
        self.layer_5 = torch.nn.Linear(15 * 50 * 50, n_targets)
        self.softmax_layer = torch.nn.LogSoftmax(dim=1)

        super().__post_init__()

    def forward(self, x):
        x_1 = self.layer_1(x)
        x_2 = self.layer_2(x_1)
        x_3 = self.layer_3(x_2)
        x_4 = self.layer_4(x_3)
        x_5 = self.layer_5(x_4)
        result = self.softmax_layer(x_5)
        return result


@logger.catch
def bootstrap(
    model=BasicModel(learning_rate=LEARNING_RATE),
    log_level=LOG_LEVEL,
    data_manifest_path=DATA_MANIFEST_PATH,
    image_folder_path=IMAGE_FOLDER_PATH,
    height=HEIGHT,
    width=WIDTH,
    scale_height=SCALE_HEIGHT,
    batch_size=BATCH_SIZE,
    num_folds=NUM_FOLDS,
    export_path=EXPORT_PATH,
    seed_no=SEED_NO,
    trainer_factory: TrainerFactory = TrainerFactory(),
    logger_kwargs: Dict[str, Any] = dict(),
    trainer_kwargs: Dict[str, Any] = dict(),
):
    export_path.mkdir(exist_ok=True, parents=True)
    image_folder_path.mkdir(exist_ok=True, parents=True)

    if seed_no:
        L.seed_everything(seed_no, workers=True)

    # System logger.
    logger.remove()
    logger.add(sys.stderr, level=log_level)

    x_encoder = partial(
        get_image_encoder(
            desired_height=height, desired_width=width, scale_height=scale_height
        )
    )

    datamodule = LepusStratifiedKFoldDataModule(
        data_manifest_path=data_manifest_path,
        image_folder_path=image_folder_path,
        transform_features=x_encoder,
        batch_size=batch_size,
        num_folds=num_folds,
    )

    lt_trainer_kwargs = trainer_factory.get_trainer_kwargs(logger_kwargs, trainer_kwargs)
    wandb_logger = trainer_factory.logger

    wandb_logger.watch(model, log_freq=50)

    kfold_trainer = KFoldTrainer(
        num_folds=num_folds,
        export_path=export_path,
        loss_fn=F.nll_loss,
        **lt_trainer_kwargs,
    )
    kfold_trainer.fit(model, datamodule)

    wandb.finish()


if __name__ == "__main__":
    bootstrap()
