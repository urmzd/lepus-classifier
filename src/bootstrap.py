from dataclasses import dataclass
from functools import partial
import sys
from pytorch_lightning import seed_everything, Trainer
from pathlib import Path
from loguru import logger

from src.data.data_processing import get_image_encoder
from src.data.data_handler import (
    BasicModel,
    LepusStratifiedKFoldDataModule,
    StratifiedKFoldLoop,
)
from typing import Optional

from pytorch_lightning.loggers import WandbLogger
import wandb


@dataclass
class TrainerFactory:
    strategy: str = "ddp"
    max_epochs: int = 10
    devices: int = 1
    internal_logger: Optional[WandbLogger] = None

    def get_trainer(self):
        if self.internal_logger:
            trainer = Trainer(
                max_epochs=self.max_epochs,
                limit_train_batches=None,
                limit_val_batches=None,
                limit_test_batches=None,
                num_sanity_val_steps=0,
                devices=self.devices,
                accelerator="auto",
                strategy=self.strategy,
                logger=self.internal_logger,
            )

            return trainer
        else:
            raise Exception("Must specify logging entity.")


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


def bootstrap(
    model=BasicModel(learning_rate=LEARNING_RATE),
    log_level=LOG_LEVEL,
    data_manfiest_path=DATA_MANFIEST_PATH,
    image_folder_path=IMAGE_FOLDER_PATH,
    height=HEIGHT,
    width=WIDTH,
    scale_height=SCALE_HEIGHT,
    batch_size=BATCH_SIZE,
    num_folds=NUM_FOLDS,
    export_path=EXPORT_PATH,
    wandb_logger: WandbLogger = WandbLogger(project="rabbit-classifier"),
    trainer_factory: TrainerFactory = TrainerFactory(),
    seed_no: Optional[int] = 42,
):
    export_path.mkdir(exist_ok=True, parents=True)

    if seed_no:
        seed_everything(seed_no)

    # System logger.
    logger.remove()
    logger.add(sys.stderr, level=log_level)

    trainer_factory.internal_logger = wandb_logger
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
    wandb.finish()


if __name__ == "__main__":
    bootstrap()
