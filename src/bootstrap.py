import sys
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import List, Optional, Union

from loguru import logger
from pytorch_lightning import Callback, Trainer, seed_everything
from pytorch_lightning.loggers import WandbLogger
from src.data.data_handler import (
    BasicModel,
    LepusStratifiedKFoldDataModule,
    MetricsCallback,
    StratifiedKFoldLoop,
)
from src.data.data_processing import get_image_encoder

LOG_LEVEL = "INFO"
DATA_MANIFEST_PATH = Path("./resources/data.csv")
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
    logger: Optional[WandbLogger] = None
    callbacks: List[Callback] = field(default_factory=get_default_callbacks)
    strategy: str = "ddp"
    max_epochs: int = 10
    devices: Union[List[int], str, None] = "auto"
    deterministic: bool = True
    project_name = PROJECT_NAME
    run_name: Optional[str] = RUN_NAME

    def __post_init__(self):
        self.logger = WandbLogger(project=PROJECT_NAME, log_model="all")

    def get_trainer(self, **kwargs):
        if self.logger is not None:
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

        raise ValueError("Expected self.logger to exist.")


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
        data_manifest_path=data_manifest_path,
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
