from functools import partial
import sys
from pytorch_lightning import seed_everything, Trainer
from pathlib import Path
from loguru import logger

from src.data.data_processing import get_image_encoder
from src.data.data_handler import (
    SampleModel,
    LepusStratifiedKFoldDataModule,
    StratifiedKFoldLoop,
)

from pytorch_lightning.loggers import WandbLogger

LOG_LEVEL = "INFO"
DATA_MANFIEST_PATH = Path("./resources/data.csv")
IMAGE_FOLDER_PATH = Path("/tmp/images")
HEIGHT = 200
WIDTH = 200
SCALE_HEIGHT = False
BATCH_SIZE = 2
NUM_FOLDS = 5
EXPORT_PATH = Path("model_checkpoints")

def bootstrap(
    model=SampleModel(),
    log_level=LOG_LEVEL,
    data_manfiest_path=DATA_MANFIEST_PATH,
    image_folder_path=IMAGE_FOLDER_PATH,
    height=HEIGHT,
    width=WIDTH,
    scale_height=SCALE_HEIGHT,
    batch_size=BATCH_SIZE,
    num_folds=NUM_FOLDS,
    export_path=EXPORT_PATH
):
    export_path.mkdir(exist_ok=True, parents=True)
    seed_everything(42)
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

    wandb_logger = WandbLogger()

    trainer = Trainer(
        max_epochs=10,
        limit_train_batches=2,
        limit_val_batches=2,
        limit_test_batches=2,
        num_sanity_val_steps=0,
        devices=1,
        accelerator="auto",
        strategy="ddp",
        logger=wandb_logger,
    )


    internal_fit_loop = trainer.fit_loop
    trainer.fit_loop = StratifiedKFoldLoop(num_folds, export_path=export_path)
    trainer.fit_loop.connect(internal_fit_loop)
    trainer.fit(model, datamodule)


if __name__ == "__main__":
    bootstrap()
