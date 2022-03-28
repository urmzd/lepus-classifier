from functools import partial
from pytorch_lightning import seed_everything, Trainer
from sklearn.preprocessing import OneHotEncoder
from pathlib import Path

from src.data.data_processing import get_image_encoder
from src.data.data_handler import (
    SampleModel,
    LepusStratifiedKFoldDataModule,
    StratifiedKFoldLoop,
)

seed_everything(42)

if __name__ == "__main__":
    seed_everything(42)
    DATA_MANFIEST_PATH = Path("../resources/data.csv")
    IMAGE_FOLDER_PATH = Path("/tmp/images")
    HEIGHT = 200
    WIDTH = 200
    SCALE_HEIGHT = False
    model = SampleModel(height=HEIGHT, width=WIDTH)
    x_encoder = partial(
        get_image_encoder(
            desired_height=HEIGHT, desired_width=WIDTH, scale_height=SCALE_HEIGHT
        )
    )
    datamodule = LepusStratifiedKFoldDataModule(
        data_manifest_path=DATA_MANFIEST_PATH,
        image_folder_path=IMAGE_FOLDER_PATH,
        transform_features=x_encoder,
    )
    trainer = Trainer(
        max_epochs=10,
        limit_train_batches=2,
        limit_val_batches=2,
        limit_test_batches=2,
        num_sanity_val_steps=0,
        devices=1,
        accelerator="auto",
        strategy="ddp",
    )
    internal_fit_loop = trainer.fit_loop
    trainer.fit_loop = StratifiedKFoldLoop(5, export_path="./")
    trainer.fit_loop.connect(internal_fit_loop)
    trainer.fit(model, datamodule)
