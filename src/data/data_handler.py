from copy import deepcopy
from dataclasses import dataclass
from loguru import logger
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from sklearn.model_selection import (
    StratifiedShuffleSplit,
    train_test_split,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset
from torchmetrics import Accuracy, ConfusionMatrix
from torchmetrics.functional import accuracy

from typing import Any, Dict, List, Optional, Type
from abc import ABC, abstractmethod
from pytorch_lightning.loops.base import Loop
from pytorch_lightning.loops.fit_loop import FitLoop
from pytorch_lightning.trainer.states import TrainerFn
import wandb

from src.data.data_processing import get_target_encoder
from src.data.data_types import TargetEncoder, FeaturesEncoder
from src.data.data_extractor import download_image, get_data, get_image


class StratifiedKFoldDataModule(pl.LightningDataModule, ABC):
    @abstractmethod
    def setup_fold_using_index(self, fold_index: int) -> None:
        pass


class LepusDataset(Dataset):
    def __init__(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        image_folder_path: Path,
        transform: Optional[FeaturesEncoder] = None,
        target_transform: Optional[TargetEncoder] = None,
    ) -> None:
        self.features = features
        self.targets = targets
        self.image_folder_path = image_folder_path
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        image_link = self.features[idx]
        image_label = self.targets[idx]
        image_path = download_image(image_link, self.image_folder_path)
        image = get_image(image_path)

        if self.transform is not None:
            image = self.transform(image)

        if self.target_transform is not None:
            image_label = self.target_transform(image_label)

        return image, image_label


@dataclass
class LepusStratifiedKFoldDataModule(StratifiedKFoldDataModule):
    transform_features: FeaturesEncoder
    image_folder_path: Path
    data_manifest_path: Path
    batch_size: int = 1
    n_splits: int = 1
    transform_targets: bool = True
    train_size: float = 0.8

    def prepare_data(self) -> None:
        if (
            self.data_manifest_path is not None
            and self.image_folder_path is not None
            and self.transform_features is not None
        ):
            self.data = get_data(self.data_manifest_path).to_numpy()
        else:
            raise ValueError("data_manifest_path must be a Path object.")

    def setup(self, stage: Optional[str] = None):
        if self.transform_targets:
            self.target_encoder = get_target_encoder(self.data[:, 0])
        else:
            self.target_encoder = None

        # 80 / 20
        train_set, test_set = train_test_split(self.data, train_size=self.train_size)
        self.train_dataset, self.test_dataset = train_set, test_set

        features, targets = train_set[:, 1], train_set[:, 0]

        if self.n_splits >= 2:
            # 80 / 20 * 80 / 20
            splitter = StratifiedKFold(self.n_splits)
            self.splits = [split for split in splitter.split(features, targets)]
        else:
            splitter = StratifiedShuffleSplit(1, test_size=0.8)
            self.splits = [split for split in splitter.split(features, targets)]

    def setup_fold_using_index(self, fold_index: int) -> None:
        train_indices, val_indices = self.splits[fold_index]
        self.train_indices = train_indices
        self.val_indices = val_indices

    def train_dataloader(self) -> DataLoader:
        features = self.train_dataset[self.train_indices, 1]
        targets = self.train_dataset[self.train_indices, 0]
        return DataLoader(
            LepusDataset(
                features,
                targets,
                self.image_folder_path,
                self.transform_features,
                self.target_encoder,
            ),
            self.batch_size,
        )

    def val_dataloader(self) -> DataLoader:
        features = self.train_dataset[self.val_indices, 1]
        targets = self.train_dataset[self.val_indices, 0]
        return DataLoader(
            LepusDataset(
                features,
                targets,
                self.image_folder_path,
                self.transform_features,
                self.target_encoder,
            ),
            self.batch_size,
        )

    def test_dataloader(self) -> DataLoader:
        features = self.test_dataset[:, 1]
        targets = self.test_dataset[:, 0]
        return DataLoader(
            LepusDataset(
                features,
                targets,
                self.image_folder_path,
                self.transform_features,
                self.target_encoder,
            ),
            self.batch_size,
        )

    def __post_init__(cls):
        super().__init__()


class EnsembleVotingModel(pl.LightningModule):
    def __init__(
        self, model_cls: Type[pl.LightningModule], checkpoint_paths: List[Path]
    ) -> None:
        super().__init__()
        self.models = torch.nn.ModuleList(
            [model_cls.load_from_checkpoint(p) for p in checkpoint_paths]
        )
        self.test_acc = Accuracy()

    def test_step(self, batch: Any, dataloader_idx: int = 0) -> None:
        logits = torch.stack([m(batch[0]) for m in self.models]).mean(0)
        loss = F.cross_entropy(logits, batch[1])
        self.test_acc(logits, batch[1])
        self.log("test_acc", self.test_acc, on_step=False, on_epoch=True)
        self.log("test_loss", loss, on_step=False, on_epoch=True)


class StratifiedKFoldLoop(Loop):
    def __init__(self, num_folds: int, export_path: Path) -> None:
        super().__init__()
        self.num_folds = num_folds
        self.current_fold = 0
        self.export_path = export_path

    @property
    def done(self) -> bool:
        return self.current_fold >= self.num_folds

    def connect(self, fit_loop: FitLoop) -> None:
        self.fit_loop = fit_loop

    def on_run_start(self, *args: Any, **kwargs: Any) -> None:
        assert isinstance(self.trainer.datamodule, StratifiedKFoldDataModule)
        self.lighning_module_state_dict = deepcopy(
            self.trainer.lightning_module.state_dict()
        )

    def on_advance_start(self, *args: Any, **kwargs: Any) -> None:
        logger.info(f"STARTING FOLD {self.current_fold}")
        assert isinstance(self.trainer.datamodule, StratifiedKFoldDataModule)
        self.trainer.datamodule.setup_fold_using_index(self.current_fold)

    def advance(self, *args: Any, **kwargs: Any) -> None:
        self._reset_fitting()
        self.fit_loop.run()

        self._reset_testing()
        self.trainer.test_loop.run()
        self.current_fold += 1

    def on_advance_end(self) -> None:
        self.trainer.save_checkpoint(
            self.export_path / f"model.fold-{self.current_fold}.pt"
        )
        self.trainer.lightning_module.load_state_dict(self.lighning_module_state_dict)
        self.trainer.strategy.setup_optimizers(self.trainer)
        self.replace(fit_loop=FitLoop)

    def reset(self) -> None:
        return None

    def on_run_end(self) -> None:
        checkpoint_paths = [
            self.export_path / f"model.fold-{fold + 1}.pt"
            for fold in range(self.num_folds)
        ]
        voting_model = EnsembleVotingModel(
            type(self.trainer.lightning_module), checkpoint_paths
        )
        voting_model.trainer = self.trainer
        self.trainer.strategy.connect(voting_model)
        self.trainer.strategy.model_to_device()
        self.trainer.test_loop.run()

    def on_save_checkpoint(self) -> Dict[str, int]:
        return {"current_fold": self.current_fold}

    def on_load_checkpoint(self, state_dict: Dict[str, int]) -> None:
        self.current_fold = state_dict["current_fold"]

    def _reset_fitting(self) -> None:
        self.trainer.reset_train_dataloader()
        self.trainer.reset_val_dataloader()
        self.trainer.state.fn = TrainerFn.FITTING
        self.trainer.training = True

    def _reset_testing(self) -> None:
        self.trainer.reset_test_dataloader()
        self.trainer.state.fn = TrainerFn.TESTING
        self.trainer.testing = True

    def __getattr__(self, key) -> Any:
        if key not in self.__dict__:
            return getattr(self.fit_loop, key)

        return self.__dict__[key]


class BasicModel(pl.LightningModule):
    def __init__(self, n_targets=2) -> None:
        super().__init__()
        # O: H/2, W/2
        self.layer_1 = torch.nn.Conv2d(1, 15, 2, 2)
        # O: H/2, W/2
        self.layer_2 = torch.nn.MaxPool2d(2, 2)
        self.layer_3 = torch.nn.ReLU()
        self.layer_4 = torch.nn.Flatten(1, -1)
        self.layer_5 = torch.nn.Linear(15 * 50 * 50, n_targets)
        self.softmax_layer = torch.nn.LogSoftmax()

        # Logs
        self.train_acc = Accuracy()
        self.val_acc = Accuracy()
        self.test_acc = Accuracy()

        self.train_conf_mat = ConfusionMatrix(num_classes=n_targets)

        self.save_hyperparameters()

    def forward(self, x):
        x_1 = self.layer_1(x)
        x_2 = self.layer_2(x_1)
        x_3 = self.layer_3(x_2)
        x_4 = self.layer_4(x_3)
        x_5 = self.layer_5(x_4)
        result = self.softmax_layer(x_5)
        return result

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        x, y = batch
        preds = self.forward(x)
        loss = F.nll_loss(preds, y)

        self.train_acc(preds, y)
        self.train_conf_mat(preds, y)

        self.log("train_acc", self.train_acc, on_epoch=True, on_step=False)
        self.log("train_loss", loss, on_epoch=True, on_step=False)
        return loss

    def test_step(self, batch, batch_idx) -> None:
        x, y = batch
        preds = self.forward(x)
        loss = F.nll_loss(preds, y)
        self.test_acc(preds, y)

        self.log("test_acc", self.test_acc, on_epoch=True, on_step=False)
        self.log("test_loss", loss, on_epoch=True, on_step=False)

    def validation_step(self, batch, batch_idx) -> torch.Tensor:
        x, y = batch
        y_hat = self.forward(x)
        val_loss = F.nll_loss(y_hat, y)
        self.val_acc(y_hat, y)
        self.log("val_acc", self.val_acc, on_epoch=True, on_step=False)
        self.log("val_loss", val_loss, on_epoch=True, on_step=False)
        return val_loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.02)
