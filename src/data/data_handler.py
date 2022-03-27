from copy import deepcopy
from dataclasses import dataclass
import pandas as pd
from pathlib import Path

import pytorch_lightning as pl
from sklearn.model_selection import StratifiedKFold, train_test_split
import torch
from torch.nn import functional as F
from torchmetrics import Accuracy
from data.data_types import TargetEncoder, FeaturesEncoder

from typing import Any, Dict, List, Optional, Type
from torch.utils.data import Dataset
from abc import ABC, abstractmethod
from pytorch_lightning.loops.base import Loop, FitLoop
from pytorch_lightning.trainer.states import TrainerFn

from src.data.data_extractor import download_image, get_data, get_image


class StratifiedKFoldDataModule(pl.LightningDataModule, ABC):
    @abstractmethod
    def setup_fold_using_index(self, fold_index: int) -> None:
        pass


class LepusDataset(Dataset):
    def __init__(
        self,
        features: pd.Series,
        targets: pd.Series,
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
        return len(self.targets.index)

    def __getitem__(self, idx):
        image_link = self.features[idx]
        image_label = self.targets[idx]
        image_path = download_image(image_link, self.image_folder_path)
        image = get_image(image_path)

        if self.transform is not None:
            image = self.transform(image)

        if self.target_transform is not None:
            label = self.target_transform(image_label)

        return image, label


@dataclass
class LepusStratifiedKFoldDataModule(StratifiedKFoldDataModule):
    train_dataset: Optional[LepusDataset] = None
    test_dataset: Optional[LepusDataset] = None
    batch_size: int = 1
    data_manfiest_path: Optional[str] = None
    n_splits: int = 1
    image_folder_path: Optional[Path] = None
    transform_features: Optional[FeaturesEncoder] = None
    transform_targets: Optional[TargetEncoder] = None

    def prepare_data(self) -> None:
        if (
            self.data_manfiest_path is not None
            and self.image_folder_path is not None
            and self.transform_features is not None
            and self.transform_targets is not None
        ):
            self.data = get_data(self.data_manfiest_path)

        raise ValueError("data_manifest_path must be a Path object.")

    def setup(self, stage: Optional[str] = None):
        train_set, test_set = train_test_split(self.data)
        self.train_dataset, self.test_dataset = train_set, test_set

        features, targets = train_set[:, 1], train_set[:, 0]

        splitter = StratifiedKFold(self.n_splits)
        self.splits = [split for split in splitter.split(features, targets)]

    def setup_fold_using_index(self, fold_index: int) -> None:
        train_indices, val_indices = self.splits[fold_index]
        self.train_indices = train_indices
        self.val_indices = val_indices

    def train_dataloader(self) -> LepusDataset:
        features = self.train_dataset[self.train_indices, 1]
        targets = self.train_dataset[self.train_indices, 0]
        return LepusDataset(
            features,
            targets,
            self.image_folder_path,
            self.transform_features,
            self.transform_targets,
        )

    def val_dataloader(self) -> LepusDataset:
        features = self.train_dataset[self.val_indices, 1]
        targets = self.train_dataset[self.val_indices, 0]
        return LepusDataset(
            features,
            targets,
            self.image_folder_path,
            self.transform_features,
            self.transform_targets,
        )

    def test_dataloader(self) -> LepusDataset:
        features = self.test_dataset[:, 1]
        targets = self.test_dataset[:, 0]
        return LepusDataset(
            features,
            targets,
            self.image_folder_path,
            self.transform_features,
            self.transform_targets,
        )


class StratifiedKFoldLoop(Loop):
    def __init__(self, num_folds: int, export_path: str) -> None:
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
        assert isinstance(self.trainer.datamodule, StratifiedKFoldDataModule)
        self.trainer.datamodule.setup_fold_using_index(self.current_fold)

    def advance(self, *args: Any, **kwargs: Any) -> None:
        self._reset_fitting()
        self.fit_loop.run()

        self._reset_testing()
        self.trainer.test_loop.run()
        self.current_fold += 1

    def on_advance_end(self) -> None:
        self.trainer.save_checkpoint()
        return super().on_advance_end()

    def on_advance_end(self) -> None:
        self.trainer.save_checkpoint(
            Path(self.export_path) / f"model.fold-{self.current_fold}".pt
        )
        self.trainer.lightning_module.load_state_dict(self.lighning_module_state_dict)
        self.trainer.strategy.setup_optimizers(self.trainer)
        self.replace(fitloop=FitLoop)

    def on_run_end(self) -> None:
        checkpoint_paths = [
            Path(self.export_path) / f"model.fold-{fold}"
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


class EnsembleVotingModel(pl.LightningModule):
    def __init__(
        self, model_cls: Type[pl.LightningModule], checkpoint_paths: List[Path]
    ) -> None:
        super().__init__()
        self.models = torch.nn.ModuleList(
            [model_cls.load_from_checkpoint(p) for p in checkpoint_paths]
        )
        self.test_acc = Accuracy()

    def test_step(self, batch: Any) -> None:
        logits = torch.stack([m(batch[0]) for m in self.models]).mean(0)
        loss = F.nll_loss()
        self.test_acc(logits, batch[1])
        self.log("test_acc", self.test_acc)
        self.log("test_loss", loss)