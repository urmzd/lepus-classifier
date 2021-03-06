from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Type

import numpy as np
import plotly.express as px
import pytorch_lightning as pl
import torch
from loguru import logger
from pytorch_lightning.loops.base import Loop
from pytorch_lightning.loops.fit_loop import FitLoop
from pytorch_lightning.trainer.states import TrainerFn
from sklearn.model_selection import (StratifiedKFold, StratifiedShuffleSplit,
                                     train_test_split)
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torchmetrics import (Accuracy, ConfusionMatrix, F1Score, MetricCollection,
                          Precision, Recall)
from typing_extensions import TypedDict

import wandb
from src.data.data_extractor import (download_image_from_link,
                                     extract_path_from_link, get_data,
                                     get_image)
from src.data.data_processing import get_target_encoder
from src.data.data_types import FeaturesEncoder, TargetEncoder


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
        image_path = extract_path_from_link(image_link, self.image_folder_path)
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
        data = get_data(self.data_manifest_path).to_numpy()
        for image_link in data[:, 1]:
            download_image_from_link(image_link, self.image_folder_path)

    def setup(self, stage: Optional[str] = None):
        data = get_data(self.data_manifest_path).to_numpy()

        if self.transform_targets:
            self.target_encoder = get_target_encoder(data[:, 0])
        else:
            self.target_encoder = None

        train_set, test_set = train_test_split(data, train_size=self.train_size)
        self.train_dataset, self.test_dataset = train_set, test_set

        features, targets = train_set[:, 1], train_set[:, 0]

        if self.n_splits >= 2:
            splitter = StratifiedKFold(self.n_splits)
        else:
            splitter = StratifiedShuffleSplit(1, train_size=self.train_size)

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

    def __post_init__(self):
        super().__init__()


class StepOutputDict(TypedDict):
    loss: torch.Tensor
    logits: torch.Tensor
    y_true: torch.Tensor


class EnsembleVotingModel(pl.LightningModule):
    def __init__(
        self,
        model_cls: Type[pl.LightningModule],
        checkpoint_paths: List[Path],
    ) -> None:
        super().__init__()
        self.models = torch.nn.ModuleList(
            [model_cls.load_from_checkpoint(p) for p in checkpoint_paths]
        )

    def test_step(self, batch: Any, dataloader_idx: int = 0) -> StepOutputDict:
        logits = torch.stack([m(batch[0]) for m in self.models]).mean(0)
        loss = F.nll_loss(logits, batch[1])
        output: StepOutputDict = {"loss": loss, "logits": logits, "y_true": batch[1]}
        return output


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
        self.lightning_module_state_dict = deepcopy(
            self.trainer.lightning_module.state_dict()
        )

    def on_advance_start(self, *args: Any, **kwargs: Any) -> None:
        logger.info(f"STARTING FOLD # {self.current_fold}")
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
        self.trainer.lightning_module.load_state_dict(self.lightning_module_state_dict)
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


class BaseModel(pl.LightningModule, ABC):
    def __init__(self, n_targets=2, learning_rate=0.02) -> None:
        super().__init__()

        self.learning_rate = learning_rate
        self.n_targets = n_targets

    def __post_init__(self) -> None:
        self.save_hyperparameters()

    @abstractmethod
    def forward(self, x):
        return x

    def _compute_loss(self, batch) -> StepOutputDict:
        x, y = batch
        logits = self.forward(x)
        loss = F.nll_loss(logits, y)

        output: StepOutputDict = {"loss": loss, "logits": logits, "y_true": y}

        return output

    def training_step(self, batch, batch_idx) -> StepOutputDict:
        return self._compute_loss(batch)

    def validation_step(self, batch, batch_idx) -> StepOutputDict:
        return self._compute_loss(batch)

    def test_step(self, batch, batch_idx) -> StepOutputDict:
        return self._compute_loss(batch)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)


class MetricState(TypedDict):
    epochs: int
    fold: int


class MetricsCallback(pl.Callback):
    def __init__(self, n_targets=2, accuracy_average="micro", default_average="macro"):
        metrics = MetricCollection(
            Accuracy(num_classes=n_targets, average=accuracy_average),
            Precision(num_classes=n_targets, average=default_average),
            F1Score(num_classes=n_targets, average=default_average),
            Recall(num_classes=n_targets, average=default_average),
            ConfusionMatrix(num_classes=n_targets),
        )

        self.train_metrics = metrics.clone(prefix="train/")
        self.val_metrics = metrics.clone(prefix="val/")
        self.test_metrics = metrics.clone(prefix="test/")
        self.state: MetricState = {"fold": 0, "epoch": 0}

    def load_state_dict(self, state_dict: MetricState) -> None:
        self.state.update(state_dict)

    def state_dict(self) -> MetricState:
        return self.state.copy()

    def on_train_start(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
    ) -> None:
        self.train_metrics = self.train_metrics.to(pl_module.device)

    def on_validation_start(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        self.val_metrics = self.val_metrics.to(pl_module.device)

    def on_test_start(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        self.test_metrics = self.test_metrics.to(pl_module.device)

    def on_train_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: StepOutputDict,
        batch: Any,
        batch_idx: int,
        unused: int = 0,
    ) -> None:
        self._log_metric_on_batch(self.train_metrics, outputs, trainer, "train")

    def on_validation_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: StepOutputDict,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        self._log_metric_on_batch(self.val_metrics, outputs, trainer, "val")

    def on_test_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: StepOutputDict,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        self._log_metric_on_batch(self.test_metrics, outputs, trainer, "test")

    def on_train_epoch_start(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        self.state["epoch"] += 1

    def on_train_epoch_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        self._log_metric_on_epoch_end(self.train_metrics, trainer)

    def on_validation_epoch_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        self._log_metric_on_epoch_end(self.val_metrics, trainer)

    def on_test_epoch_start(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        self.state["epoch"] += 1

    def on_test_epoch_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        self._log_metric_on_epoch_end(self.test_metrics, trainer)
        self.state["fold"] += 1

    def _log_metric_on_epoch_end(
        self,
        metrics: MetricCollection,
        trainer: pl.Trainer,
    ):
        metrics_dict = {}
        confusion_matrix_key = None
        confusion_matrix = None

        computed_metrics = metrics.compute()

        for key in computed_metrics:
            if "ConfusionMatrix" in key:
                confusion_matrix_key = key
                confusion_matrix = computed_metrics[key]
            else:
                metrics_dict[key] = computed_metrics[key]

        plot = px.imshow(confusion_matrix.cpu().detach().numpy(), text_auto=True)
        wandb.log({confusion_matrix_key: plot})
        wandb.log(
            {
                "global_step": trainer.global_step,
                "epoch": self.state["epoch"],
                "fold": self.state["fold"],
                **metrics_dict,
            }
        )

        metrics.reset()

    def _log_metric_on_batch(
        self,
        metrics: MetricCollection,
        step_output_dict: StepOutputDict,
        trainer: pl.Trainer,
        stage: str,
    ):
        assert (
            "loss" in step_output_dict
            and "y_true" in step_output_dict
            and "logits" in step_output_dict
        )

        loss = step_output_dict["loss"]
        y_true = step_output_dict["y_true"]
        logits = step_output_dict["logits"]

        metrics.update(logits, y_true)

        metric_dict = {
            key: metric.compute()
            for key, metric in metrics.items()
            if "ConfusionMatrix" not in key
        }

        wandb.log(
            {
                "global_step": trainer.global_step,
                "epoch": self.state["epoch"],
                "fold": self.state["fold"],
                f"{stage}/loss": loss,
                **metric_dict,
            }
        )
