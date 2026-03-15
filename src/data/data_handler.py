from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import lightning as L
import numpy as np
import plotly.express as px
import torch
from lightning_kfold import KFoldDataModule
from sklearn.model_selection import train_test_split
from torch.nn import functional as F
from torch.utils.data import Dataset
from torchmetrics import (Accuracy, ConfusionMatrix, F1Score, MetricCollection,
                          Precision, Recall)
from typing_extensions import TypedDict

import wandb
from src.data.data_extractor import (download_image_from_link,
                                     extract_path_from_link, get_data,
                                     get_image)
from src.data.data_processing import get_target_encoder
from src.data.data_types import FeaturesEncoder, TargetEncoder


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
class LepusStratifiedKFoldDataModule(KFoldDataModule):
    transform_features: FeaturesEncoder = None
    image_folder_path: Path = None
    data_manifest_path: Path = None
    num_folds: int = 1
    batch_size: int = 1
    train_size: float = 0.8
    transform_targets: bool = True

    def __post_init__(self):
        super().__init__(
            num_folds=self.num_folds,
            batch_size=self.batch_size,
            train_size=self.train_size,
        )

    def prepare_data(self) -> None:
        data = get_data(self.data_manifest_path).to_numpy()
        for image_link in data[:, 1]:
            download_image_from_link(image_link, self.image_folder_path)

    def setup_datasets(self) -> tuple:
        data = get_data(self.data_manifest_path).to_numpy()

        if self.transform_targets:
            self.target_encoder = get_target_encoder(data[:, 0])
        else:
            self.target_encoder = None

        train_set, test_set = train_test_split(data, train_size=self.train_size)

        train_dataset = LepusDataset(
            train_set[:, 1],
            train_set[:, 0],
            self.image_folder_path,
            self.transform_features,
            self.target_encoder,
        )
        test_dataset = LepusDataset(
            test_set[:, 1],
            test_set[:, 0],
            self.image_folder_path,
            self.transform_features,
            self.target_encoder,
        )

        # Labels for stratified splitting (use raw string labels).
        train_labels = train_set[:, 0]

        return train_dataset, test_dataset, train_labels


class StepOutputDict(TypedDict):
    loss: torch.Tensor
    logits: torch.Tensor
    y_true: torch.Tensor


class BaseModel(L.LightningModule, ABC):
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


class MetricsCallback(L.Callback):
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
        trainer: "L.Trainer",
        pl_module: "L.LightningModule",
    ) -> None:
        self.train_metrics = self.train_metrics.to(pl_module.device)

    def on_validation_start(
        self, trainer: "L.Trainer", pl_module: "L.LightningModule"
    ) -> None:
        self.val_metrics = self.val_metrics.to(pl_module.device)

    def on_test_start(
        self, trainer: "L.Trainer", pl_module: "L.LightningModule"
    ) -> None:
        self.test_metrics = self.test_metrics.to(pl_module.device)

    def on_train_batch_end(
        self,
        trainer: "L.Trainer",
        pl_module: "L.LightningModule",
        outputs: StepOutputDict,
        batch: Any,
        batch_idx: int,
        unused: int = 0,
    ) -> None:
        self._log_metric_on_batch(self.train_metrics, outputs, trainer, "train")

    def on_validation_batch_end(
        self,
        trainer: "L.Trainer",
        pl_module: "L.LightningModule",
        outputs: StepOutputDict,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        self._log_metric_on_batch(self.val_metrics, outputs, trainer, "val")

    def on_test_batch_end(
        self,
        trainer: "L.Trainer",
        pl_module: "L.LightningModule",
        outputs: StepOutputDict,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        self._log_metric_on_batch(self.test_metrics, outputs, trainer, "test")

    def on_train_epoch_start(
        self, trainer: "L.Trainer", pl_module: "L.LightningModule"
    ) -> None:
        self.state["epoch"] += 1

    def on_train_epoch_end(
        self, trainer: "L.Trainer", pl_module: "L.LightningModule"
    ) -> None:
        self._log_metric_on_epoch_end(self.train_metrics, trainer)

    def on_validation_epoch_end(
        self, trainer: "L.Trainer", pl_module: "L.LightningModule"
    ) -> None:
        self._log_metric_on_epoch_end(self.val_metrics, trainer)

    def on_test_epoch_start(
        self, trainer: "L.Trainer", pl_module: "L.LightningModule"
    ) -> None:
        self.state["epoch"] += 1

    def on_test_epoch_end(
        self, trainer: "L.Trainer", pl_module: "L.LightningModule"
    ) -> None:
        self._log_metric_on_epoch_end(self.test_metrics, trainer)
        self.state["fold"] += 1

    def _log_metric_on_epoch_end(
        self,
        metrics: MetricCollection,
        trainer: L.Trainer,
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
        trainer: L.Trainer,
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
