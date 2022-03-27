"""_summary_

Returns:
    _type_: _description_
"""
from dataclasses import dataclass
from pandas import DataFrame
import pandas as pd
from pathlib import Path

from pytorch_lightning import LightningDataModule
from sklearn.model_selection import StratifiedKFold
from data.data_types import TargetEncoder, FeaturesEncoder

# from data.data_retrieval import download_image, get_data, get_image
from typing import Optional
from torch.utils.data import Dataset
from data.data_processing import (
    get_image_encoder,
    get_label_encoder_handler,
)
from abc import ABC, abstractmethod

from src.data.data_extractor import download_image, get_data, get_image


class StratifiedKFoldDataModule(LightningDataModule, ABC):
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
        self.features = (pd.Series,)
        self.targets = (pd.Series,)
        self.image_folder_path = image_folder_path
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.data.index)

    def __getitem__(self, idx):
        link, label = self.data.iloc[idx, :]
        image_path = download_image(link, self.image_folder_path)
        image = get_image(image_path)

        if self.transform is not None:
            image = self.transform(image)

        if self.target_transform is not None:
            label = self.target_transform(label)

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
            data = get_data(self.data_manfiest_path)
            self.features = data[:, 1]
            self.targets = data[:, 0]

        raise ValueError("data_manifest_path must be a Path object.")

    def setup(self, stage: Optional[str] = None):
        splitter = StratifiedKFold(self.n_splits)
        self.splits = [split for split in splitter.split(self.features, self.targets)]

    def setup_fold_using_index(self, fold_index: int) -> None:
        train_indices, val_indices = self.splits[fold_index]
        self.train_indices = train_indices
        self.val_indices = val_indices

    def train_dataloader(self) -> LepusDataset:
        features = self.features[self.train_indices]
        targets = self.targets[self.train_indices]
        return LepusDataset(
            features,
            targets,
            self.image_folder_path,
            self.transform_features,
            self.transform_targets,
        )

    def val_dataloader(self) -> LepusDataset:
        features = self.features[self.val_indices]
        targets = self.targets[self.val_indices]
        return LepusDataset(
            features,
            targets,
            self.image_folder_path,
            self.transform_features,
            self.transform_targets,
        )


def get_data_loader(
    data_path: Path,
    image_folder_path: Path,
    desired_width: int,
    desired_height: int,
    scale_height: bool = False,
):
    data = get_data(data_path)
    image_encoder = get_image_encoder(desired_height, desired_width, scale_height)
    label_encoder_handler = get_label_encoder_handler(data.loc["label"].reshape(-1, 1))
    label_encoder = lambda label: label_encoder_handler.transform(label)

    return LepusDataset(data, image_folder_path, image_encoder, label_encoder)
