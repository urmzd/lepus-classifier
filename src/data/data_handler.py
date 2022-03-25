"""_summary_

Returns:
    _type_: _description_
"""
from pandas import DataFrame
from pathlib import Path
from custom_types import LabelEncoder, XEncoder, LabelEncoderHandler
from data_retrieval import download_image, get_data, get_image
from typing import Optional
from torch.utils.data import Dataset
from processing import get_image_encoder, get_label_encoder, get_label_encoder_handler


class LepusDataset(Dataset):
    """_summary_

    Args:
        Dataset (_type_): _description_
    """
    def __init__(
        self,
        data: DataFrame,
        image_folder_path: Path,
        transform: Optional[XEncoder] = None,
        target_transform: Optional[LabelEncoder] = None,
    ) -> None:
        self.data = data
        self.image_folder_path = image_folder_path
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        """_summary_

        Returns:
            _type_: _description_
        """
        return len(self.data.index)

    def __getitem__(self, idx):
        """_summary_

        Args:
            idx (_type_): _description_

        Returns:
            _type_: _description_
        """
        link, label = self.data.iloc[idx, :]
        image_path = download_image(link, self.image_folder_path)
        image = get_image(image_path)

        if self.transform is not None:
            image = self.transform(image)

        if self.target_transform is not None:
            label = self.target_transform(label)

        return image, label


def get_data_loader(
    data_path: Path,
    image_folder_path: Path,
    desired_width: int,
    desired_height: int,
    scale_height: bool = False,
):
    """_summary_

    Args:
        data_path (Path): _description_
        image_folder_path (Path): _description_
        desired_width (int): _description_
        desired_height (int): _description_
        scale_height (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
    """
    data = get_data(data_path)
    image_encoder = get_image_encoder(desired_height, desired_width, scale_height)
    label_encoder_handler = get_label_encoder_handler(data.loc["label"].reshape(-1, 1))
    label_encoder = lambda label: label_encoder.transform(label)


    return LepusDataset(data, image_folder_path, image_encoder, label_encoder)
