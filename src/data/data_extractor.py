import re
from functools import lru_cache
from pathlib import Path

import cv2
import pandas as pd

from src.data.data_types import Image


def get_data(data_path: Path) -> pd.DataFrame:
    df = pd.read_csv(data_path, usecols=range(2))
    if isinstance(df, pd.DataFrame):
        return df

    raise Exception("This should never happen.")


@lru_cache()
def extract_path_from_link(link: str, image_folder_path: Path) -> Path:
    """Extract the image path from the specified link."""
    file_name_regex = re.compile(r".*/(.*\.(jpeg|jpg|png)?)\??", flags=re.IGNORECASE)
    regex_matches = file_name_regex.match(link)

    if not regex_matches:
        raise Exception(f"Failed to match file_name for link {link}")

    if len(regex_matches.groups()) < 3:
        file_name = regex_matches.group(1) + ".png"
    else:
        file_name = regex_matches.group(1)

    file_path = image_folder_path / file_name

    return file_path


def download_image_from_link(link: str, image_folder_path: Path) -> Path:
    """Return the local path for an image. Images are expected to already exist locally."""
    file_path = extract_path_from_link(link, image_folder_path)
    if not file_path.exists():
        raise FileNotFoundError(
            f"Image not found at {file_path}. Run 'python scripts/download_images.py' first."
        )
    return file_path


def get_image(file_path: Path) -> Image:
    image = cv2.imread(str(file_path))

    return Image(image)
