import re
from pathlib import Path

import cv2
import pandas as pd
import requests

from src.data.data_types import Image


def get_data(data_path: Path) -> pd.DataFrame:
    df = pd.read_csv(data_path, usecols=range(2))
    if isinstance(df, pd.DataFrame):
        return df

    raise Exception("This should never happen.")


def download_image(link: str, image_folder_path: Path) -> Path:
    """Downloads the image from the specified link to the given folder path.

    View  : https://regex101.com/r/3bhDMM/1
    Delete: https://regex101.com/delete/N5sItwbrPF73ZllTnRDltxZ1
    """
    file_name_regex = re.compile(
        r".*\/(.*(\.(jpeg|jpg|png))?)\??.*", flags=re.IGNORECASE
    )
    regex_matches = file_name_regex.match(link)

    if not regex_matches:
        raise Exception(f"Failed to match file_name for link {link}")

    if len(regex_matches.groups()) < 3:
        file_name = regex_matches.group(1) + ".png"
    else:
        file_name = regex_matches.group(1)

    image_folder_path.mkdir(parents=True, exist_ok=True)

    file_path = image_folder_path / file_name

    if file_path.exists():
        return file_path

    image_request_headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.102 Safari/537.36"
    }

    image = requests.get(link, headers=image_request_headers)

    with open(file_path, "wb") as handle:
        handle.write(image.content)

    return file_path


def get_image(file_path: Path) -> Image:
    image = cv2.imread(str(file_path))

    return Image(image)
