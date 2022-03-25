import pandas as pd
import re
import requests
import matplotlib.pyplot as plt
import cv2
from pathlib import Path
from custom_types import ImageLabelPair, Image, PreprocessedImage, PreprocessedLabel


def get_data(data_path: str) -> pd.DataFrame:
    df = pd.read_csv(data_path, usecols=range(2))
    if isinstance(df, pd.DataFrame):
        return df

    raise Exception("This should never happen.")


def download_image(link: str, image_folder_path: str) -> Path:
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

    content_path = Path(image_folder_path)
    content_path.mkdir(parents=True, exist_ok=True)

    file_path = content_path / file_name

    if file_path.exists():
        return file_path

    image_request_headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.102 Safari/537.36"
    }

    image = requests.get(link, headers=image_request_headers)

    with open(file_path, "wb") as handle:
        handle.write(image.content)

    return file_path


def get_image(file_path: Path, show=False) -> Image:
    image = cv2.imread(str(file_path))

    if show:
        plt.imshow(image)

    return Image(image)


def get_x_y(raw_data: pd.DataFrame, image_folder_path: str) -> ImageLabelPair:
    y = raw_data.iloc[:, 0].to_numpy()
    y = y.reshape(-1, 1)

    x_links = raw_data.iloc[:, 1].tolist()
    x_paths = [download_image(link, image_folder_path) for link in x_links]
    x = [get_image(path) for path in x_paths]

    return ImageLabelPair(PreprocessedImage(x), PreprocessedLabel(y))
