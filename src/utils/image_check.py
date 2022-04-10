#!/usr/bin/env python

from pathlib import Path

from difPy import dif

from src.data.data_extractor import download_image_from_link, get_data

DATA_PATH = Path("../resources/data.csv")
IMAGE_PATH = Path("/tmp/images")

if __name__ == "__main__":
    df = get_data(DATA_PATH)

    for link in df.iloc[:, 1].tolist():
        download_image_from_link(link, IMAGE_PATH)

    search = dif("/tmp/images")

    print(search.result)
