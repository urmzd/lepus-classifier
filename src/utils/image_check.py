#!/usr/bin/env python

from data_retrieval import download_image, get_data
from difPy import dif

DATA_PATH = "../resources/data.csv"
IMAGE_PATH = "/tmp/images"

if __name__ == "__main__":
    df = get_data(DATA_PATH)

    for link in df.iloc[:, 1].tolist():
        download_image(link, IMAGE_PATH)

    search = dif("/tmp/images")

    print(search.result)
