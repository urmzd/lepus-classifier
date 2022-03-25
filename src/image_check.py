#!/usr/bin/env python
'''
Program: Image Check
Description: Image Checking Script using difPy to identify duplicate images in the collected dataset

Script Results:
    Found 1 image with one or more duplicate/similar images in 3.1972 seconds.
    {'40.jpg': {'location': 'images/40.jpg', 'duplicates': ['images/8.jpg']}}

'''

from difPy import dif
from data_retrieval import get_data, download_image

DATA_PATH="resources/data.csv"
IMAGE_PATH="/tmp/images"

if __name__ == "__main__":
    df = get_data(DATA_PATH)

    for link in df.iloc[:, 1].tolist():
        download_image(link, IMAGE_PATH)

    search = dif("/tmp/images")

    print(search.result)
