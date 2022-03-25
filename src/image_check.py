#!/usr/bin/env python
'''
Program: Image Check
Description: Image Checking Script using difPy to identify duplicate images in the collected dataset

Script Results:
    Found 1 image with one or more duplicate/similar images in 3.1972 seconds.
    {'40.jpg': {'location': 'images/40.jpg', 'duplicates': ['images/8.jpg']}}

'''

from os import name
from typing import Optional
from difPy import dif
import requests
import pathlib


import pandas as pd
import requests
import re
import pathlib

DATA_PATH="resources/data.csv"
IMAGE_PATH="/tmp/images"

def get_data() -> pd.DataFrame:
  df = pd.read_csv(DATA_PATH, usecols=range(2))
  if isinstance(df, pd.DataFrame):
      return df

  raise Exception("THIS SHOULD NEVER HAPPEN.")

def download_image(link: str) -> pathlib.Path:
  # view  : https://regex101.com/r/3bhDMM/1
  # delete: https://regex101.com/delete/N5sItwbrPF73ZllTnRDltxZ1
  file_name_regex = re.compile(r".*\/(.*(\.(jpeg|jpg|png))?)\??.*", flags=re.IGNORECASE)
  regex_matches = file_name_regex.match(link)

  if not regex_matches:
    raise Exception(f"Failed to match file_name for link {link}")

  if len(regex_matches.groups()) < 3:
    file_name = regex_matches.group(1) + ".png"
  else:
    file_name = regex_matches.group(1)

  content_path = pathlib.Path(IMAGE_PATH)
  content_path.mkdir(parents=True, exist_ok=True)

  file_path = content_path / file_name

  if file_path.exists():
    return file_path

  image_request_headers={
      'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.102 Safari/537.36'
  }

  image = requests.get(link, headers=image_request_headers)

  with open(file_path, "wb") as handle:
    handle.write(image.content)

  return file_path

if name == "__main__":
    df = get_data()

    for link in df.iloc[:, 1].tolist():
        download_image(link)

    search = dif("/tmp/images")

    print(search.result)
