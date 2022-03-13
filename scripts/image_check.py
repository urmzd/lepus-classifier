'''
Program: Image Check
Description: Image Checking Script using difPy to identify duplicate images in the collected dataset

Script Results:
    Found 1 image with one or more duplicate/similar images in 3.1972 seconds.
    {'40.jpg': {'location': 'images/40.jpg', 'duplicates': ['images/8.jpg']}}

'''

from difPy import dif
import requests
import shutil
import os
import csv

if not os.path.exists('images'):
    os.mkdir('images')

urls = []
with open('../resources/data.csv','r') as f:
    csv_read = csv.reader(f)
    header = next(csv_read)
    if header is not None:
        for row in csv_read:

            line = row[1]

            if line.strip():
                image_url = line.strip()
                urls.append(image_url)

f.close()

for i, link in enumerate(urls):
    response = requests.get(link)

    image_name = 'images' + '/' + str(i + 1) + '.jpg'

    with open(image_name, 'wb') as fh:
        fh.write(response.content)

fh.close()

search = dif("images")
print(search.result)