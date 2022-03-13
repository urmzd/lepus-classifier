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

if not os.path.exists('images'):
    os.mkdir('images')

urls = []
with open('data.txt') as f:
    line = f.readline()
    while line:
        line = f.readline()

        if line.strip():
            image_url = line.split(',')
            image_url = image_url[1].strip()
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