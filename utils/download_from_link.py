#this file is used to download images from a link and save them in a folder

import requests
import os
from PIL import Image
from io import BytesIO
import cv2
import numpy as np
import pandas as pd
import json

#read links from a json and download the images
def download_images_from_json(json_file_path, folder_path):
    with open(json_file_path, 'r', encoding='utf-8') as json_file:
        data =json.load(json_file)
    for d in data:
        response = requests.get(d['url'])
        img = Image.open(BytesIO(response.content))
        img.save(folder_path+'/'+d['filename'])

'''
json_file_path = 'YESBUT_dataset_info.json'
folder_path = 'original_image/'
download_images_from_json(json_file_path, folder_path)
'''

#download the images from a csv file
def download_images_from_csv(csv_file_path, folder_path):
    data = pd.read_csv(csv_file_path)
    for index, row in data.iterrows():
        response = requests.get(row['url'])
        img = Image.open(BytesIO(response.content))
        img.save(folder_path+'/'+row['filename'])

'''        
csv_path='all_dataset.csv'
dataset_path='dataset/'
download_images_from_csv(csv_path,dataset_path)
'''

#download the images through a link
def download_images_from_link(link,path):
    response = requests.get(link)
    img = Image.open(BytesIO(response.content))
    img.save(path)
        
'''
url='https://www.google.com/images/branding/googlelogo/1x/googlelogo_color_272x92dp.png'
path='dataset/google.png'
download_images_from_link(url,path)
'''

download_images_from_json('YESBUT_dataset_info.json', 'original_image')