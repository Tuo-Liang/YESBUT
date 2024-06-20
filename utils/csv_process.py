#Description: This file is used to deal with the csv file of the dataset
import pandas as pd
import numpy as np
import os
import json
import requests
from PIL import Image
from io import BytesIO
import cv2 


#delte the row with the filename
def delete_row_with_filename(csv_file_path, filename):
    data = pd.read_csv(csv_file_path)
    data = data[data['filename'] != filename]
    data.to_csv(csv_file_path, index=False)

#add a row to the csv file and this row follow the order of the filename such as '00001.jpg', '00002.jpg'
def add_row_to_csv(csv_file_path, row):
    data = pd.read_csv(csv_file_path)
    data = data.append(row, ignore_index=True)
    data.to_csv(csv_file_path, index=False)

#merge two csv files and save to a new csv file
def merge_csv(csv_file_path1, csv_file_path2,new_csv_file_path):
    data1 = pd.read_csv(csv_file_path1)
    data2 = pd.read_csv(csv_file_path2)
    data = data1.append(data2, ignore_index=True)
    data.to_csv(new_csv_file_path, index=False)

#function transfer csv file to json file
#the same function as the one in json_process.py
def csv_to_json(csv_path, json_file_path):
    # Use a list to store data
    data = []

    # Read data from CSV file
    df = pd.read_csv(csv_path)

    for i in range(len(df)):
        if pd.isna(df.iloc[i]['filename']):
            continue

        # Create a dictionary for each row
        row_data = {
            'filename': df.iloc[i]['filename'],
            'url': df.iloc[i]['url'],
            'num_of_pic': df.iloc[i]['num_of_pic'],
            'bounding_box': [],
            'labels': []
        }

        # Append the dictionary to the list
        data.append(row_data)

    # Write data to JSON file
    with open(json_file_path, 'w', encoding='utf-8') as json_file:
        json.dump(data, json_file, indent=4)
'''
csv_path = 'YESBUT_dataset_info.csv'
json_file_path = 'YESBUT_dataset_info.json'

csv_to_json(csv_path, json_file_path)
'''

#json file to csv file
def json_to_csv(json_file_path, csv_file_path):
    with open(json_file_path, 'r', encoding='utf-8') as json_file:
        data = json.load(json_file)
    df = pd.DataFrame(data)
    df.to_csv(csv_file_path, index=False)

'''
json_file_path = 'YESBUT_dataset_info.json'
csv_file_path = 'test.csv'
json_to_csv(json_file_path, csv_file_path)
'''

#delete the image record in csv for those who is not in the dataset.
def delete_image_record(image_folder,csv_path):
    #read the images in the folder
    images = os.listdir(image_folder)

    #read csv from the dataset
    df = pd.read_csv(csv_path)
    #iterate the df and delete the record if there is no the image in the folder
    i=0
    while True:
        if df.iloc[i,0] not in images:
            #0 is the column of the filename
            print(df.iloc[i,0],i)
            df.drop(i, axis=0, inplace=True)
        i=i+1
        if i==len(df):
            break
    #save the new dataset       
    df.to_csv(csv_path, index=False)