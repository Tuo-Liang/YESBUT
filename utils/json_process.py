#Codes in this file is to process the json file
#The json file contains the bounding box,datasetname, filename and the label of the image


import pandas as pd
import numpy as np
import os
import cv2
from skimage.metrics import structural_similarity as SSIM
import requests
from PIL import Image
from io import BytesIO
import json

#function to get the bounding box
def get_bounding_box(json_path):
    with open(json_path) as f:
        data = json.load(f)
    return data

#function transfer csv file to json file
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
            'num_of_pic': df.iloc[i]['grid'],
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

#a function to update json file through csv file
def update_json_with_csv(csv_path,json_path,action):
    df=pd.read_csv(csv_path)
    #load the json file
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    if action=='add':
        for i in range(len(df)):
            print(df.iloc[i])
            #check if the the file has already been added
            for d in data:
                if d['filename']==df.iloc[i,0]:
                    p=1
                    break
                else:
                    p=0

            if p==1:
                continue
            else:
                d={}
                d['filename']=df.iloc[i,0]
                d['url']=df.iloc[i,1]
                d['num_of_pic']=int(df.iloc[i,2])

                d['bounding_box']=[]
                d['label']=[]
                #dataset name
                d['dataset']='YESBUT'
                data.append(d)

        with open(json_path, 'w') as f:
            json.dump(data, f, indent=4)

    elif action=='delete':
        for index,row in df.iterrows():
            for d in data:
                if d['filename']==row['filename']:
                    data.remove(d)
                    break

'''#update the json file
csv_path='YESBUT_dataset_info.csv'
json_path='YESBUT_dataset_info.json'
update_json_with_csv(csv_path,json_path)'''

#a function to add an item to a json file
def add_item(json_file_path,itemname,itemvalue):
    with open(json_file_path, 'r', encoding='utf-8') as json_file:
        data =json.load(json_file)
    new_data = []
    for d in data:
        d[itemname]=itemvalue
        new_data.append(d)
    
    with open(json_file_path, 'w', encoding='utf-8') as json_file:
        json.dump(new_data, json_file, indent=4)
'''
json_file_path = 'YESBUT_dataset_info.json'
add_item(json_file_path, 'bounding_box', [[21,12],[1233,213]])
'''

#a function to read a json file and return the ordinate of the bounding box of a specific image
def get_bounding_box_by_name(json_file_path,image_name):
    with open(json_file_path, 'r', encoding='utf-8') as json_file:
        data =json.load(json_file)
    for d in data:
        if d['filename']==image_name:
            return d['bounding_box']
    return None
        
#given new oridinate of the bounding box and imagesize, return the new ordinate of the bounding box
#we may scale the images at first, and then we need to transfer the bounding box to the original size
def get_new_box(json_file_path,img_folder,original_image_size):
    with open(json_file_path, 'r', encoding='utf-8') as json_file:
        data =json.load(json_file)

    for d in data:
        original_coordinates = d['bounding_box']
        img_path =img_folder+'/'+ d['filename']
        new_image_size = cv2.imread(img_path).shape[:2]
        
        x_ratio = new_image_size[0]/original_image_size[0]
        y_ratio = new_image_size[1]/original_image_size[1]
        new_coordinates = []
        for box in original_coordinates:
            new_box = []
            for point in box:
                new_box.append([int(point[0]*x_ratio),int(point[1]*y_ratio)])
            new_coordinates.append(new_box)
        d['bounding_box'] = new_coordinates

    with open(json_file_path, 'w', encoding='utf-8') as json_file:
        json.dump(data, json_file, indent=4)
#get_new_box('YESBUT_dataset_info.json','images/yesbut_dataset_fromX',(1080,1080))
