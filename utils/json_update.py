import numpy as np
import pandas as pd
import os
import json
import cv2

'''def csv_to_json(csv_path, json_file_path):
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
            'bounding_box': [[[270, 12], [919, 529]], [[270, 551], [919, 1068]]],
            'labels': [0, 1]
        }

        # Append the dictionary to the list
        data.append(row_data)

    # Write data to JSON file
    with open(json_file_path, 'w', encoding='utf-8') as json_file:
        json.dump(data, json_file, indent=4)

csv_path = 'YESBUT_dataset_info.csv'
json_file_path = 'YESBUT_dataset_info.json'

csv_to_json(csv_path, json_file_path)
'''

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


#a function to read a json file and return the ordinate of the bounding box of a specific image
def get_box_from_name(json_file_path,image_name):
    with open(json_file_path, 'r', encoding='utf-8') as json_file:
        data =json.load(json_file)
    for d in data:
        if d['filename']==image_name:
            return d['bounding_box']
    return None
        
#given new oridinate of the bounding box and imagesize, return the new ordinate of the bounding box
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


get_new_box('YESBUT_dataset_info.json','images/yesbut_dataset_fromX',(1080,1080))