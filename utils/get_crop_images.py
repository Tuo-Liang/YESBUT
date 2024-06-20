# Description: This script is used to crop the images based on the bounding box information in the json file.
#
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import json

#resize to 1080x1080
def resize(img_path):
    #read the images from the folder
    img_list=os.listdir(img_path)
    
    for i in img_list:
        if i =='.DS_Store':
            continue
        img=cv2.imread(img_path+'/'+i)
        img=cv2.resize(img,(1080,1080))
        cv2.imwrite(img_path+'/'+i,img)

#function to get the oridinate of the bounding box
def get_bounding_box(name,json_path):
    with open(json_path, 'r', encoding='utf-8') as json_file:
        data =json.load(json_file)
    for d in data:
        if d['filename']==name:
            return d['bounding_box']
    return None

#crop the images and save to another folder
def crop(img_path,img_path2,json_path):
    img_list=os.listdir(img_path)
    for i in img_list:
        if i =='.DS_Store':
            continue
        img=cv2.imread(img_path+'/'+i)
        box=get_bounding_box(i,json_path)
        if box is None:
            continue
        j=0
        for b in box:

            x1=b[0][0]
            x2=b[1][0]
            y1=b[0][1]
            y2=b[1][1]

            img_temp=img[x1:x2,y1:y2,:]
            name=i.split('.')[0]

            cv2.imwrite(img_path2+'/'+name+'_'+str(j)+'.jpg',img_temp)
            j=j+1

#path to the original images
img_path='original_image'
#resize(img_path)

#path to the folder to save the cropped images

img_path2='split_data'
if not os.path.exists(img_path2):
    os.makedirs(img_path2)

json_path='YESBUT_dataset_info.json'
crop(img_path,img_path2,json_path)