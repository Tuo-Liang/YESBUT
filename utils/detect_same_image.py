#Description: This file contains the functions to detect the same image in two folders and merge two different datasets into one
import pandas as pd
import numpy as np
import os
import cv2
from skimage.metrics import structural_similarity as SSIM
import requests
from PIL import Image
from io import BytesIO
import json


#detect the same image in two folders
def detect_similar_images(image_folder,csv_path):
    #read the images in the folder
    images = [os.path.join(image_folder, f) for f in os.listdir(image_folder)]

    #read csv from the dataset
    df = pd.read_csv(csv_path)
    #iterate through the images and compare them
    for i in images:
        for j in images:
            if i==j:
                continue
            #read the images
            try:
                img1 = cv2.imread(i)
                img2 = cv2.imread(j)
            
                #resize the images to a fixed size
                img1 = cv2.resize(img1, (200, 200))
                img2 = cv2.resize(img2, (200, 200))

                #calculate the structural similarity index
                ssim = SSIM(cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY), cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY))
                         
                if ssim>0.95:
                    print(ssim)
                    #save the similar images
                    cv2.imwrite('test1.jpg', img1)
                    cv2.imwrite('test2.jpg', img2)   
                    print(j)
                    os.remove(j)
                    #index of j
                    filename=j.split('/')[-1]
                    
                    idx_j=df[df['filename']==filename].index[0]
                    print(idx_j,df.iloc[idx_j,0],j)
                    
                    df.iloc[idx_j,0]=np.nan
                    df.iloc[idx_j,1]=np.nan
                    df.iloc[idx_j,2]=np.nan

            except:
                continue
    #save the new dataset       
    df.to_csv(csv_path, index=False)

#a function to merge two different dataset into one and rename this files with its label
def merge_dataset(imagepath1,imagepath2,csvpath1, csvpath2,dataset_path,final_csv_path):
    #read the dataset
    df1 = pd.read_csv(csvpath1)
    df1 = df1.iloc[1:]
    df2 = pd.read_csv(csvpath2)
    df2 = df2.iloc[1:]

    df=pd.DataFrame(columns=['filename', 'url', 'grid'])

    #rename the columns
    index=0
    for img_name in os.listdir(imagepath1):
        index=index+1
        img = cv2.imread(os.path.join(imagepath1, img_name))
        #get the index of the image in csv
        i= int(img_name.split('.')[0])
        #rename the image
        print(i,img_name,df1.iloc[i-2,0])
        img_name= f"{index:05}.jpg"


        temp=pd.DataFrame({'filename':[img_name], 'url':[df1.iloc[i-2, 1]], 'grid':[df1.iloc[i-2, 2]]})
        df=pd.concat([df, temp])
        cv2.imwrite(dataset_path+'/'+img_name, img)



    for img_name in os.listdir(imagepath2):
        
        index=index+1
        img = cv2.imread(os.path.join(imagepath2, img_name))
        #get the index of the image in csv


        i=int(img_name.split('.')[0])
        print(i,img_name,df2.iloc[i-2,0])

        #rename the image
        img_name= f"{index:05}.jpg"

        #get the index of the image match to the name i in csv

        #rename the image
        if df2.iloc[i-2, 1] is np.nan:
            continue
        if df2.iloc[i-2, 2] is np.nan:
            continue
        temp2=pd.DataFrame({'filename':[img_name], 'url':[df2.iloc[i-2, 1]], 'grid':[df2.iloc[i-2, 2]]})
        df=pd.concat([df, temp2])
    
        cv2.imwrite(dataset_path+'/'+img_name, img)
            
    #save the new dataset
    df.to_csv(final_csv_path, index=False)



