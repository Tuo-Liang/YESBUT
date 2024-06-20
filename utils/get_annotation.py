# Description: This script is used to annotate images by clicking on the image and saving the annotated points to a text file.
# The script loads images from a folder, displays them one by one, and allows the user to click on the image to annotate points.
#The point we click on the image should be top-left and bottom-right of the object we want to detect.
import cv2
import os
import json
import numpy as np


#globals
annotations = []

def on_EVENT_LBUTTONDOWN(event, x, y, flags, param, img):
    if event == cv2.EVENT_LBUTTONDOWN:
        xy = "%d , %d" % (y,x) #cv2 uses y,x

        cv2.circle(img, (x, y), 3, (255, 0, 0), thickness=-1)
        cv2.putText(img, xy, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 0), thickness=1)
        cv2.namedWindow("image", cv2.WINDOW_NORMAL)        # Create window with freedom of dimensions
        cv2.resizeWindow("image", 1000, 600)
        cv2.imshow("image", img)
        # write the annotated points to a list
        annotations.append(xy)

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            img = cv2.imread(os.path.join(folder, filename))
            if img is not None:
                images.append(img)
    return images, os.listdir(folder)




# load images from a folder,
folder_path = 'yesbut_dataset_fromX'
label_path='testlabels'
json_path='YESBUT_dataset_info.json'

#write the annotations to the json file
with open(json_path, 'r') as file:
    data = json.load(file)

# load images from a folder
try:
    images, filenames = load_images_from_folder(folder_path)
except FileNotFoundError:
    print(f"Folder {folder_path} does not exist.")
    exit(1)


for img,filename in zip(images,filenames):
    # check if the image has been annotated
    # if the image has been annotated, skip the image
    p=0
    for d in data:
        if d['filename']==filename and d['bounding_box']!=[]:
            p=1
            break
    if p==1:
        continue
    # annotate the image
    cv2.namedWindow("image", cv2.WINDOW_NORMAL)        # Create window with freedom of dimensions
    cv2.setMouseCallback("image", on_EVENT_LBUTTONDOWN, img)
    cv2.resizeWindow("image", 1000, 600)
    cv2.imshow("image", img)
    cv2.waitKey(0)

    #transfer the annotations into boudingbox
    temp=[]
    box=[]
    bounding_box=[]
    j=0
    for point in annotations:
        point=point.split(',')
        temp.append(int(point[0]))
        temp.append(int(point[1]))
        box.append(temp)
        if j%2==1:
            bounding_box.append(box)
            box=[]
        temp=[]
        j+=1

    #update the json file
    print(filename)
    print(bounding_box)
    print(annotations)
    for d  in data:
        if d['filename']==filename:
            d['bounding_box']=bounding_box

    # clear the annotations list
    annotations = []
    cv2.destroyAllWindows()


# write the annotations to the json file
with open(json_path, 'w') as file:
    json.dump(data, file, indent=4)
print('json file updated')
    
