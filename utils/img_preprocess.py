""" 
This script processes a collection of images, cropping each image based on predefined coordinates and saving the cropped images to an output folder.

Usage:
    python utils/img_preprocess.py 
        [--img_folder INPUT_FOLDER] 
        [--output_folder OUTPUT_FOLDER]
        [--metadata METADATA]
        [--crop_type CROP_TYPE]

Arguments:
    - img_folder (str): Path to the folder containing input images. Default: "./data/YESBUT_ori/"
    - output_folder (str): Path to the folder to save the cropped images. Default: "./data/YESBUT_cropped_yesbut"
    - metadata (str): Path to the folder to save the cropped images. Default: "./data/YESBUT_dataset_info.json"
    - crop_type (str): options are: ("yesbut", "yes", "but"). Default: "yesbut"
"""
############################

import os
import glob
import argparse
   
from PIL import Image
import json
import requests
from io import BytesIO


def download_ori_images_from_link(sample_url, input_path=None):
    # read links from a json and download the images
    response = requests.get(sample_url)
    img = Image.open(BytesIO(response.content))

    if input_path:
        img.save(input_path)

    return img


def crop_image(input_path, output_path, bounding_box, sample_url, sv_ori=False):
    """
    Load image from input_path, crop it based on upper_left and lower_bottom coordinates, and save to output_path.
    
    Args:
    input_path (str): Path to the input image file.
    output_path (str): Path to save the cropped image.
    upper_left (tuple): Tuple containing the (x, y) coordinates of the upper left corner of the crop.
    lower_bottom (tuple): Tuple containing the (x, y) coordinates of the lower bottom corner of the crop.
    """
    # upper_left = (10, 105)  
    # lower_bottom = (870, 640)
    # print(bounding_box[0][0], bounding_box[0][1])
    upper_left = tuple(bounding_box[0][::-1]) # Upper left corner coordinates (x, y)
    lower_bottom = tuple(bounding_box[1][::-1]) # Lower bottom corner coordinates (x, y)
    try:
        try:
            # Open the image
            img = Image.open(input_path)
        except:
            img = download_ori_images_from_link(sample_url)
            # img = download_ori_images_from_link(sample_url, input_path)


        # Crop the image
        cropped_img = img.crop((*upper_left, *lower_bottom))

        # Save the cropped image
        cropped_img.save(output_path)
        print(f"Image {output_path} cropped and saved successfully!")
    except Exception as e:
        print(f"Error: {e}")


def get_BB_from_json(sample, crop_type="yesbut"):
    # type="yesbut"
    BB_list = sample['bounding_box']
    if crop_type == "yesbut":
        # Extract the minimum and maximum values
        min_x = min(box[0][0] for box in BB_list)
        min_y = min(box[0][1] for box in BB_list)
        max_x = max(box[1][0] for box in BB_list)
        max_y = max(box[1][1] for box in BB_list)

        # Create the resulting bounding box coordinates
        resulting_box = [[min_x, min_y], [max_x, max_y]]

    elif crop_type == "yes":
        # Find index of "YES"
        yes_ind = sample['labels'].index(0)

        # Get BB for "YES"
        resulting_box = BB_list[yes_ind]

        ## Get all BBs for "YES"
        # yes_ind = [i for i, elem in enumerate(sample['labels']) if elem == 0]
        # resulting_box = [BB_list[i] for i in yes_ind]

    elif crop_type == "but":
        # Find index of "BUT"
        but_ind = sample['labels'].index(1)

        # Get BB for "BUT"
        resulting_box = BB_list[but_ind]

        # # Get all BBs for "BUT"
        # but_ind = [i for i, elem in enumerate(sample['labels']) if elem == 1]
        # resulting_box = [BB_list[i] for i in but_ind]

    else:
        print(BB_list)

    return resulting_box


    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_folder', type=str, required=False, default="./yu/data/YESBUT_ori/")
    parser.add_argument('--output_folder', type=str, required=False, default="./yu/data/YESBUT_cropped_yesbut")
    parser.add_argument('--metadata', type=str, required=False, default="./yu/data/YESBUT_dataset_info.json")
    parser.add_argument('--crop_type', type=str, required=False, default="yesbut", help="yesbut, yes, but")
    args = parser.parse_args()


    with open(args.metadata, 'r', encoding='utf-8') as json_file:
        data = json.load(json_file)
    # import ipdb; ipdb.set_trace()

    # input_paths = sorted(glob.glob(os.path.join(args.img_folder, f'*.png')))
    # input_paths += sorted(glob.glob(os.path.join(args.img_folder, f'*.jpg')))

    # Check if the input/output folder exists, if not, create it
    if not os.path.exists(args.img_folder):
        os.makedirs(args.img_folder)
    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)

    for sample in data:
        print(sample['filename'])
        image_name = sample['filename'] 
        input_path = os.path.join(args.img_folder, image_name)
        output_path = os.path.join(args.output_folder, image_name)

        # get bounding_boxs 
        bounding_box = get_BB_from_json(sample, crop_type=args.crop_type)
        # if len(bounding_box)>1:
        #     import ipdb; ipdb.set_trace()

        # crop images
        crop_image(input_path, output_path, bounding_box, sample['url'])


