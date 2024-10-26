import json
import requests
import os
import sys
import argparse

def download_images(json_file, folder='images'):
    # Ensure the folder exists
    if not os.path.exists(folder):
        os.makedirs(folder)

    # Read the JSON file
    with open(json_file, 'r') as f:
        data = json.load(f)

    # Loop over each entry in the JSON data
    for d in data:
        try:
            # Construct the full file path
            file_path = os.path.join(folder, d['image_file'])
            # Download the image
            response = requests.get(d['url'])
            response.raise_for_status()  # Will raise an HTTPError if the HTTP request returned an unsuccessful status code
            
            # Write the image to a file
            with open(file_path, 'wb') as f:
                f.write(response.content)
        except requests.exceptions.RequestException as e:
            print(f"Error downloading {d['url']}: {e}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--json_file', type=str, default='data/YesBut_data.json', help="annotation file")
    parser.add_argument('--save_folder', type=str, default='data/YesBut_images')

    opt = parser.parse_args()
    download_images(json_file=opt.json_file, folder=opt.save_folder)