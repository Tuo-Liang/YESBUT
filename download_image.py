import json
import requests
import os
import sys

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

# Usage
json_file = 'YesBut_data.json'
folder = 'images'
download_images(json_file, folder)
