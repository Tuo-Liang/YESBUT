import json
import os
import anthropic
from anthropic import AnthropicVertex

import time
#from nltk.tokenize import word_tokenize
import httpx
import base64
from mimetypes import guess_type
import argparse
from instruction_generation_yesbut_1 import formulate_instruction
from instruction_generation_yesbut_1 import *


projectid = "your projectid"


LOCATION = "us-east5"

client = AnthropicVertex(region=LOCATION, project_id=projectid)



# Function to encode a local image into data URL 
def local_image_to_data_url(image_path):
    # Guess the MIME type of the image based on the file extension
    mime_type, _ = guess_type(image_path)
    if mime_type is None:
        mime_type = 'application/octet-stream'  # Default MIME type if none is found

    # Read and encode the image file
    with open(image_path, "rb") as image_file:
        base64_encoded_data = base64.b64encode(image_file.read()).decode('utf-8')

    # Construct the data URL
    return f"data:{mime_type};base64,{base64_encoded_data}"


def claude3_vision_generation(input_prompt, image_path, model="claude-3-opus-20240229", temperature=1):
    

    data_url = local_image_to_data_url(image_path)
    image_media_type="image/jpeg"
    image_data = data_url.split(",")[1] 


    for _ in range(10):
        try:
            response = client.messages.create(model="claude-3-opus@20240229",
                                            max_tokens=1024,
                                            temperature=temperature,
                                            messages=[
                                                {
                                                    "role": "user",
                                                    "content": [
                                                                {
                                                                    "type": "image",
                                                                    "source": {
                                                                        "type": "base64",
                                                                        "media_type": image_media_type,
                                                                        "data": image_data,
                                                                    },
                                                                },
                                                                {
                                                                    "type": "text",
                                                                    "text": input_prompt
                                                                }
                                                                ],
                                                }
                                            ],
                                            )

            
            if response is not None:
                break
        except Exception as e:
            #print(["[CLAUDE3 ERROR]: ", [e]])
            response = None
            time.sleep(5)
    if response != None:
        #print(response.content[0].text)
        response = response.content[0].text
    return response



def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--read_path', type=str, required=False)
    parser.add_argument('--write_path', type=str, required=False)
    parser.add_argument('--task', type=str, required=True)
    parser.add_argument('--use_caption', type=bool, default=False, required=True)
    parser.add_argument('--image_folder', type=str, required=True)

    args = parser.parse_args()
    print(args)

    read_path = args.read_path
    write_path = args.write_path
    task = args.task
    use_caption = False
    image_folder = args.image_folder


    data = json.load(open(read_path))
    results = []

    for sample in data:
        if use_caption:
            cur_caption = sample["caption"] # Now this is for oracle caption. Need to add predicted caption
        else:
            cur_caption = None
        instruction = formulate_instruction(sample, caption=cur_caption, task=task)
        image_file = sample["image_file"]
        image_path = image_folder + "/" + image_file

        print("[input]: ", instruction)
        pred = claude3_vision_generation(instruction, image_path)
        print("[pred]: ", pred)

        sample["input"] = instruction
        sample["output"] = pred

        results.append(sample)
    
    with open(write_path, "w") as f_w:
        json.dump(results, f_w, indent=2, ensure_ascii=False)


def main2():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--read_path', type=str, required=False)
    parser.add_argument('--write_path', type=str, required=False)
    parser.add_argument('--task', type=str, required=True)
    parser.add_argument('--use_caption', type=bool, default=False, required=True)
    parser.add_argument('--image_folder', type=str, required=True)
    parser.add_argument('--gen_des',type=bool,default=True,required=True)

    args = parser.parse_args()
    print(args)
    read_path = args.read_path
    write_path = args.write_path
    task = args.task
    use_caption = True
    image_folder = args.image_folder
    gen_des=args.gen_des


    data = json.load(open(read_path))
    results = []

    for sample in data:
        
        if use_caption:
            cur_caption = sample["caption"] # Now this is for oracle caption. Need to add predicted caption
        else:
            cur_caption = None

        if gen_des:
            print("---------generate caption--------")
            image_file = sample["image_file"]
            image_path = image_folder + "/" + image_file

            prompt=gen_description()
            print("[Prompt]:",prompt)
            cur_caption= claude3_vision_generation(prompt, image_path)
            if "ASSISTANT: " in cur_caption:
                cur_caption = cur_caption.split("ASSISTANT: ")[1].strip()
            print('[Gen_caption]:',cur_caption)


        instruction = formulate_instruction(sample, caption=cur_caption, task=task)
        image_file = sample["image_file"]
        image_path = image_folder + "/" + image_file

        print("[input]: ", instruction)
        pred = claude3_vision_generation(instruction, image_path)
        if "ASSISTANT: " in pred:
            pred = pred.split("ASSISTANT: ")[1].strip()
        print("[pred]: ", pred)

        sample["input"] = instruction
        sample['gen_des']=cur_caption
        sample["output"] = pred

        results.append(sample)
    
    with open(write_path, "w") as f_w:
        json.dump(results, f_w, indent=2, ensure_ascii=False)

def main3():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--read_path', type=str, required=False)
    parser.add_argument('--write_path', type=str, required=False)
    parser.add_argument('--task', type=str, required=True)
    parser.add_argument('--use_caption', type=bool, default=False, required=True)
    parser.add_argument('--image_folder', type=str, required=True)
    parser.add_argument('--gen_des',type=bool,default=True,required=True)

    args = parser.parse_args()
    print(args)
    read_path = args.read_path
    write_path = args.write_path
    task = args.task
    use_caption = True
    image_folder = args.image_folder
    gen_des=args.gen_des


    data = json.load(open(read_path))
    results = []

    for sample in data:
        
        if use_caption:
            cur_caption = sample["caption"] # Now this is for oracle caption. Need to add predicted caption
        else:
            cur_caption = None

        if gen_des:
            print("---------generate caption--------")
            image_file = sample["image_file"]
            image_path = image_folder + "/" + image_file

            prompt=gen_description()
            print("[Prompt]:",prompt)
            cur_caption= claude3_vision_generation(prompt, image_path)
            if "ASSISTANT: " in cur_caption:
                cur_caption = cur_caption.split("ASSISTANT: ")[1].strip()
            print('[Gen_caption]:',cur_caption)
            sample['prompt']=prompt



        sample['gen_des']=cur_caption


        results.append(sample)
    
    with open(write_path, "w") as f_w:
        json.dump(results, f_w, indent=2, ensure_ascii=False)

if __name__ == "__main__":
   
    main()

