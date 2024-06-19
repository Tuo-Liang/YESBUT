import json
import os
import openai
# from openai import OpenAI
import openai
import time
from nltk.tokenize import word_tokenize
import base64
from mimetypes import guess_type
import argparse
from instruction_generation_yesbut import formulate_instruction
import argparse


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true'):
        return True
    elif v.lower() in ('no', 'false'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


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


def gpt4_vision_generation(input_prompt, image_path, model="gpt-4-vision-preview", temperature=1):
    API_KEY = ""  # your api_key

    data_url = local_image_to_data_url(image_path)

    for _ in range(10):
        try:
            response = openai.ChatCompletion.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": [
                        { 
                            "type": "text", 
                            "text": input_prompt 
                        },
                        { 
                            "type": "image_url",
                            "image_url": {
                                "url":data_url
                            }
                        }
                        ]}
                ],
                temperature=temperature,
                max_tokens=2048,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0,
                api_key=API_KEY,
                # api_base = API_BASE
            )
            if response is not None:
                break
        except Exception as e:
            print(["[OPENAI ERROR]: ", [e]])
            response = None
            time.sleep(5)
    if response != None:
    # print(response)
        response = response.choices[0].message.content
    return response



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--read_path', type=str, required=False)
    parser.add_argument('--write_path', type=str, required=False)
    parser.add_argument('--task', type=str, required=True)
    parser.add_argument('--use_caption', type=str2bool, default=False, required=True)
    parser.add_argument('--image_folder', type=str, required=True)
    parser.add_argument('--prompt_id', type=str, default="0", required=True)

    args = parser.parse_args()
    print(args)

    read_path = args.read_path
    write_path = args.write_path
    task = args.task
    use_caption = args.use_caption
    image_folder = args.image_folder
    prompt_id = int(args.prompt_id)


    data = json.load(open(read_path))
    results = []

    for sample in data:
        if use_caption:
            cur_caption = sample["caption"] # Now this is for oracle caption. Need to add predicted caption
        else:
            cur_caption = None
        instruction = formulate_instruction(sample, caption=cur_caption, task=task)
        instruction = instruction[prompt_id]
        image_file = sample["image_file"]
        image_path = image_folder + "/" + image_file

        print("[input]: ", instruction)
        pred = gpt4_vision_generation(instruction, image_path)
        print("[pred]: ", pred)

        sample["input"] = instruction
        sample["output"] = pred

        results.append(sample)
    
    with open(write_path, "w") as f_w:
        json.dump(results, f_w, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()

