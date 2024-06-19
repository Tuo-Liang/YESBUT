import json
import re
from transformers import BitsAndBytesConfig
from instruction_generation_yesbut_2 import formulate_instruction
from instruction_generation_yesbut_2 import *

#---
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
import torch
from PIL import Image
from accelerate import dispatch_model
#---



llava_model = LlavaNextForConditionalGeneration.from_pretrained("llava-hf/llava-v1.6-vicuna-13b-hf",  device_map="auto",torch_dtype=torch.float16) 

device = 'cuda'

llava_processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-vicuna-13b-hf",use_fast=True)


def llava_inference(instruction, image_path):
    if len(instruction)>2048:
        instruction = instruction[:2048]  

    prompt = f"<image>\nUSER: {instruction}\nASSISTANT:"
    # image_path = "data/images/sample_redlight.png"
    image = Image.open(image_path)

    inputs = llava_processor(text=prompt, images=image, return_tensors="pt")
    inputs.to(device)
    # Generate
    generate_ids = llava_model.generate(**inputs, max_length=3096)
    result = llava_processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    return result


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
    use_caption = args.use_caption
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
        pred = llava_inference(instruction, image_path)
        if "ASSISTANT: " in pred:
            pred = pred.split("ASSISTANT: ")[1].strip()
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
    use_caption = args.use_caption
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
            cur_caption= llava_inference(prompt, image_path)
            if "ASSISTANT: " in cur_caption:
                cur_caption = cur_caption.split("ASSISTANT: ")[1].strip()
            print('[Gen_caption]:',cur_caption)


        instruction = formulate_instruction(sample, caption=cur_caption, task=task)
        image_file = sample["image_file"]
        image_path = image_folder + "/" + image_file

        print("[input]: ", instruction)
        pred = llava_inference(instruction, image_path)
        if "ASSISTANT: " in pred:
            pred = pred.split("ASSISTANT: ")[1].strip()
        print("[pred]: ", pred)

        sample["input"] = instruction
        sample['gen_des']=cur_caption
        sample["output"] = pred

        results.append(sample)
    
    with open(write_path, "w") as f_w:
        json.dump(results, f_w, indent=2, ensure_ascii=False)

if __name__ == "__main__":
   
    main2()
    
