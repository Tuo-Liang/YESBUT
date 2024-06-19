import json
import re
from PIL import Image
import torch
from transformers import BitsAndBytesConfig
from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
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
    

device = 'cuda'


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def instructblip_inference(instruction, image_path, insturct_blip_processor, insturct_blip_model):
    image = Image.open(image_path)
    inputs = insturct_blip_processor(images=image, text=instruction, return_tensors="pt").to(device)

    outputs = insturct_blip_model.generate(
        **inputs,
        # do_sample=False,
        # num_beams=5,
        max_length=512,
        # min_length=1,
        # top_p=0.9,
        # repetition_penalty=1.5,
        # length_penalty=1.0,
        # temperature=1,
    )
    generated_text = insturct_blip_processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()
    print(generated_text)
    
    return generated_text



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--read_path', type=str, required=False)
    parser.add_argument('--write_path', type=str, required=False)
    parser.add_argument('--task', type=str, required=True)
    parser.add_argument('--use_caption', type=str2bool, default=False, required=True)
    parser.add_argument('--image_folder', type=str, required=True)
    parser.add_argument('--prompt_id', type=str, default="0", required=True)
    parser.add_argument('--model_size', type=str, default="7b", required=True)

    args = parser.parse_args()
    print(args)
    read_path = args.read_path
    write_path = args.write_path
    task = args.task
    use_caption = args.use_caption
    image_folder = args.image_folder
    prompt_id = int(args.prompt_id)

    if args.model_size == "7b":
        model_name = "Salesforce/instructblip-vicuna-7b"
    else:
        model_name = "Salesforce/instructblip-vicuna-13b"
    insturct_blip_model = InstructBlipForConditionalGeneration.from_pretrained(model_name, device_map="auto", torch_dtype=torch.float16)
    insturct_blip_model = insturct_blip_model.half()
    insturct_blip_processor = InstructBlipProcessor.from_pretrained(model_name)
    insturct_blip_model.eval()
    print("model params: ", count_parameters(insturct_blip_model))

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
        pred = instructblip_inference(instruction, image_path, insturct_blip_processor, insturct_blip_model)
        print("[pred]: ", pred)

        sample["input"] = instruction
        sample["output"] = pred

        results.append(sample)
    
    with open(write_path, "w") as f_w:
        json.dump(results, f_w, indent=2, ensure_ascii=False)
        
        

if __name__ == "__main__":
    
    main()
    