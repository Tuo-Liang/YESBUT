import json
import re
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration
import torch
from transformers import BitsAndBytesConfig
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


def llava_inference(instruction, image_path, llava_processor, llava_model):
    prompt = f"<image>\nUSER: {instruction}\nASSISTANT:"
    # image_path = "data/images/sample_redlight.png"
    image = Image.open(image_path)

    inputs = llava_processor(text=prompt, images=image, return_tensors="pt")
    inputs.to(device)
    # Generate
    generate_ids = llava_model.generate(**inputs, max_length=512)
    result = llava_processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--read_path', type=str, required=False)
    parser.add_argument('--write_path', type=str, required=False)
    parser.add_argument('--task', type=str, required=True)
    parser.add_argument('--use_caption', type=str2bool, required=True)
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
        llava_7b_path = "llava-hf/llava-1.5-7b-hf"
    else:
        llava_7b_path = "llava-hf/llava-1.5-13b-hf"
    print("model name: ", llava_7b_path)
    llava_model = LlavaForConditionalGeneration.from_pretrained(llava_7b_path, torch_dtype=torch.float16, device_map="auto")
    llava_processor = AutoProcessor.from_pretrained(llava_7b_path)
    llava_model.eval()
    print("model params: ", count_parameters(llava_model))


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
        pred = llava_inference(instruction, image_path, llava_processor, llava_model)
        if "ASSISTANT: " in pred:
            pred = pred.split("ASSISTANT: ")[1].strip()
        print("[pred]: ", pred)

        sample["input"] = instruction
        sample["output"] = pred

        results.append(sample)
    
    with open(write_path, "w") as f_w:
        json.dump(results, f_w, indent=2, ensure_ascii=False)
        

if __name__ == "__main__":
   
    main()
    
