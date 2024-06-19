import json
import re
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
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
    

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16
)

device = 'cuda'


def qwenvl_inference(instruction, image_path, qwenvl_tokenizer, qwenvl_model):
    query = qwenvl_tokenizer.from_list_format([
        {'image': image_path},
        {'text': instruction},
    ])

    response, history = qwenvl_model.chat(qwenvl_tokenizer, query=query, history=None)

    return response


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


    model_name = "Qwen/Qwen-VL-Chat"
    print("model name: ", model_name)
    qwenvl_tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    qwenvl_model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", trust_remote_code=True, fp16=True).eval()
    qwenvl_model.eval()
    qwenvl_model.generation_config = GenerationConfig.from_pretrained(model_name, trust_remote_code=True)
    print("model params: ", count_parameters(qwenvl_model))

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
        pred = qwenvl_inference(instruction, image_path, qwenvl_tokenizer, qwenvl_model)
        print("[pred]: ", pred)

        sample["input"] = instruction
        sample["output"] = pred

        results.append(sample)
    
    with open(write_path, "w") as f_w:
        json.dump(results, f_w, indent=2, ensure_ascii=False)


if __name__ == "__main__":

    main()
    # model_name = "Qwen/Qwen-VL-Plus"
    # qwenvl_tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    # qwenvl_model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", trust_remote_code=True, fp16=True).eval()
    # qwenvl_model.eval()
    # qwenvl_model.generation_config = GenerationConfig.from_pretrained(model_name, trust_remote_code=True)
    # print("model params: ", count_parameters(qwenvl_model))

    # print(qwenvl_inference("what does this image describe?", "../YESBUT_cropped_yesbut/00253.jpg", qwenvl_tokenizer, qwenvl_model))