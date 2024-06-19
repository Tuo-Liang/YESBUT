import json
import re
from transformers import BitsAndBytesConfig
from instruction_generation_yesbut_all import formulate_instruction

#---
import torch
import requests
from PIL import Image
from transformers import AutoModelForCausalLM, LlamaTokenizer

tokenizer = LlamaTokenizer.from_pretrained('lmsys/vicuna-7b-v1.5')
model = AutoModelForCausalLM.from_pretrained(
    'THUDM/cogvlm-chat-hf',
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    trust_remote_code=True
).to('cuda').eval()
#---


def llava_inference(instruction, image_path):
    prompt = f"<image>\nUSER: {instruction}\nASSISTANT:"
    # image_path = "data/images/sample_redlight.png"
    image = Image.open(image_path)

    query = prompt

    inputs = model.build_conversation_input_ids(tokenizer, query=query, history=[], images=[image])  # chat mode
    inputs = {
        'input_ids': inputs['input_ids'].unsqueeze(0).to('cuda'),
        'token_type_ids': inputs['token_type_ids'].unsqueeze(0).to('cuda'),
        'attention_mask': inputs['attention_mask'].unsqueeze(0).to('cuda'),
        'images': [[inputs['images'][0].to('cuda').to(torch.bfloat16)]],
        }   
    gen_kwargs = {"max_length": 2048, "do_sample": False}

    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)
        outputs = outputs[:, inputs['input_ids'].shape[1]:]
        result=tokenizer.decode(outputs[0])

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
        

if __name__ == "__main__":
   
    main()
    
