import json
import re
import requests
import argparse
from openai_generate import openai_generate
import re
import os
import numpy as np
import ast
import openai
import time


def openai_generate(input_prompt, model="gpt-3.5-turbo-0125", temperature=1):

    API_KEY = "" # Your openai key


    if model == "chatgpt":
        model = "gpt-3.5-turbo"
    elif model == "gpt4":
        response = "gpt-4"

    for _ in range(5):
        try:
            response = openai.ChatCompletion.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": input_prompt}
                ],
                temperature=temperature,
                max_tokens=2048,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0,
                api_key=API_KEY,
            )
            break
        except Exception as e:
            print(["[OPENAI ERROR]: ", e])
            response = None
            time.sleep(5)
    if response != None:
        # print(response)
        response = response.choices[0].message.content
    return response



def eval_one_caption(ref, gen):
    prompt = f'''
You need to determine how accurately a candidate literal description matches a reference literal description of a comic narrative.

- Candidate literal description:
{gen}

- Reference literal description:
{ref}

Using a scale from 1 to 5, rate the accuracy with which the candidate description matches the reference description, with 1 being the least accurate and 5 being the most accurate.
Please directly output a score by strictly following this format: \"[[score]]\", for example: \"Rating: [[3]]\".
'''
    judgment = openai_generate(prompt)
    one_score_pattern = re.compile("\[\[(\d+\.?\d*)\]\]")
    one_score_pattern_backup = re.compile("\[(\d+\.?\d*)\]")
    # print("[log-llm_judge_eval-judgement]: ", [judgment])
    match = re.search(one_score_pattern, judgment)
    if not match:
        match = re.search(one_score_pattern_backup, judgment)

    if match:
        rating = ast.literal_eval(match.groups()[0])
    else:
        rating = -1
    return rating


def evaluate_caption(read_path):
    data = json.load(open(read_path))
    score_list = []
    for sample in data:
        ref = sample["caption"]
        if "output" not in sample:
            gen = sample["gen_des"]
        else:
            gen = sample["output"]
        score = eval_one_caption(ref, gen)
        score_list.append(score)
    return score_list


def eval_one(caption, ref, gen):
    prompt = f'''
Background: You are an impartial judge. You will be given a literal description of a comic that presents the same situation from two opposing perspectives, highlighting contradictions. You will also be provided with a gold-standard illustration as reference that effectively demonstrates these narrative contradictions.

Your task is to evaluate the quality of a generated illustration and determine whether it accurately depicts the narrative contradictions in the comic. Then, assign a score on a scale of 1 to 5, where 1 is the lowest and 5 is the highest, based on its quality.

- The literal description of the comic:
{caption}

- The reference contradiction illustration:
{ref}

- The generated contradiction illustration:
{gen}

Please directly output a score by strictly following this format: \"[[score]]\", for example: \"Rating: [[3]]\".
'''
    judgment = openai_generate(prompt)
    one_score_pattern = re.compile("\[\[(\d+\.?\d*)\]\]")
    one_score_pattern_backup = re.compile("\[(\d+\.?\d*)\]")
    # print("[log-llm_judge_eval-judgement]: ", [judgment])
    match = re.search(one_score_pattern, judgment)
    if not match:
        match = re.search(one_score_pattern_backup, judgment)

    if match:
        rating = ast.literal_eval(match.groups()[0])
    else:
        rating = -1
    return rating


def evaluate_contradiction(read_path):
    data = json.load(open(read_path))
    score_list = []
    for sample in data:
        caption = sample["caption"]
        ref = sample["contradiction"]
        gen = sample["output"]
        score = eval_one(caption, ref, gen)
        score_list.append(score)
    return score_list


def eval_contradiction_all():
    folder = "./results/results_contradiction/"
    task = "contradiction"  # moral, title, contradiction, caption
    all_files = [fil for fil in os.listdir(folder) if task in fil]

    all_models = [elem.split("_prompt")[0].replace("results_", "").strip() for elem in all_files if "prompt" in elem]
    model_list = set(all_models)
    model_list = ['llava_contradiction', 'mistral_llavanext13bcaption_contradiction',
                  'instructblip_contradiction', 'claude3_contradiction', 'llama3_llavanext13bcaption_contradiction']
    print("models: ", model_list)

    for model in model_list:
        print(model)
        cur_model_files = [elem for elem in all_files if model in elem and "prompt" in elem]
        print(cur_model_files)

        scores = []
        for fil_name in cur_model_files:
            print("File: ", fil_name)
            score_list = evaluate_contradiction(folder + "/" + fil_name)
            scores.append(np.mean(score_list))
        if len(scores) > 0:
            print(model)
            print(scores)
            print(np.mean(scores))
            print()


def eval_description_all():
    folder = "./results/captions/"
    task = "caption"  # moral, title, contradiction, caption
    all_files = [fil for fil in os.listdir(folder) if task in fil]

    all_models = [elem.split("_prompt")[0].replace("results_", "").strip() for elem in all_files if "prompt" in elem]
    model_list = set(all_models)
    print("models: ", model_list)

    for model in model_list:
        print(model)
        cur_model_files = [elem for elem in all_files if model in elem and "prompt" in elem]
        print(cur_model_files)

        scores = []
        for fil_name in cur_model_files:
            print("File: ", fil_name)
            score_list = evaluate_caption(folder + "/" + fil_name)
            print(score_list[:5])
            scores.append(np.mean(score_list))
        if len(scores) > 0:
            print(model)
            print(scores)
            print(np.mean(scores))
            print()


if __name__ == "__main__":
    # eval_description_all()
    eval_contradiction_all()
   
