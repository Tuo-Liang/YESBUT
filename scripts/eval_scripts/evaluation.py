import json
import os
import re
import numpy as np
from sklearn.metrics import accuracy_score
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer, scoring
from nltk.tokenize import word_tokenize
import json
from google_bleu import compute_bleu
import numpy as np
import string
import collections
from nltk.util import ngrams
from meteor import meteor_score
from bert_score import score
from evaluate import load

# mauve = load('mauve')



def eval_meteor(gen_list, ref_list):
    score_list = []
    for gen, ref in zip(gen_list, ref_list):
        score = round(meteor_score([" ".join(ref)], " ".join(gen)),4)
        score_list.append(score)
    # return np.mean(score_list)
    # print("meteor score: ", np.mean(score_list))
    return  np.mean(score_list)


def eval_rouge(gen_list, ref_list):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    aggregator = scoring.BootstrapAggregator()
    for gen, ref in zip(gen_list, ref_list):
        gen_str = " ".join(gen)
        ref_str = " ".join(ref)
        cur_score = scorer.score(ref_str, gen_str)
        aggregator.add_scores(cur_score)

    aggregates = aggregator.aggregate()
    rouge_socre_dict = {}
    for score_type, aggregate in sorted(aggregates.items()):
        # print("%s-R,%f,%f,%f\n" %
        #      (score_type, aggregate.low.recall, aggregate.mid.recall,
        #      aggregate.high.recall))
        # print("%s-P,%f,%f,%f\n" %
        #      (score_type, aggregate.low.precision,
        #      aggregate.mid.precision, aggregate.high.precision))
        # print("%s-F,%f,%f,%f\n" %
        #      (score_type, aggregate.low.fmeasure,
        #      aggregate.mid.fmeasure, aggregate.high.fmeasure))
        rouge_socre_dict[score_type] = {
            "p": aggregate.mid.precision, 
            "r": aggregate.mid.recall, 
            "f": aggregate.mid.fmeasure
        }
    return rouge_socre_dict


def eval_bleu(gen_list, ref_list):
    reference_corpus = [[elem] for elem in ref_list]
    translation_corpus = gen_list

    bleu_score_dict = {}
    for max_order in [1, 2, 3, 4]:
        bleu_score = compute_bleu(
            reference_corpus, 
            translation_corpus, 
            max_order=max_order,
            smooth=True
            )
        bleu_score_dict[max_order] = bleu_score[0]
    
    chencherry = SmoothingFunction()
    nltk_bleu = corpus_bleu(
        reference_corpus, 
        translation_corpus, 
        smoothing_function=chencherry.method1
        )
    bleu_score_dict["nltk"] = nltk_bleu
    
    # return bleu_score_dict
    # for score_type in bleu_score_dict:
    #     print("bleu-", score_type, ": ", bleu_score_dict[score_type])
    return bleu_score_dict


def eval_bert_score(gen_list, ref_list):
    (P, R, F), hashname = score(gen_list, ref_list, lang="en", return_hash=True)
    # print(
    #     f"{hashname}: P={P.mean().item():.6f} R={R.mean().item():.6f} F={F.mean().item():.6f}"
    # )
    # roberta-large_L17_no-idf_version=0.3.12(hug_trans=4.39.3): P=0.851257 R=0.866510 F=0.858768
    bleu_score_dict = {"p": P.mean().item(), "r": R.mean().item(), "f": F.mean().item()}
    return bleu_score_dict


def eval_acc(pred_list, label_list):
    # each elem is a tuple of (pred, answer)
    options = ["A", "B", "C", "D", "E"]
    #options = list(set(label_list))
    parsed_pred_list = []
    for pred in pred_list:
        if pred is None:
            parsed_pred_list.append(-1)
            continue
        if "USER" in pred and "\nASSISTANT:" in pred:
            pred = pred.split("\nASSISTANT:")[1].strip()
        parsed_pred = ""
        for ch in pred:
            if ch in options:
                parsed_pred = ch
                break
        parsed_pred_list.append(parsed_pred)

    # print("[LOG-pred]: ", parsed_pred_list)
    # print("[LOG-label]: ", label_list)
    print(f"[LOG]: {len(parsed_pred_list)}, {len(label_list)}")
    print(f"[LOG]: {parsed_pred_list[:5]}, {label_list[:5]}")
    
    acc = len([i for i in range(len(pred_list)) if parsed_pred_list[i] == label_list[i]]) / len(label_list)
    return acc
            

def eval_one(read_path, task="moral"):
    data = json.load(open(read_path))
    if task == "moral":
        answer_list = [elem["moral_mcq_answer"] for elem in data]
    elif task == "title":
        answer_list = [elem["title_mcq_answer"] for elem in data]
    else:
        print("[TASK ERROR!!!]")
        return None
    pred_list = [elem["output"] for elem in data]

    accuracy = eval_acc(pred_list, answer_list)
    return accuracy


def eval_generation(read_path):
    data = json.load(open(read_path))

    gen_list = []
    ref_list = []
    for elem in data:
        if elem["output"] is None:
            continue
        gen_list.append(elem["output"])
        ref_list.append(elem["contradiction"])
    print(len(data), len(gen_list), len(ref_list))

    # print("=======evaluate bleu=======")
    bleu_score_dict = eval_bleu(gen_list, ref_list)

    # print("\n=======evaluate rouge=======")
    rouge_score_dict = eval_rouge(gen_list, ref_list)

    # print("\n=======evaluate meteor=======")
    meteor_score = eval_meteor(gen_list, ref_list)

    # print("\n=======evaluate bert score=======")
    bert_score = eval_bert_score(gen_list, ref_list)
    # bert_score = {"r": 0, "f": 0}

    # print("\n=======evaluate mauve score=======")
    # mauve_results = mauve.compute(predictions=gen_list, references=ref_list)
    # print(mauve_results.mauve)

    # print("\n=======evaluate bleurt score=======")
    bleurt = load("bleurt", "BLEURT-20", module_type="metric")
    bleurt_results = bleurt.compute(predictions=gen_list, references=ref_list)
    del bleurt

    # bleurt_results = {"scores": []}
    # for gen, ref in zip(gen_list, ref_list):
    #     cur_bleurt_results = bleurt.compute(predictions=[gen], references=[ref])


    return_dict = {
        "bleu_score_dict": bleu_score_dict,
        "rouge_score_dict": rouge_score_dict,
        "meteor_score": meteor_score,
        "bert_score": bert_score,
        "bleurt": np.mean(bleurt_results["scores"])
    }
    return return_dict


if __name__ == "__main__":
    model_list = ["instructblip", "qwenvl", "llava_", "llavanext", "mplug_owl2", "gpt4vision"]
    # model_list = ["llava"]
    folder = "../results_all/results_captionTrue/"
    task = "contradiction" # moral, title, contradiction
    all_files = [fil for fil in os.listdir(folder) if task in fil]


    for model in model_list:
        print(model)
        cur_model_files = [elem for elem in all_files if model in elem and "prompt" in elem]
        print(cur_model_files)
        

        if task == "contradiction":
            metrics = ["bleu3", "bleu4", "rouge2_f", "rougel_f", "rouge2_r", "rougel_r", "meteor", "bert_f", "bert_r", "bleurt"]
            score_dict = {}
            for metric in metrics:
                score_dict[metric] = []
            for idx, fil_name in enumerate(cur_model_files):
                print("File: ", fil_name)
                cur_score = eval_generation(folder + "/" + fil_name)
                print(cur_score)
                score_dict["bleu3"] += [cur_score["bleu_score_dict"][3]]
                score_dict["bleu4"] += [cur_score["bleu_score_dict"][4]]
                score_dict["rouge2_f"] += [cur_score["rouge_score_dict"]["rouge2"]["f"]]
                score_dict["rougel_f"] += [cur_score["rouge_score_dict"]["rougeL"]["f"]]
                score_dict["rouge2_r"] += [cur_score["rouge_score_dict"]["rouge2"]["r"]]
                score_dict["rougel_r"] += [cur_score["rouge_score_dict"]["rougeL"]["r"]]
                score_dict["meteor"] += [cur_score["meteor_score"]]
                score_dict["bert_r"] += [cur_score["bert_score"]["r"]]
                score_dict["bert_f"] += [cur_score["bert_score"]["f"]]
                score_dict["bleurt"] += [cur_score["bleurt"]]

            print(score_dict)
            print(f"---------avg. {model}----------")
            for key in score_dict:
                print(key, np.mean(score_dict[key]))
            print()

        else:
            scores = []
            for fil_name in cur_model_files:
                print("File: ", fil_name)
                acc = eval_one(folder + "/" + fil_name, task)
                scores.append(acc)
            if len(scores) > 0:
                print(scores)
                print(np.mean(scores))
                print()

