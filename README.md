# Cracking the Code of Juxtaposition: Can AI Models Understand the Humorous Contradictions (NeurIPS'24 Oral)

This repository provides the official *data release* and *code implementation* of our paper:

Hu Z, Liang T, Li J, Lu Y, Zhou Y, Qiao Y, Ma J, Yin Y., **Cracking the Code of Juxtaposition: Can AI Models Understand the Humorous Contradictions**, NeurIPS, 2024.

### [[Paper](https://neurips.cc/virtual/2024/oral/97967)] [[Arxiv](https://arxiv.org/pdf/2405.19088)] [[Webpage](https://vulab-ai.github.io/YESBUT_Homepage/)]


## Our Goals

We aim to challenge AI systems in their ability to recognize and interpret visual humor, grasp nuances in human behavior, comprehend wordplay, and appreciate cultural references. This understanding can enhance AI's ability to interact with users, generate creative content, and interpret multimedia content more effectively, thereby improving user experience in various applications such as content recommendation systems, virtual assistants, and automated content creation tools.

We collect and annotate images which convey various forms of visual humor and storytelling through simple comic panels. They explore themes such as human behavior, animal antics, and wordplay, often leading to unexpected or ironic conclusions.
<div align='left'><img src="./samples/samples.jpg"  alt="NAME" width="100%"/></div>

## Dataset

### Download
- Annotation File: The annotated data is available [here: data/YesBut_data.json](https://github.com/Tuo-Liang/YESBUT/blob/main/data/YesBut_data.json).

- Image Download: Download the associated images by running the following command:
```
python download_images.py --json_file='data/YesBut_data.json' --save_folder='data/YesBut_images'
```
This will save the images to the specified `data/YesBut_images` folder.


### Annotated data format
- The file is in `/data/YesBut_data.json`
- The file has the format such as following.
```
 {
        "image_file": "00001.jpg",
        "description": "The comic is divided into two panels, each presenting a contradictory perspective of the same object\u2014a mug. In the first panel, the mug is illustrated as an adorable fox with closed eyes, giving off a serene and cute vibe. It's an object that one would admire or find endearing. However, in the second panel, we see a person drinking from this fox-shaped mug. The contradiction lies in the mug's impracticality: its ears and head protrude awkwardly, obstructing the person's ability to sip comfortably. Despite its endearing appearance, the mug fails its primary function as a practical vessel for beverages.",
        "caption": "The comic is divided into two panels, each presenting a contradictory perspective of the same object\u2014a mug. In the first panel, the mug is illustrated as an adorable fox with closed eyes, giving off a serene and cute vibe. It's an object that one would admire or find endearing. However, the second panel reveals a practical issue: a person attempts to drink from the fox-shaped mug, but its design\u2014featuring protruding ears and head\u2014awkwardly interferes, complicating the act of sipping comfortably.",
        "contradiction": "The comic illustrates a contradiction where a mug designed as an adorable fox is charming to look at but proves impractical to use due to its awkwardly protruding ears and head that hinder drinking.",
        "moral": "The illustration critiques the clash between aesthetics and usability, emphasizing the need for a balanced consideration of both to ensure a harmonious and practical experience in any aspect of life.",
        "title": "Charming Design, Prickly Reality: The Fox Mug's Surprise",
        "neg_title": [
            "A Toast to Vulpine Grace",
            "Harmony in a Sip",
            "Enchanting Elixir: The Fox's Secret Brew"
        ],
        "neg_moral": [
            "The comic shows that adding more decorative elements to an object will enhance its value and enjoyment, when in fact, the opposite is true in this case.",
            "The illustration suggests that the initial charming appearance of an item will always lead to a positive overall experience, disregarding any practical complications that arise later.",
            "The image shows enduring inconvenience is a worthwhile sacrifice for the sake of owning something that looks unique or cute."
        ],
        "category": "the theme of expectation versus reality",
        "moral_mcq": "A. The comic shows that adding more decorative elements to an object will enhance its value and enjoyment, when in fact, the opposite is true in this case.\nB. The illustration critiques the clash between aesthetics and usability, emphasizing the need for a balanced consideration of both to ensure a harmonious and practical experience in any aspect of life.\nC. The illustration suggests that the initial charming appearance of an item will always lead to a positive overall experience, disregarding any practical complications that arise later.\nD. The image shows enduring inconvenience is a worthwhile sacrifice for the sake of owning something that looks unique or cute.",
        "moral_mcq_answer": "B",
        "title_mcq": "A. A Toast to Vulpine Grace\nB. Charming Design, Prickly Reality: The Fox Mug's Surprise\nC. Enchanting Elixir: The Fox's Secret Brew\nD. Harmony in a Sip",
        "title_mcq_answer": "B",
        "url": "https://pbs.twimg.com/media/F9Y1i8zXIAEtKy-?format=jpg&name=medium",
        "bounding_box": [
            [
                [
                    270,
                    12
                ],
                [
                    919,
                    529
                ]
            ],
            [
                [
                    270,
                    551
                ],
                [
                    919,
                    1068
                ]
            ]
        ]
    }

```

## Experimental Design

### Experimental Setting
- Sample components: (image, caption, contradiction, philosophy, title)

#### Task 1: Description Generation
- Image Setting: p(description|image)
  
#### Task 2: Contradiction Generation
- Image Setting: p(contradiction|image)
- Full Setting: p(contradiction|image, caption)
	- oracle caption: written by annotators (upper bound)
 	- system caption: generated by VLM itself

#### Task 3: Title MCQ
- Image Setting: p(title_option|image)
- Full Setting: p(title_option|image, caption)
	- oracle caption: written by annotators (upper bound)
 	- system caption: generated by VLM itself

#### Task 4: Deep Philosophy MCQ
- Image Setting: p(philosophy_option|image)
- Full Setting: p(philosophy_option|image, caption)
	- oracle caption: written by annotators (upper bound)
 	- system caption: generated by VLM itself


## Evaluation
Modify the "predict_model_name.sh".
'''
#Task claude3 as an example
data="annotated_data/data_annotation.json"
image_folder="YESBUT_cropped_yesbut"
write_path_surffix=".json"
#task options: contradiction | moral_mcq | title_mcq

use_caption=False

task="contradiction"
echo "==============================="
echo "claude3 eval"
echo "==============================="
python3 -u predict_claude_opus.py \
    --read_path ${data} \
    --write_path "results/results_claude3_"${task}"_"${write_path_surffix} \
    --task ${task} \
    --use_caption ${use_caption} \
    --image_folder ${image_folder}

'''

Then run the command:
'''
bash predict_model_name.sh
'''

## Citation
```
@article{hu2024cracking,
  title={Cracking the Code of Juxtaposition: Can AI Models Understand the Humorous Contradictions},
  author={Hu, Zhe and Liang, Tuo and Li, Jing and Lu, Yiren and Zhou, Yunlai and Qiao, Yiran and Ma, Jing and Yin, Yu},
  journal={arXiv preprint arXiv:2405.19088},
  year={2024}
}

```





