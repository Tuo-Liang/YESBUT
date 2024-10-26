- Prompts Set 1
{
    "prompts_caption_gen":[
        "The given comic shows the same situation from two opposite sides with contradictions. Write a one-paragraph literal description to describe the narrative of the comic."
    ],
    "prompts_contradiction_gen_w_caption": [
        "The given comic shows the same situation from two opposite sides with contradictions.\nThe literal caption of the comic is: {image_caption}\nWrite a short explanation to illustrate the contradiction of the two sides."
    ],
    "prompts_contradiction_gen_wo_caption": [
        "The given comic shows the same situation from two opposite sides with contradictions. Write a short explanation to illustrate the contradiction of the two sides."
    ],
    "prompts_philosophy_gen_w_caption": [
        "The given comic shows the same situation from two opposite sides with contradictions.\nThe literal caption of the comic is: {image_caption}\nWhich of the following options best represents the underlying philosophy of the comic?\n{philosophy_options}\n\nJust output the choice:"
    ],
    "prompts_philosophy_gen_wo_caption": [
        "The given comic shows the same situation from two opposite sides with contradictions.\nWhich of the following options best represents the underlying philosophy of the comic?\n{philosophy_options}\n\nJust output the choice:"
    ]
    "prompts_title_gen_w_caption": [
        "The given comic shows the same situation from two opposite sides with contradictions.\nThe literal caption of the comic is: {image_caption}\nWhich of the following titles are the most suitable for the comic?\n{title_options}\n\nJust output the choice:"
    ],
    "prompts_title_gen_wo_caption": [
        "The given comic shows the same situation from two opposite sides with contradictions.\nWhich of the following titles are the most suitable for the comic?\n{title_options}\n\nJust output the choice:"
    ]
}


- Prompts Set 2 
{
    "prompts_caption_gen":[
        "Please literally describe the context of the image in detail."
    ],
    "prompts_contradiction_gen_w_caption": [
        "Analyze the provided image with the following description: {image_caption}. Identify and concisely describe the contradiction depicted in the image in one or two sentences."
    ],
    "prompts_contradiction_gen_wo_caption": [
        "Analyze the provided image, which is divided into two or more panels, each illustrating contrasting views of the same scenario. Describe the elements visible in each panel. Then concisely interpret how these elements convey contrasting perspectives in one or two sentences. Focus and only output the contradiction."
    ],
    "prompts_philosophy_gen_w_caption": [
        "You are presented with an image with the following description: {image_caption}. \nWhich of the following options best represents the philosophy of the image provided? \n{philosophy_options} \nSelect the correct option by typing the corresponding letter (A, B, C, or D)."
    ],
    "prompts_philosophy_gen_wo_caption": [
        "You are presented with an image, which is divided into two or more panels, each illustrating contrasting views of the same scenario. \nWhich of the following options best represents the philosophy of the image provided? \n{philosophy_options} \nSelect the correct option by typing the corresponding letter (A, B, C, or D)."
    ]
    "prompts_title_gen_w_caption": [
        "You are presented with an image with the following description: {image_caption}. \nWhich of the following title options best represents the image provided? \n{title_options} \nSelect the correct option by typing the corresponding letter (A, B, C, or D)."
    ],
    "prompts_title_gen_wo_caption": [
        "You are presented with an image, which is divided into two or more panels, each illustrating contrasting views of the same scenario. \nWhich of the following title options best represents the image provided? \n{title_options} \nSelect the correct option by typing the corresponding letter (A, B, C, or D)."
    ]
}

- Prompts Set 3
{
    "prompts_caption_gen":[
        "Give me a detailed literally description of the image."
    ],
    "prompts_contradiction_gen_w_caption": [
        "Based on the following image's description: {image_caption}. Give me the concise contradiction depicted in the image in one or two sentences."
    ],
    "prompts_contradiction_gen_wo_caption": [
        "Given an image, the image is divided into two or more panels. There is the contrast relationship in the image through panels. Describe the elements visible in each panel. Give me the concise interpretation how these panels convey contrasting perspectives, which you only need to output the contradiction in one or two sentences."
    ],
    "prompts_philosophy_gen_w_caption": [
        "Given an image with the following description: {image_caption}. \nTell me the best option in the following options who represents the deep semantic of the image? \n{philosophy_options} \nJust tell me the correct option by outputing corresponding letter (A, B, C, or D), no more explanation."
    ],
    "prompts_philosophy_gen_wo_caption": [
        "Given an image, which has two or more panels. There is contrast in these panels. \nTell me the best option in the following options who represents the deep semantic of the image? \n{philosophy_options} \nJust tell me the correct option by outputing corresponding letter (A, B, C, or D), no more explanation."
    ]
    "prompts_title_gen_w_caption": [
        "Given an image with the following description: {image_caption}. \nTell me the best title in the following title options who represents the image? \n{title_options} \n Just tell me the correct option by outputing corresponding letter (A, B, C, or D), no more explanation."
    ],
    "prompts_title_gen_wo_caption": [
        "Given an image, the image is divided into two or more panels. There is the contrast relationship in the image through panels. \nTell me the best title in the following title options who represents the image? \n{title_options} \nJust tell me the correct option by outputing corresponding letter (A, B, C, or D), no more explanation."
    ]
}
