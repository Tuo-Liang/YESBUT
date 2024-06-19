conda activate base
data="../annotated_data/data_annotation.json"
image_folder="../YESBUT_cropped_yesbut"
#task options: contradiction | moral_mcq | title_mcq
task="title_mcq"
use_caption=False
model_size="13b"
prompt_ids=(0 1 2)
CUDA_DEVICE=2

for prompt_id in "${prompt_ids[@]}"
do
    echo "==============================="
    echo "llava eval"
    echo "==============================="
    CUDA_VISIBLE_DEVICES=$CUDA_DEVICE \
    python3 -u predict_llava.py \
        --read_path ${data} \
        --write_path "../results/results_llava_"${task}"_prompt"${prompt_id}"_param"${model_size}".json" \
        --task ${task} \
        --use_caption ${use_caption} \
        --image_folder ${image_folder} \
        --prompt_id ${prompt_id} \
        --model_size ${model_size}
    # echo "==============================="
    # echo "qwenvl eval"
    # echo "==============================="
    # CUDA_VISIBLE_DEVICES=$CUDA_DEVICE \
    # python3 -u predict_qwenvl.py \
    #     --read_path ${data} \
    #     --write_path "../results/results_qwenvl_"${task}"_prompt"${prompt_id}"_param"${model_size}".json" \
    #     --task ${task} \
    #     --use_caption ${use_caption} \
    #     --image_folder ${image_folder} \
    #     --prompt_id ${prompt_id} \
    #      --model_size ${model_size}
    echo "==============================="
    echo "instructblip eval"
    echo "==============================="
    CUDA_VISIBLE_DEVICES=$CUDA_DEVICE \
    python3 -u predict_instructblip.py \
        --read_path ${data} \
        --write_path "../results/results_instructblip_"${task}"_prompt"${prompt_id}"_param"${model_size}".json" \
        --task ${task} \
        --use_caption ${use_caption} \
        --image_folder ${image_folder} \
        --prompt_id ${prompt_id} \
         --model_size ${model_size}
    # echo "==============================="
    # echo "gpt4 eval"
    # echo "==============================="
    # python3 -u predict_gpt4.py \
    #     --read_path ${data} \
    #     --write_path "../results/results_gpt4vision_preview_"${task}"_prompt"${prompt_id}".json" \
    #     --task ${task} \
    #     --use_caption ${use_caption} \
    #     --image_folder ${image_folder} \
    #     --prompt_id ${prompt_id}
done