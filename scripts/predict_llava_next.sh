data="annotated_data/data_annotation.json"
image_folder="YESBUT_cropped_yesbut"
write_path_surffix="contradiction.json"
#task options: contradiction | moral_mcq | title_mcq
use_caption=true

export CUDA_VISIBLE_DEVICES=0



task="moral_mcq"

echo "==============================="
echo "llava eval"
echo "==============================="
CUDA_VISIBLE_DEVICES=0  python -u  predict_llava_next.py \
    --read_path ${data} \
    --write_path "results/results_llava_next_"${task}"_prompt0"${write_path_surffix} \
    --task ${task} \
    --use_caption ${use_caption} \
    --image_folder ${image_folder}