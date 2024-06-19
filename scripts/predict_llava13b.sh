data="annotated_data/data_annotation.json"
image_folder="YESBUT_cropped_yesbut"
write_path_surffix="contradiction.json"
#task options: contradiction | moral_mcq | title_mcq
use_caption=True

export CUDA_VISIBLE_DEVICES=0

task="contradiction"

echo "==============================="
echo "llava eval2"
echo "==============================="
CUDA_VISIBLE_DEVICES=0  python -u  predict_llava13b.py \
    --read_path ${data} \
    --write_path "results/results_llava13b_"${task}"_"${write_path_surffix} \
    --task ${task} \
    --use_caption ${use_caption} \
    --image_folder ${image_folder}\
    --gen_des True

task="moral_mcq"

echo "==============================="
echo "llava eval2"
echo "==============================="
CUDA_VISIBLE_DEVICES=0  python -u  predict_llava13b.py \
    --read_path ${data} \
    --write_path "results/results_llava13b_"${task}"_"${write_path_surffix} \
    --task ${task} \
    --use_caption ${use_caption} \
    --image_folder ${image_folder}\
    --gen_des True

task="title_mcq"

echo "==============================="
echo "llava eval2"
echo "==============================="
CUDA_VISIBLE_DEVICES=0  python -u  predict_llava13b.py \
    --read_path ${data} \
    --write_path "results/results_llava13b_"${task}"_"${write_path_surffix} \
    --task ${task} \
    --use_caption ${use_caption} \
    --image_folder ${image_folder}\
    --gen_des True