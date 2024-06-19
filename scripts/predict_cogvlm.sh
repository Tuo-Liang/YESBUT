data="annotated_data/data_annotation.json"
image_folder="YESBUT_cropped_yesbut"
write_path_surffix="contradiction.json"
#task options: contradiction | moral_mcq | title_mcq
task="moral_mcq"
use_caption=true

CUDA_DEVICE=0

echo "==============================="
echo "cogvlm eval"
echo "==============================="
CUDA_VISIBLE_DEVICES=$CUDA_DEVICE \
python3 -u predict_cogvlm.py \
    --read_path ${data} \
    --write_path "results/results_cogvlm_"${task}"_"${write_path_surffix} \
    --task ${task} \
    --use_caption ${use_caption} \
    --image_folder ${image_folder}



