#!/bin/bash
#SBATCH --job-name=bigbench_nli_prototex
#SBATCH --output=slurm_execution/%x-%j.out
#SBATCH --error=slurm_execution/%x-%j.out
#SBATCH --time=3-00:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=10240
#SBATCH --partition=nodes
#SBATCH --gres=gpu:a100:1
#SBATCH --chdir=/cluster/raid/home/himanshu.rawlani/propaganda_detection/prototex_custom
# Verify working directory
echo $(pwd)
# Print gpu configuration for this job
nvidia-smi
# Verify gpu allocation (should be 1 GPU)
echo $CUDA_VISIBLE_DEVICES
# Initialize the shell to use local conda
eval "$(conda shell.bash hook)"
# Activate (local) env
conda activate prototex

echo "Starting training with parameters:"

dataset="data/bigbench"
echo "dataset: ${dataset}"
modelname="bigbench_nli_prototex"
echo "modelname: ${modelname}"
# model_checkpoint="Models/curr_lf_coarse_updated_aug_with_none_nli_prototex"


# for num_prototypes in 30 50 70
# do
# echo "number of prototypes: ${num_prototypes}"
# echo "number of positive prototypes: $((num_prototypes-1))"
# for i in {1..3}
# do
# echo "run number: $i"
# python main.py \
#     --num_prototypes ${num_prototypes} \
#     --num_pos_prototypes $((num_prototypes-1)) \
#     --data_dir ${dataset} \
#     --modelname ${modelname}
# done
# done

python main.py --num_prototypes 50 --num_pos_prototypes 49 --data_dir ${dataset} --modelname ${modelname} --project "direct-fine-tuning" --experiment "bigbench_classification_1" --none_class "Yes" --augmentation "No" --nli_intialization "Yes" --curriculum "No" --architecture "BART"

conda deactivate
