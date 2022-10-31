#!/bin/bash
#SBATCH --job-name=general
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --time=3-00:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=10240
#SBATCH --partition=nodes
#SBATCH --gres=gpu:a100:1
#SBATCH --chdir=/cluster/raid/home/zhivar.sourati/ProtoTEx
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

dataset="data/logical_fallacy"
echo "dataset: ${dataset}"


for num_prototypes in 30 50 70
do
echo "number of prototypes: ${num_prototypes}"
for i in {1..5}
do
echo "run number: $i"
python main.py \
    --num_prototypes ${num_prototypes} \
    --num_pos_prototypes ${num_prototypes} \
    --data_dir ${dataset}
done
done

conda deactivate