dataset=$2
WANDB_MODE="offline" CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 python adv_attack_prototex.py \
    --batch_size 128 \
    --dataset $dataset \
    --data_dir "datasets/${dataset}_dataset" \
    --modelname "${dataset}_model" \
    --attack_type $1
