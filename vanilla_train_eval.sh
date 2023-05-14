################################ Training ################################

echo "Mode" $1

if [ "$1" = "train" ]; then

    TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=2,3 python train_vanilla_model.py \
        --mode train \
        --batch_size 512 \
        --logging_steps 50 \
        --data_dir "data/glue_data/sst2/"

################################ Testing ################################

else

    TOKENIZERS_PARALLELISM=false WANDB_MODE="offline" CUDA_VISIBLE_DEVICES=2 python train_vanilla_model.py \
        --mode eval \
        --batch_size 512 \
        --logging_steps 50 \
        --data_dir "data/glue_data/sst2/"

    TOKENIZERS_PARALLELISM=false WANDB_MODE="offline" CUDA_VISIBLE_DEVICES=2 python train_vanilla_model.py \
        --mode eval_adv \
        --batch_size 512 \
        --logging_steps 50 \
        --data_dir "data/glue_data/sst2/"
fi
