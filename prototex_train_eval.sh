################################ Training ################################

dataset=$2
echo "Mode" $1

if [ "$1" = "train" ]; then

    WANDB_MODE="offline" CUDA_VISIBLE_DEVICES=$3 python main.py \
        --batch_size $4 \
        --dataset $dataset \
        --data_dir "datasets/${dataset}_dataset" \
        --not_use_p1 \
        --modelname "${dataset}_model_not_use_p1"

# elif [ "$1" = "inference" ]; then

#     WANDB_MODE="offline" CUDA_VISIBLE_DEVICES=6,7 python inference_and_explanations.py \
#         --num_prototypes 50 \
#         --data_dir "data/glue_data/sst2" \
#         --model_checkpoint "Models/finegrained_nli_bart_prototex_sst2"

################################ Testing ################################

elif [ "$1" = "test" ]; then

    WANDB_MODE="offline" CUDA_VISIBLE_DEVICES=6 python evaluate_model.py \
        --batch_size 128 \
        --dataset $dataset \
        --data_dir "datasets/${dataset}_dataset" \
        --modelname "${dataset}_model"

else

    echo "Invalid mode"

fi
