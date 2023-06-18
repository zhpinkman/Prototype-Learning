################################ Training ################################

dataset=$2
echo "Mode" $1

if [ "$1" = "train" ]; then

    for p1_lamb in 0.9; do
        for p2_lamb in 0.9; do
            for p3_lamb in 10.0 20.0; do

                WANDB_MODE="offline" CUDA_VISIBLE_DEVICES=$3 python main.py \
                    --batch_size $4 \
                    --dataset $dataset \
                    --data_dir "datasets/${dataset}_dataset" \
                    --p1_lamb $p1_lamb \
                    --p2_lamb $p2_lamb \
                    --p3_lamb $p3_lamb \
                    --modelname "${dataset}_model_${p1_lamb}_${p2_lamb}_${p3_lamb}"
            done
        done
    done

# elif [ "$1" = "inference" ]; then

#     WANDB_MODE="offline" CUDA_VISIBLE_DEVICES=6,7 python inference_and_explanations.py \
#         --num_prototypes 50 \
#         --data_dir "data/glue_data/sst2" \
#         --model_checkpoint "Models/finegrained_nli_bart_prototex_sst2"

################################ Testing ################################

elif [ "$1" = "test" ]; then

    p1_lamb=0.9
    p2_lamb=0.9
    p3_lamb=4.0
    WANDB_MODE="offline" CUDA_VISIBLE_DEVICES=2 python evaluate_model.py \
        --batch_size 128 \
        --dataset $dataset \
        --data_dir "datasets/${dataset}_dataset" \
        --modelname "${dataset}_model_${p1_lamb}_${p2_lamb}_${p3_lamb}"

else

    echo "Invalid mode"

fi
