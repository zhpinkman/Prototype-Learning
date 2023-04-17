################################ Training ################################

echo "Mode" $1

if [ "$1" = "train" ]; then

    TOKENIZERS_PARALLELISM=false WANDB_MODE="offline" CUDA_VISIBLE_DEVICES=5 python main.py \
        --num_prototypes 50 \
        --batch_size 128 \
        --data_dir "data/glue_data/sst2" \
        --modelname "finegrained_nli_bart_prototex_sst2" \
        --project "test-prototex" \
        --experiment "test-prototex" \
        --none_class "No" \
        --augmentation "No" \
        --nli_intialization "Yes" \
        --curriculum "No" \
        --architecture "BART" \
        --use_max_length \
        --batchnormlp1

elif [ "$1" = "inference" ]; then

    TOKENIZERS_PARALLELISM=false WANDB_MODE="offline" CUDA_VISIBLE_DEVICES=5 python inference_and_explanations.py \
        --num_prototypes 50 \
        --data_dir "data/glue_data/sst2" \
        --model_checkpoint "Models/finegrained_nli_bart_prototex_sst2" \
        --none_class "No" \
        --augmentation "No" \
        --nli_intialization "Yes" \
        --curriculum "No" \
        --architecture "BART" \
        --use_max_length \
        --batchnormlp1

################################ Testing ################################

elif [ "$1" = "test" ]; then

    TOKENIZERS_PARALLELISM=false WANDB_MODE="offline" CUDA_VISIBLE_DEVICES=5 python evaluate_model.py \
        --num_prototypes 50 \
        --batch_size 128 \
        --data_dir "data/glue_data/sst2" \
        --model_checkpoint "Models/finegrained_nli_bart_prototex_sst2" \
        --none_class "No" \
        --augmentation "No" \
        --nli_intialization "Yes" \
        --curriculum "No" \
        --architecture "BART" \
        --use_max_length \
        --batchnormlp1

else

    echo "Invalid mode"

fi
