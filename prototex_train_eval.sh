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
        --use_max_length

################################ Testing ################################

else

    TOKENIZERS_PARALLELISM=false WANDB_MODE="offline" CUDA_VISIBLE_DEVICES=5 python evaluate_model.py \
        --num_prototypes 50 \
        --batch_size 128 \
        --data_dir "data/glue_data/sst2" \
        --modelname "finegrained_nli_bart_prototex_sst2" \
        --model_checkpoint "Models/finegrained_nli_bart_prototex_sst2" \
        --project "test-prototex" \
        --experiment "test-prototex" \
        --none_class "No" \
        --augmentation "No" \
        --nli_intialization "Yes" \
        --curriculum "No" \
        --architecture "BART" \
        --use_max_length

fi
