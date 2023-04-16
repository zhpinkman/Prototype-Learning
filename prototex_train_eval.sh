################################ Training ################################

# TOKENIZERS_PARALLELISM=false WANDB_MODE="offline" CUDA_VISIBLE_DEVICES=2,3 python main.py \
#     --num_pos_prototypes 50 \
#     --num_prototypes 50 \
#     --batch_size 128 \
#     --data_dir "data/glue_data/sst2" \
#     --modelname "finegrained_nli_bart_prototex_sst2" \
#     --project "test-prototex" \
#     --experiment "test-prototex" \
#     --none_class "No" \
#     --augmentation "No" \
#     --nli_intialization "Yes" \
#     --curriculum "No" \
#     --architecture "BART" \
#     --use_max_length

################################ Testing ################################

TOKENIZERS_PARALLELISM=false WANDB_MODE="offline" CUDA_VISIBLE_DEVICES=6 python evaluate_model.py \
    --num_pos_prototypes 50 \
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

################################ Prototex ################################
