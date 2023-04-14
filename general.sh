################################ Training ################################

for conf in "qqp" "mnli" "mrpc" "qnli"; do

    WANDB_MODE="offline" CUDA_VISIBLE_DEVICES=2 python main.py \
        --num_pos_prototypes 50 \
        --num_prototypes 50 \
        --batch_size 128 \
        --data_dir "data/glue_data/$conf" \
        --modelname "finegrained_nli_bart_prototex_$conf" \
        --project "test-prototex" \
        --experiment "test-prototex" \
        --none_class "No" \
        --augmentation "No" \
        --nli_intialization "Yes" \
        --curriculum "No" \
        --architecture "BART" \
        --use_max_length

done

# for conf in "rte" "sst2" "wnli" "cola"; do

#     WANDB_MODE="offline" CUDA_VISIBLE_DEVICES=3 python main.py \
#         --num_pos_prototypes 50 \
#         --num_prototypes 50 \
#         --batch_size 128 \
#         --data_dir "data/glue_data/$conf" \
#         --modelname "finegrained_nli_bart_prototex_$conf" \
#         --project "test-prototex" \
#         --experiment "test-prototex" \
#         --none_class "No" \
#         --augmentation "No" \
#         --nli_intialization "Yes" \
#         --curriculum "No" \
#         --architecture "BART" \
#         --use_max_length

# done

################################ Testing ################################

# for conf in "mnli"; do

#     WANDB_MODE="offline" CUDA_VISIBLE_DEVICES=3 python evaluate_model.py \
#         --num_pos_prototypes 50 \
#         --num_prototypes 50 \
#         --batch_size 128 \
#         --data_dir "data/glue_data/$conf" \
#         --modelname "finegrained_nli_bart_prototex_$conf" \
#         --model_checkpoint "Models/finegrained_nli_bart_prototex_$conf" \
#         --project "test-prototex" \
#         --experiment "test-prototex" \
#         --none_class "No" \
#         --augmentation "No" \
#         --nli_intialization "Yes" \
#         --curriculum "No" \
#         --architecture "BART" \
#         --use_max_length

# done

# for conf in "rte" "sst2" "wnli" "cola"; do

#     WANDB_MODE="offline" CUDA_VISIBLE_DEVICES=3 python main.py \
#         --num_pos_prototypes 50 \
#         --num_prototypes 50 \
#         --batch_size 128 \
#         --data_dir "data/glue_data/$conf" \
#         --modelname "finegrained_nli_bart_prototex_$conf" \
#         --project "test-prototex" \
#         --experiment "test-prototex" \
#         --none_class "No" \
#         --augmentation "No" \
#         --nli_intialization "Yes" \
#         --curriculum "No" \
#         --architecture "BART" \
#         --use_max_length

# done
