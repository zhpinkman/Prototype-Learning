# for conf in "mnli" "mrpc" "qnli" "qqp" "rte" "sst2" "wnli" "cola"; do
for conf in "rte"; do

    WANDB_MODE="offline" CUDA_VISIBLE_DEVICES=2,3,4 python main.py \
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
