# python main.py \
#     --num_prototypes 50 \
#     --num_pos_prototypes 50 \
#     --data_dir "data/finegrained" \
#     --modelname "finegrained_nli_bart_prototex" \
#     --project "test-prototex" \
#     --experiment "test-prototex" \
#     --none_class "No" \
#     --augmentation "No" \
#     --nli_intialization "Yes" \
#     --curriculum "No" \
#     --architecture "BART"

# python main.py \
#     --num_prototypes 50 \
#     --num_pos_prototypes 50 \
#     --data_dir "data/SST-2" \
#     --modelname "finegrained_nli_bart_prototex_sst_2" \
#     --project "test-prototex" \
#     --experiment "test-prototex" \
#     --none_class "No" \
#     --augmentation "No" \
#     --nli_intialization "Yes" \
#     --curriculum "No" \
#     --architecture "BART"

python main.py \
    --num_prototypes 50 \
    --num_pos_prototypes 50 \
    --data_dir "data/CoLA" \
    --modelname "finegrained_nli_bart_prototex_CoLA" \
    --project "test-prototex" \
    --experiment "test-prototex" \
    --none_class "No" \
    --augmentation "No" \
    --nli_intialization "Yes" \
    --curriculum "No" \
    --architecture "BART"
