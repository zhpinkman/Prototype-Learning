python inference_and_explanations.py \
    --num_prototypes 50 \
    --num_pos_prototypes 50 \
    --data_dir "data/finegrained" \
    --modelname "finegrained_nli_bart_prototex" \
    --project "test-prototex" \
    --experiment "test-prototex" \
    --none_class "No" \
    --augmentation "No" \
    --nli_intialization "Yes" \
    --curriculum "No" \
    --architecture "BART"
