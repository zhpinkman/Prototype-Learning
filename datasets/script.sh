for dataset in "imdb" "ag_news" "dbpedia"; do
    for attack_type in "textfooler" "textbugger"; do
        if [ "$dataset" = "ag_news" ]; then
            for model_checkpoint in "textattack/roberta-base-ag-news" "textattack/bert-base-uncased-ag-news" "andi611/distilbert-base-uncased-ner-agnews"; do
                echo " Attack type: " $attack_type
                echo " Dataset: " $dataset
                echo " Model checkpoint: " $model_checkpoint
                CUDA_VISIBLE_DEVICES=4,5,6,7 python adv_attack.py \
                    --dataset $dataset \
                    --attack_type $attack_type \
                    --model_checkpoint $model_checkpoint
                # --mode "attack"
            done
        elif [ "$dataset" = "imdb" ]; then
            for model_checkpoint in "textattack/bert-base-uncased-imdb" "textattack/albert-base-v2-imdb" "textattack/roberta-base-imdb"; do
                echo " Attack type: " $attack_type
                echo " Dataset: " $dataset
                echo " Model checkpoint: " $model_checkpoint
                CUDA_VISIBLE_DEVICES=0,1,2,3 python adv_attack.py \
                    --dataset $dataset \
                    --attack_type $attack_type \
                    --model_checkpoint $model_checkpoint
                # --mode "attack"
            done
        elif [ "$dataset" = "dbpedia" ]; then
            for model_checkpoint in "../normal_models/models/dbpedia_bert-base-uncased" "../normal_models/models/dbpedia_distilbert-base-uncased" "../normal_models/models/dbpedia_roberta-base"; do
                echo " Attack type: " $attack_type
                echo " Dataset: " $dataset
                echo " Model checkpoint: " $model_checkpoint
                CUDA_VISIBLE_DEVICES=4,5,6,7 python adv_attack.py \
                    --dataset $dataset \
                    --attack_type $attack_type \
                    --model_checkpoint $model_checkpoint
                # --mode "attack"
            done
        else
            echo "Invalid dataset"
            exit 1
        fi
    done
done
