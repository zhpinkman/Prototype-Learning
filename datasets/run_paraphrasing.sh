pwd

eval "$(conda shell.bash hook)"
conda activate prototype_learning

echo "starting with 'IMDB'"

input="imdb_dataset/test.csv"
output="imdb_dataset/adv_paraphrased_1.csv"

python paraphrase.py --input_file=$input --output_file=$output --start_from=400


echo "starting with 'DBPedia'"

input="dbpedia_dataset/test.csv"
output="dbpedia_dataset/adv_paraphrased_1.csv"

python paraphrase.py --input_file=$input --output_file=$output --start_from=400


echo "starting with 'AG News'"

input="ag_news_dataset/test.csv"
output="ag_news_dataset/adv_paraphrased_1.csv"

python paraphrase.py --input_file=$input --output_file=$output --start_from=400


echo "starting with 'SST2'"

input="sst2_dataset/test.csv"
output="sst2_dataset/adv_paraphrased_1.csv"

python paraphrase.py --input_file=$input --output_file=$output --start_from=400

echo "Finished"

conda deactivate