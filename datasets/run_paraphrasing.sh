pwd

eval "$(conda shell.bash hook)"
conda activate prototype_learning

echo "starting with 'IMDB'"

input="imdb_dataset/test_deepwordbug.csv"
output="imdb_dataset/adv_paraphrased.csv"

python paraphrase.py --input_file=$input --output_file=$output


echo "starting with 'DBPedia'"

input="dbpedia_dataset/test_deepwordbug.csv"
output="dbpedia_dataset/adv_paraphrased.csv"

python paraphrase.py --input_file=$input --output_file=$output


echo "starting with 'AG News'"

input="ag_news_dataset/test_deepwordbug.csv"
output="ag_news_dataset/adv_paraphrased.csv"

python paraphrase.py --input_file=$input --output_file=$output


echo "starting with 'SST2'"

input="sst2_dataset/test.csv"
output="sst2_dataset/adv_paraphrased.csv"

python paraphrase.py --input_file=$input --output_file=$output

echo "Finished"

conda deactivate