import pandas as pd
import os
import json
from IPython import embed

for dataset in ["imdb", "ag_news", "dbpedia"]:
    dataset_dir = f"{dataset}_dataset"
    print("Dataset:", dataset_dir)

    for attack in ["textfooler", "textbugger"]:
        print("Attack:", attack)
        files = [
            os.path.join(dataset_dir, file)
            for file in os.listdir(dataset_dir)
            if file.startswith(f"adv_{attack}_")
        ]
        print("Number of files:", len(files))
        print(files)
        print("------------------")
        dfs = [pd.read_csv(file) for file in files]
        all_original_texts = [df["original_text"].values for df in dfs]
        all_perturbed_texts = [df["perturbed_text"].values for df in dfs]
        all_labels = [df["label"].values for df in dfs]

        all_original_texts_clean = [
            list(map(lambda x: x.replace("[", "").replace("]", ""), texts))
            for texts in all_original_texts
        ]
        all_perturbed_texts_clean = [
            list(map(lambda x: x.replace("[", "").replace("]", ""), texts))
            for texts in all_perturbed_texts
        ]

        common_original_texts = set.intersection(
            *[set(all_original_texts_clean[i]) for i in range(len(files))]
        )

        print(len(common_original_texts))

        results_original_texts = []
        results_perturbed_texts = []
        results_labels = []

        for file_index in range(len(files)):
            for original_text, perturbed_text, label in zip(
                all_original_texts_clean[file_index],
                all_perturbed_texts_clean[file_index],
                all_labels[file_index],
            ):
                if original_text in common_original_texts:
                    results_original_texts.append(original_text)
                    results_perturbed_texts.append(perturbed_text)
                    results_labels.append(label)

        pd.DataFrame(
            {
                "text": results_perturbed_texts,
                "label": results_labels,
            }
        ).to_csv(os.path.join(dataset_dir, f"adv_{attack}.csv"), index=False)
        pd.DataFrame(
            {
                "text": results_original_texts,
                "label": results_labels,
            }
        ).to_csv(os.path.join(dataset_dir, f"test_{attack}.csv"), index=False)

        # remove the file in files
        for file in files:
            os.remove(file)
