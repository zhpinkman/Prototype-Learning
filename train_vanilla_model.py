from IPython import embed
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from datasets import Dataset, DatasetDict
import pandas as pd
import argparse
import os
import evaluate
import numpy as np


def preprocess_data(tokenizer, dataset, args):
    def tokenize_function(examples):
        return tokenizer(
            examples["sentence"],
            padding="max_length",
            truncation=True,
            max_length=args.max_length,
        )

    tokenized_dataset = dataset.map(
        tokenize_function, batched=True, remove_columns=["sentence", "idx"]
    )
    return tokenized_dataset


def load_data(data_dir):
    train_df = pd.read_csv(os.path.join(data_dir, "train.csv"))
    dev_df = pd.read_csv(os.path.join(data_dir, "val.csv"))
    test_df = pd.read_csv(os.path.join(data_dir, "test.csv"))

    dataset = DatasetDict(
        {
            "train": Dataset.from_pandas(train_df),
            "validation": Dataset.from_pandas(dev_df),
            "test": Dataset.from_pandas(test_df),
        }
    )
    return dataset


def compute_metrics(eval_preds):
    clf_metrics = evaluate.combine(["accuracy", "f1", "precision", "recall"])
    predictions, labels = eval_preds
    predictions = np.argmax(predictions[0], axis=1)
    return clf_metrics.compute(predictions=predictions, references=labels)


def load_adv_data(data_dir):
    all_dataframes = []
    file_names = []
    for file in os.listdir(data_dir):
        if file.startswith("adv"):
            all_dataframes.append(pd.read_csv(os.path.join(data_dir, file)))
            file_names.append(file)
    dataset = DatasetDict(
        {file: Dataset.from_pandas(df) for file, df in zip(file_names, all_dataframes)}
    )
    return dataset


def main(args):
    if args.mode == "train":
        tokenizer = AutoTokenizer.from_pretrained(args.model_checkpoint)
        model = AutoModelForSequenceClassification.from_pretrained(
            args.model_checkpoint, num_labels=2, ignore_mismatched_sizes=True
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
        model = AutoModelForSequenceClassification.from_pretrained(
            args.model_dir, num_labels=2, ignore_mismatched_sizes=True
        )

    dataset = load_data(args.data_dir)
    tokenized_dataset = preprocess_data(tokenizer, dataset, args)

    training_args = TrainingArguments(
        output_dir=args.model_dir,
        num_train_epochs=3,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        weight_decay=0.01,
        logging_dir=args.log_dir,
        report_to="wandb",
        evaluation_strategy="steps",
        save_strategy="steps",
        save_total_limit=2,
        logging_strategy="steps",
        save_steps=args.logging_steps,
        logging_steps=args.logging_steps,
        eval_steps=args.logging_steps,
        load_best_model_at_end=True,
    )

    os.environ["WANDB_PROJECT"] = "prototex_project_vanilla_bart_model"

    trainer = Trainer(
        model,
        training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    if args.mode == "train":
        trainer.evaluate(tokenized_dataset["validation"])
        trainer.train()
        trainer.save_model(args.model_dir)
    elif args.mode == "eval":
        eval_metrics = trainer.evaluate(tokenized_dataset["validation"])
        test_metrics = trainer.evaluate(tokenized_dataset["test"])
        print("Validation metrics: ", eval_metrics)
        print("Test metrics: ", test_metrics)

    elif args.mode == "test":
        predictions = trainer.predict(tokenized_dataset["test"]).predictions[0]
        predicted_labels = np.argmax(predictions, axis=1)
        for label in predicted_labels:
            print(label)

    elif args.mode == "eval_adv":
        adv_dataset = load_adv_data(args.data_dir)
        tokenized_adv_dataset = preprocess_data(tokenizer, adv_dataset, args)
        for file in tokenized_adv_dataset:
            print("results for Adversarial file: ", file)
            metrics = trainer.evaluate(tokenized_adv_dataset[file])
            print(metrics)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "test", "eval", "eval_adv"],
        default="train",
    )

    parser.add_argument(
        "--data_dir",
        type=str,
    )
    parser.add_argument(
        "--model_dir", type=str, default="Models/base_model_bart_wo_prototex"
    )
    parser.add_argument(
        "--model_checkpoint", type=str, default="ModelTC/bart-base-mnli"
    )
    parser.add_argument("--batch_size", type=int, default=128)

    parser.add_argument("--log_dir", type=str, default="./logs")

    parser.add_argument("--max_length", type=int, default=40)
    
    parser.add_argument("--logging_steps", type = int)

    args = parser.parse_args()
    main(args)
