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
        save_steps=128,
        logging_steps=128,
        eval_steps=128,
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

    embed()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode", type=str, choices=["train", "test", "eval"], default="train"
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

    args = parser.parse_args()
    main(args)
