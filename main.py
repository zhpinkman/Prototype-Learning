import torch
from transformers import AutoTokenizer
import wandb
import utils
import argparse
import sys

sys.path.append("datasets")
import configs

# Custom modules
from training import train_ProtoTEx_w_neg

# Set cuda
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def main(args):
    # preprocess the propaganda dataset loaded in the data folder. Original dataset can be found here
    # https://propaganda.math.unipd.it/fine-grained-propaganda-emnlp.html

    if args.architecture == "BART":
        tokenizer = AutoTokenizer.from_pretrained("ModelTC/bart-base-mnli")
        # tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-mnli")
    elif args.architecture == "RoBERTa":
        tokenizer = AutoTokenizer.from_pretrained("cross-encoder/nli-roberta-base")
    elif args.architecture == "ELECTRA":
        tokenizer = AutoTokenizer.from_pretrained("google/electra-base-discriminator")
    else:
        print(f"Invalid backbone architecture: {args.architecture}")

    # Load the dataset
    all_datasets = utils.load_dataset(
        data_dir=args.data_dir,
        tokenizer=tokenizer,
        max_length=configs.dataset_to_max_length[args.dataset],
    )

    train_dl = torch.utils.data.DataLoader(
        all_datasets["train"],
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda batch: {
            "input_ids": torch.LongTensor([i["input_ids"] for i in batch]),
            "attention_mask": torch.Tensor([i["attention_mask"] for i in batch]),
            "label": torch.LongTensor([i["label"] for i in batch]),
        },
    )

    test_dl = torch.utils.data.DataLoader(
        all_datasets["test_paraphrased"],
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda batch: {
            "input_ids": torch.LongTensor([i["input_ids"] for i in batch]),
            "attention_mask": torch.Tensor([i["attention_mask"] for i in batch]),
            "label": torch.LongTensor([i["label"] for i in batch]),
        },
    )
    # train_dl_eval=torch.utils.data.DataLoader(train_dataset_eval,batch_size=20,shuffle=False,
    #                                  collate_fn=train_dataset_eval.collate_fn)

    # Compute class weights

    class_weight_vect = utils.get_class_weights(all_datasets["train"]["label"])

    # Initialize wandb
    # wandb.init(
    #     # Set the project where this run will be logged
    #     project=args.project,
    #     # We pass a run name (otherwise it’ll be randomly assigned, like sunshine-lollypop-10)
    #     name=args.experiment,
    #     # Track hyperparameters and run metadata
    #     config={
    #         "data_dir": args.data_dir,
    #         "modelname": args.modelname,
    #         "num_prototypes": args.num_prototypes,
    #         "none_class": args.none_class,
    #         "augmentation": args.augmentation,
    #         "nli_intialization": args.nli_intialization,
    #         "curriculum": args.curriculum,
    #         "architecture": args.architecture,
    #         "model_checkpoint": args.model_checkpoint,
    #         "use_max_length": args.use_max_length,
    #         "tiny_sample": args.tiny_sample,
    #     },
    # )

    if args.model == "ProtoTEx":
        print("ProtoTEx best model: {0}".format(args.num_prototypes))

        print(f"Using backone: {args.architecture}")
        train_ProtoTEx_w_neg(
            architecture=args.architecture,
            train_dl=train_dl,
            val_dl=test_dl,
            test_dl=test_dl,
            n_classes=configs.dataset_to_num_labels[args.dataset],
            max_length=configs.dataset_to_max_length[args.dataset],
            num_prototypes=args.num_prototypes,
            class_weights=class_weight_vect,
            modelname=args.modelname,
            learning_rate=args.learning_rate,
            p1_lamb=args.p1_lamb,
            p2_lamb=args.p2_lamb,
            p3_lamb=args.p3_lamb,
        )

    else:
        print(f"Invalid backbone architecture: {args.architecture}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--tiny_sample", dest="tiny_sample", action="store_true")
    # parser.add_argument("--nli_dataset", help="check if the dataset is in nli
    # format that has sentence1, sentence2, label", action="store_true")
    parser.add_argument("--num_prototypes", type=int, default=16)
    parser.add_argument("--model", type=str, default="ProtoTEx")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--modelname", type=str)
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--data_dir", type=str)
    parser.add_argument("--learning_rate", type=float, default="1e-4")

    # Wandb parameters
    parser.add_argument("--project", type=str)
    parser.add_argument("--experiment", type=str)
    parser.add_argument("--nli_intialization", type=str, default="Yes")
    parser.add_argument("--none_class", type=str, default="No")
    parser.add_argument("--curriculum", type=str, default="No")
    parser.add_argument("--augmentation", type=str, default="No")
    parser.add_argument("--architecture", type=str, default="BART")

    parser.add_argument("--p1_lamb", type=float, default=0.9)
    parser.add_argument("--p2_lamb", type=float, default=0.9)
    parser.add_argument("--p3_lamb", type=float, default=0.9)

    args = parser.parse_args()
    main(args)
