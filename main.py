import torch
from transformers import AutoTokenizer
import wandb
import utils
import argparse

# Custom modules
from training import train_ProtoTEx_w_neg, train_simple_ProtoTEx

# Set cuda
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def main(args):
    # preprocess the propaganda dataset loaded in the data folder. Original dataset can be found here
    # https://propaganda.math.unipd.it/fine-grained-propaganda-emnlp.html

    if args.architecture == "BART":
        tokenizer = AutoTokenizer.from_pretrained("ModelTC/bart-base-mnli")
    elif args.architecture == "RoBERTa":
        tokenizer = AutoTokenizer.from_pretrained("cross-encoder/nli-roberta-base")
    elif args.architecture == "Electra":
        tokenizer = AutoTokenizer.from_pretrained("howey/electra-base-mnli")
    else:
        print(f"Invalid backbone architecture: {args.architecture}")

    # Load the dataset
    dataset_info = utils.DatasetInfo(
        data_dir=args.data_dir, use_max_length=args.use_max_length
    )
    train_dataset, val_dataset, test_dataset = utils.load_dataset(
        dataset_info=dataset_info, data_dir=args.data_dir, tokenizer=tokenizer
    )

    train_dl = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=train_dataset.collate_fn,
    )
    val_dl = torch.utils.data.DataLoader(
        val_dataset, batch_size=128, shuffle=False, collate_fn=val_dataset.collate_fn
    )
    test_dl = torch.utils.data.DataLoader(
        test_dataset, batch_size=128, shuffle=False, collate_fn=test_dataset.collate_fn
    )
    # train_dl_eval=torch.utils.data.DataLoader(train_dataset_eval,batch_size=20,shuffle=False,
    #                                  collate_fn=train_dataset_eval.collate_fn)

    # Compute class weights
    class_weight_vect = utils.get_class_weights(train_dataset.y)

    # Initialize wandb
    wandb.init(
        # Set the project where this run will be logged
        project=args.project,
        # We pass a run name (otherwise it’ll be randomly assigned, like sunshine-lollypop-10)
        name=args.experiment,
        # Track hyperparameters and run metadata
        config={
            "data_dir": args.data_dir,
            "modelname": args.modelname,
            "num_prototypes": args.num_prototypes,
            "none_class": args.none_class,
            "augmentation": args.augmentation,
            "nli_intialization": args.nli_intialization,
            "curriculum": args.curriculum,
            "architecture": args.architecture,
            "batchnormlp1": args.batchnormlp1,
            "model_checkpoint": args.model_checkpoint,
            "use_max_length": args.use_max_length,
            "tiny_sample": args.tiny_sample,
        },
    )

    if args.model == "ProtoTEx":
        print("ProtoTEx best model: {0}".format(args.num_prototypes))
        if args.architecture == "BART":
            print(f"Using backone: {args.architecture}")
            train_ProtoTEx_w_neg(
                train_dl=train_dl,
                val_dl=val_dl,
                test_dl=test_dl,
                n_classes=dataset_info.num_classes,
                max_length=dataset_info.max_length,
                num_prototypes=args.num_prototypes,
                batchnormlp1=args.batchnormlp1,
                class_weights=class_weight_vect,
                modelname=args.modelname,
                model_checkpoint=args.model_checkpoint,
                learning_rate=args.learning_rate,
            )
    elif args.model == "SimpleProtoTEx":
        train_simple_ProtoTEx(
            train_dl,
            val_dl,
            test_dl,
            train_dataset_len=len(train_dataset),
            modelname="SimpleProtoTEx",
            num_prototypes=args.num_prototypes,
        )
    else:
        print(f"Invalid backbone architecture: {args.architecture}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--tiny_sample", dest="tiny_sample", action="store_true")
    # parser.add_argument("--nli_dataset", help="check if the dataset is in nli
    # format that has sentence1, sentence2, label", action="store_true")
    parser.add_argument("--num_prototypes", type=int, default=50)
    parser.add_argument("--model", type=str, default="ProtoTEx")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--modelname", type=str)
    parser.add_argument("--data_dir", type=str)
    parser.add_argument("--model_checkpoint", type=str, default=None)
    parser.add_argument("--use_max_length", action="store_true")
    parser.add_argument("--batchnormlp1", action="store_true")
    parser.add_argument("--learning_rate", type=float, default="3e-5")

    # Wandb parameters
    parser.add_argument("--project", type=str)
    parser.add_argument("--experiment", type=str)
    parser.add_argument("--nli_intialization", type=str, default="Yes")
    parser.add_argument("--none_class", type=str, default="No")
    parser.add_argument("--curriculum", type=str, default="No")
    parser.add_argument("--augmentation", type=str, default="No")
    parser.add_argument("--architecture", type=str, default="BART")

    args = parser.parse_args()
    main(args)
