import torch
from transformers import AutoTokenizer
from args import args
import wandb
import utils

# Custom modules
from training import (
    train_ProtoTEx_w_neg,
    train_ProtoTEx_w_neg_roberta,
    train_ProtoTEx_w_neg_electra,
)

# Set cuda
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def main():
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
    train_dataset, val_dataset, test_dataset = utils.load_dataset(tokenizer=tokenizer)

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
                n_classes=utils.DatasetInfo().num_classes,
                max_length=utils.DatasetInfo().max_length,
                num_prototypes=args.num_prototypes,
                class_weights=class_weight_vect,
                modelname=args.modelname,
                model_checkpoint=args.model_checkpoint,
            )
        elif args.architecture == "RoBERTa":
            print(f"Using backone: {args.architecture}")
            train_ProtoTEx_w_neg_roberta(
                train_dl=train_dl,
                val_dl=val_dl,
                test_dl=test_dl,
                n_classes=utils.DatasetInfo().num_classes,
                max_length=utils.DatasetInfo().max_length,
                num_prototypes=args.num_prototypes,
                class_weights=class_weight_vect,
                modelname=args.modelname,
                model_checkpoint=args.model_checkpoint,
            )
        elif args.architecture == "Electra":
            print(f"Using backone: {args.architecture}")
            train_ProtoTEx_w_neg_electra(
                train_dl=train_dl,
                val_dl=val_dl,
                test_dl=test_dl,
                n_classes=utils.DatasetInfo().num_classes,
                max_length=utils.DatasetInfo().max_length,
                num_prototypes=args.num_prototypes,
                class_weights=class_weight_vect,
                modelname=args.modelname,
                model_checkpoint=args.model_checkpoint,
            )
        else:
            print(f"Invalid backbone architecture: {args.architecture}")


if __name__ == "__main__":
    main()
