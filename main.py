import os

# os.environ['TRANSFORMERS_CACHE'] = '/mnt/infonas/data/baekgupta/cache/'
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"]="2"
from importlib import reload
import numpy as np
import torch, time
from transformers import BartModel, BartConfig, BartForConditionalGeneration
from transformers import BartTokenizer
from tqdm.notebook import tqdm
import pathlib
from args import args, bad_classes, datasets_config
import pandas as pd
from sklearn.utils.class_weight import compute_class_weight
import random

## Custom modules
from preprocess import CustomNonBinaryClassDataset

from training import train_ProtoTEx_w_neg

## Set cuda
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def main():
    ## preprocess the propaganda dataset loaded in the data folder. Original dataset can be found here
    ## https://propaganda.math.unipd.it/fine-grained-propaganda-emnlp.html

    tokenizer = BartTokenizer.from_pretrained(
        "ynie/bart-large-snli_mnli_fever_anli_R1_R2_R3-nli"
    )

    train_df = pd.read_csv(os.path.join(args.data_dir, "train.csv"))
    dev_df = pd.read_csv(os.path.join(args.data_dir, "val.csv"))
    test_df = pd.read_csv(os.path.join(args.data_dir, "test.csv"))

    train_df = train_df[
        ~train_df[datasets_config[args.data_dir]["features"]["label"]].isin(bad_classes)
    ]
    dev_df = dev_df[
        ~dev_df[datasets_config[args.data_dir]["features"]["label"]].isin(bad_classes)
    ]
    test_df = test_df[
        ~test_df[datasets_config[args.data_dir]["features"]["label"]].isin(bad_classes)
    ]

    train_sentences = train_df[
        datasets_config[args.data_dir]["features"]["text"]
    ].tolist()
    dev_sentences = dev_df[datasets_config[args.data_dir]["features"]["text"]].tolist()
    test_sentences = test_df[
        datasets_config[args.data_dir]["features"]["text"]
    ].tolist()

    train_labels = train_df[
        datasets_config[args.data_dir]["features"]["label"]
    ].tolist()
    dev_labels = dev_df[datasets_config[args.data_dir]["features"]["label"]].tolist()
    test_labels = test_df[datasets_config[args.data_dir]["features"]["label"]].tolist()

    # Shuffle train dataset
    train_artifacts = list(zip(train_sentences, train_labels))
    random.seed(42)
    random.shuffle(train_artifacts)
    train_sentences, train_labels = zip(*train_artifacts)

    train_dataset = CustomNonBinaryClassDataset(
        sentences=train_sentences, labels=train_labels, tokenizer=tokenizer
    )
    dev_dataset = CustomNonBinaryClassDataset(
        sentences=dev_sentences, labels=dev_labels, tokenizer=tokenizer
    )
    test_dataset = CustomNonBinaryClassDataset(
        sentences=test_sentences, labels=test_labels, tokenizer=tokenizer
    )

    train_dl = torch.utils.data.DataLoader(
        train_dataset, batch_size=20, shuffle=True, collate_fn=train_dataset.collate_fn
    )
    val_dl = torch.utils.data.DataLoader(
        dev_dataset, batch_size=128, shuffle=False, collate_fn=dev_dataset.collate_fn
    )
    test_dl = torch.utils.data.DataLoader(
        test_dataset, batch_size=128, shuffle=False, collate_fn=test_dataset.collate_fn
    )
    # train_dl_eval=torch.utils.data.DataLoader(train_dataset_eval,batch_size=20,shuffle=False,
    #                                  collate_fn=train_dataset_eval.collate_fn)

    # Compute class weights
    class_weight_vect = compute_class_weight(
        "balanced", classes=np.unique(train_labels), y=train_labels
    )

    if args.model == "ProtoTEx":
        print(
            "ProtoTEx best model: {0}, {1}".format(
                args.num_prototypes, args.num_pos_prototypes
            )
        )
        train_ProtoTEx_w_neg(
            train_dl=train_dl,
            val_dl=val_dl,
            test_dl=test_dl,
            num_prototypes=args.num_prototypes,
            num_pos_prototypes=args.num_pos_prototypes,
            class_weights=class_weight_vect,
            modelname=args.modelname,
            model_checkpoint=args.model_checkpoint
        )


if __name__ == "__main__":
    main()
