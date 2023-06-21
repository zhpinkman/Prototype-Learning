import os
import torch
from transformers import AutoTokenizer
import argparse
from IPython import embed
import utils
from models import ProtoTEx
from models_electra import ProtoTEx_Electra
import sys

sys.path.append("datasets")
import configs


def main(args):
    # preprocess the propaganda dataset loaded in the data folder. Original dataset can be found here
    # https://propaganda.math.unipd.it/fine-grained-propaganda-emnlp.html

    if args.architecture == "BART":
        tokenizer = AutoTokenizer.from_pretrained("ModelTC/bart-base-mnli")
        # tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-mnli")
    elif args.architecture == "ELECTRA":
        tokenizer = AutoTokenizer.from_pretrained("google/electra-base-discriminator")
    else:
        print(f"Invalid backbone architecture: {args.architecture}")

    all_datasets = utils.load_dataset(
        data_dir=args.data_dir,
        tokenizer=tokenizer,
        max_length=configs.dataset_to_max_length[args.dataset],
    )

    all_dataloaders = {
        dataset_name: torch.utils.data.DataLoader(
            all_datasets[dataset_name],
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=lambda batch: {
                "input_ids": torch.LongTensor([i["input_ids"] for i in batch]),
                "attention_mask": torch.Tensor([i["attention_mask"] for i in batch]),
                "label": torch.LongTensor([i["label"] for i in batch]),
            },
        )
        for dataset_name in all_datasets.keys()
    }

    if args.model == "ProtoTEx":
        print("ProtoTEx best model: {0}".format(args.num_prototypes))
        if args.architecture == "BART":
            print(f"Using backone: {args.architecture}")
            torch.cuda.empty_cache()
            model = ProtoTEx(
                num_prototypes=args.num_prototypes,
                class_weights=None,
                n_classes=configs.dataset_to_num_labels[args.dataset],
                max_length=configs.dataset_to_max_length[args.dataset],
                bias=False,
                dropout=False,
                special_classfn=True,  # special_classfn=False, # apply dropout only on bias
                p=1,  # p=0.75,
                batchnormlp1=True,
            )
        elif args.architecture == "ELECTRA":
            model = ProtoTEx_Electra(
                num_prototypes=args.num_prototypes,
                class_weights=None,
                n_classes=configs.dataset_to_num_labels[args.dataset],
                max_length=configs.dataset_to_max_length[args.dataset],
                bias=False,
                dropout=False,
                special_classfn=True,  # special_classfn=False, # apply dropout only on bias
                p=1,  # p=0.75,
                batchnormlp1=True,
            )

        else:
            print(f"Invalid backbone architecture: {args.architecture}")

        print(f"Loading model checkpoint: Models/{args.modelname}")
        pretrained_dict = torch.load(f"Models/{args.modelname}")
        # Fiter out unneccessary keys
        model_dict = model.state_dict()
        filtered_dict = {}
        for k, v in pretrained_dict.items():
            if k in model_dict and model_dict[k].shape == v.shape:
                filtered_dict[k] = v
            else:
                print(f"Skipping weights for: {k}")
        model_dict.update(filtered_dict)
        model.load_state_dict(model_dict)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # model = torch.nn.DataParallel(model)
        model = model.to(device)

        # utils.print_predictions(
        #     os.path.join("Logs", "test_predictions.csv"), y_pred, y_true
        # )

        for dataset_name, dataloader in all_dataloaders.items():
            if not (
                dataset_name.startswith("test_") or dataset_name.startswith("adv_")
            ):
                continue
            print(f"Evaluating on {dataset_name}")
            (
                total_loss,
                mac_prec,
                mac_recall,
                mac_f1_score,
                accuracy,
                y_true,
                y_pred,
            ) = utils.evaluate(dataloader, model_new=model)
            utils.print_logs(
                None,
                f"{dataset_name} TEST SCORES",
                0,
                total_loss,
                mac_prec,
                mac_recall,
                mac_f1_score,
                accuracy,
            )
            # utils.print_predictions(
            #     os.path.join("Logs", f"adv_predictions.csv"), y_pred, y_true
            # )
            embed()


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
