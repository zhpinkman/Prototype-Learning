import os
import torch
from transformers import AutoTokenizer
import argparse
from IPython import embed
import utils
from models import ProtoTEx


def main(args):
    # preprocess the propaganda dataset loaded in the data folder. Original dataset can be found here
    # https://propaganda.math.unipd.it/fine-grained-propaganda-emnlp.html

    if args.architecture == "BART":
        tokenizer = AutoTokenizer.from_pretrained("ModelTC/bart-base-mnli")
    else:
        print(f"Invalid backbone architecture: {args.architecture}")

    dataset_info = utils.DatasetInfo(
        data_dir=args.data_dir, use_max_length=args.use_max_length
    )
    _, _, test_dataset = utils.load_dataset(
        dataset_info=dataset_info, data_dir=args.data_dir, tokenizer=tokenizer
    )
    adversarial_datasets = utils.load_adv_data(
        dataset_info=dataset_info, data_dir=args.data_dir, tokenizer=tokenizer
    )

    adversarial_dataloaders = {
        file_name: torch.utils.data.DataLoader(
            dataset, batch_size=128, shuffle=False, collate_fn=dataset.collate_fn
        )
        for file_name, dataset in adversarial_datasets.items()
    }

    test_dl = torch.utils.data.DataLoader(
        test_dataset, batch_size=128, shuffle=False, collate_fn=test_dataset.collate_fn
    )

    if args.model == "ProtoTEx":
        print("ProtoTEx best model: {0}".format(args.num_prototypes))
        if args.architecture == "BART":
            print(f"Using backone: {args.architecture}")
            torch.cuda.empty_cache()
            model = ProtoTEx(
                num_prototypes=args.num_prototypes,
                bias=False,
                dropout=False,
                special_classfn=True,  # special_classfn=False, # apply dropout only on bias
                p=1,  # p=0.75,
                batchnormlp1=args.batchnormlp1,
                n_classes=dataset_info.num_classes,
                max_length=dataset_info.max_length,
            )

        else:
            print(f"Invalid backbone architecture: {args.architecture}")

        print(f"Loading model checkpoint: {args.model_checkpoint}")
        pretrained_dict = torch.load(args.model_checkpoint)
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

        (
            total_loss,
            mac_prec,
            mac_recall,
            mac_f1_score,
            accuracy,
            y_true,
            y_pred,
        ) = utils.evaluate(test_dl, model_new=model)
        utils.print_logs(
            None,
            "TEST SCORES",
            0,
            total_loss,
            mac_prec,
            mac_recall,
            mac_f1_score,
            accuracy,
        )

        utils.print_predictions(
            os.path.join("Logs", "test_predictions.csv"), y_pred, y_true
        )

        for file_name, dataloader in adversarial_dataloaders.items():
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
                f"{file_name} TEST SCORES",
                0,
                total_loss,
                mac_prec,
                mac_recall,
                mac_f1_score,
                accuracy,
            )
            utils.print_predictions(
                os.path.join("Logs", f"adv_predictions.csv"), y_pred, y_true
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--tiny_sample", dest="tiny_sample", action="store_true")
    # parser.add_argument("--nli_dataset", help="check if the dataset is in nli
    # format that has sentence1, sentence2, label", action="store_true")
    parser.add_argument("--num_prototypes", type=int, default=50)
    parser.add_argument("--model", type=str, default="ProtoTEx")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--data_dir", type=str)
    parser.add_argument("--model_checkpoint", type=str, default=None)
    parser.add_argument("--use_max_length", action="store_true")
    parser.add_argument("--batchnormlp1", action="store_true")

    # Wandb parameters
    parser.add_argument("--experiment", type=str)
    parser.add_argument("--nli_intialization", type=str, default="Yes")
    parser.add_argument("--none_class", type=str, default="No")
    parser.add_argument("--curriculum", type=str, default="No")
    parser.add_argument("--augmentation", type=str, default="No")
    parser.add_argument("--architecture", type=str, default="BART")

    args = parser.parse_args()

    main(args)
