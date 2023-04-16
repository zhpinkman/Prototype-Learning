import torch
from transformers import AutoTokenizer

from args import args
import utils
from models import ProtoTEx, ProtoTEx_roberta, ProtoTEx_electra


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

    _, _, test_dataset = utils.load_dataset(tokenizer=tokenizer)
    adversarial_datasets = utils.load_adv_data(tokenizer=tokenizer)

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
                batchnormlp1=True,
                n_classes=utils.DatasetInfo().num_classes,
                max_length=utils.DatasetInfo().max_length,
            )

        elif args.architecture == "RoBERTa":
            print(f"Using backone: {args.architecture}")
            torch.cuda.empty_cache()
            model = ProtoTEx_roberta(
                num_prototypes=args.num_prototypes,
                bias=False,
                dropout=False,
                special_classfn=True,  # special_classfn=False, # apply dropout only on bias
                p=1,  # p=0.75,
                batchnormlp1=True,
                n_classes=utils.DatasetInfo().num_classes,
                max_length=utils.DatasetInfo().max_length,
            ).cuda()
        elif args.architecture == "Electra":
            print(f"Using backone: {args.architecture}")
            model = ProtoTEx_electra(
                num_prototypes=args.num_prototypes,
                bias=False,
                dropout=False,
                special_classfn=True,  # special_classfn=False, # apply dropout only on bias
                p=1,  # p=0.75,
                batchnormlp1=True,
                n_classes=utils.DatasetInfo().num_classes,
                max_length=utils.DatasetInfo().max_length,
            ).cuda()
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

        total_loss, mac_prec, mac_recall, mac_f1_score, accuracy = utils.evaluate(
            test_dl, model_new=model
        )
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

        for file_name, dataloader in adversarial_dataloaders.items():
            total_loss, mac_prec, mac_recall, mac_f1_score, accuracy = utils.evaluate(
                dataloader, model_new=model
            )
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


if __name__ == "__main__":
    main()
