import argparse
from models import ProtoTEx
import textattack
import argparse
from IPython import embed
import utils
from transformers import AutoTokenizer
import torch
import warnings
import sys

sys.path.append("datasets")
import configs

warnings.filterwarnings("ignore")


def main(args):
    tokenizer = AutoTokenizer.from_pretrained("ModelTC/bart-base-mnli")

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

    model_wrapper = textattack.models.wrappers.PyTorchModelWrapper(
        model=model, tokenizer=tokenizer
    )

    all_datasets = utils.load_dataset(
        data_dir=args.data_dir,
        tokenizer=tokenizer,
        max_length=configs.dataset_to_max_length[args.dataset],
    )

    embed()
    exit()

    # all_dataloaders = {
    #     dataset_name: torch.utils.data.DataLoader(
    #         all_datasets[dataset_name],
    #         batch_size=args.batch_size,
    #         shuffle=True,
    #         collate_fn=lambda batch: {
    #             "input_ids": torch.LongTensor([i["input_ids"] for i in batch]),
    #             "attention_mask": torch.Tensor([i["attention_mask"] for i in batch]),
    #             "label": torch.LongTensor([i["label"] for i in batch]),
    #         },
    #     )
    #     for dataset_name in all_datasets.keys()
    # }

    dataset = textattack.datasets.HuggingFaceDataset(all_datasets["test"])

    if args.attack_type == "textfooler":
        attack = textattack.attack_recipes.TextFoolerJin2019.build(model_wrapper)
    elif args.attack_type == "textbugger":
        attack = textattack.attack_recipes.TextBuggerLi2018.build(model_wrapper)

    print("Loaded attack and dataset")

    # Attack 20 samples with CSV logging and checkpoint saved every 5 interval

    log_file = f"log_{args.dataset}_{args.attack_type}_{args.modelname.replace('.', '').replace('/', '_')}.csv"
    summary_file = f"summary_{args.dataset}_{args.attack_type}_{args.modelname.replace('.', '').replace('/', '_')}.json"

    attack_args = textattack.AttackArgs(
        random_seed=1234,
        num_successful_examples=800,
        shuffle=True,
        log_to_csv=log_file,
        log_summary_to_json=summary_file,
        checkpoint_interval=None,
        checkpoint_dir="checkpoints",
        disable_stdout=True,
        parallel=False,
    )
    print("Created attack")
    attacker = textattack.Attacker(attack, dataset, attack_args)

    if args.mode == "attack":
        print("Attacking")
        attacker.attack_dataset()

    # if not os.path.exists(f"{args.dataset}_dataset"):
    #     os.makedirs(f"{args.dataset}_dataset")
    #     train_dataset = textattack.datasets.HuggingFaceDataset(
    #         args.dataset, split="train"
    #     )
    #     sentences = []
    #     labels = []
    #     for text, label in train_dataset:
    #         sentences.append(text["text"])
    #         labels.append(label)
    #     pd.DataFrame({"text": sentences, "label": labels}).to_csv(
    #         f"{args.dataset}_dataset/train.csv", index=False
    #     )
    # resulted_df = pd.read_csv(log_file)
    # resulted_df = resulted_df[resulted_df["result_type"] == "Successful"]
    # test_sentences = resulted_df["original_text"].tolist()
    # test_labels = resulted_df["ground_truth_output"].tolist()
    # adv_sentences = resulted_df["perturbed_text"].tolist()

    # pd.DataFrame(
    #     {
    #         "original_text": test_sentences,
    #         "perturbed_text": adv_sentences,
    #         "label": test_labels,
    #     }
    # ).to_csv(
    #     f"{args.dataset}_dataset/adv_{args.attack_type}_{args.model_checkpoint.replace('/', '_')}.csv",
    #     index=False,
    # )


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

    parser.add_argument(
        "--attack_type", type=str, default="textfooler", help="attack type"
    )

    args = parser.parse_args()
    main(args=args)
