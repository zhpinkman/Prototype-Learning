import joblib
import numpy as np
from IPython import embed
import collections
import os
from sklearn.metrics import (
    precision_recall_fscore_support,
    accuracy_score,
    classification_report,
)
import datasets
from sklearn.model_selection import train_test_split
import torch.nn as nn
from sklearn.utils.class_weight import compute_class_weight
import torch
import pandas as pd
from tqdm import tqdm
from preprocess import CustomNonBinaryClassDataset
import json


class dce_loss(torch.nn.Module):
    def __init__(self, n_classes, feat_dim, init_weight=True):
        super(dce_loss, self).__init__()
        self.n_classes = n_classes
        self.feat_dim = feat_dim
        self.centers = nn.Parameter(
            torch.randn(self.feat_dim, self.n_classes).cuda(), requires_grad=True
        )
        if init_weight:
            self.__init_weight()

    def __init_weight(self):
        nn.init.kaiming_normal_(self.centers)

    def forward(self, x):
        features_square = torch.sum(torch.pow(x, 2), 1, keepdim=True)
        centers_square = torch.sum(torch.pow(self.centers, 2), 0, keepdim=True)
        features_into_centers = 2 * torch.matmul(x, (self.centers))
        dist = features_square + centers_square - features_into_centers

        return self.centers, -dist


def regularization(features, centers, labels):
    distance = features - torch.t(centers)[labels]

    distance = torch.sum(torch.pow(distance, 2), 1, keepdim=True)

    distance = (torch.sum(distance, 0, keepdim=True)) / features.shape[0]

    return distance


# Compute class weights
def get_class_weights(train_labels):
    class_weight_vect = compute_class_weight(
        "balanced", classes=np.unique(train_labels), y=train_labels
    )
    print(f"Class weight vectors: {class_weight_vect}")
    return class_weight_vect


def load_adv_data(dataset_info, data_dir, tokenizer):
    all_dataframes = []
    file_names = []
    for file in os.listdir(data_dir):
        if file.startswith("adv"):
            all_dataframes.append(pd.read_csv(os.path.join(data_dir, file)))
            file_names.append(file)
    return {
        file_name: load_classification_dataset(dataset_info, df, tokenizer)
        for file_name, df in zip(file_names, all_dataframes)
    }


def load_dataset(data_dir, tokenizer, max_length):
    train_df = pd.read_csv(os.path.join(data_dir, "train.csv"))

    if train_df.shape[0] > 80000:
        train_text = train_df["text"].tolist()
        train_labels = train_df["label"].tolist()
        train_text, _, train_labels, _ = train_test_split(
            train_text, train_labels, test_size=0.8, stratify=train_labels
        )
        train_df = pd.DataFrame({"text": train_text, "label": train_labels})

    print("Train data shape: ", train_df.shape)

    test_files = {
        file[: file.find(".")]: os.path.join(data_dir, file)
        for file in os.listdir(data_dir)
        if (file.startswith("test") or file.startswith("adv"))
    }

    test_dfs = {
        file_name: pd.read_csv(file_path) for file_name, file_path in test_files.items()
    }

    return {
        "train": load_classification_dataset(train_df, tokenizer, max_length),
        **{
            file_name: load_classification_dataset(df, tokenizer, max_length)
            for file_name, df in test_dfs.items()
        },
    }


# def load_nli_dataset(dataset_info, df, tokenizer):
#     sentences1 = df["sentence1"].tolist()
#     sentences2 = df["sentence2"].tolist()
#     labels = df["label"].tolist()

#     sentences = (sentences1, sentences2)

#     dataset = CustomNonBinaryClassDataset(
#         sentences=sentences,
#         labels=labels,
#         tokenizer=tokenizer,
#         max_length=dataset_info.max_length,
#     )

#     return dataset


def preprocess_data(tokenizer, dataset, max_length):
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )

    tokenized_dataset = dataset.map(
        tokenize_function, batched=True, remove_columns=["text"]
    )
    return tokenized_dataset


def load_classification_dataset(df, tokenizer, max_length):
    dataset = datasets.Dataset.from_pandas(df)
    tokenized_dataset = preprocess_data(tokenizer, dataset, max_length)

    return tokenized_dataset


def print_predictions(file, predictions, labels):
    df = pd.DataFrame(
        {"index": range(len(predictions)), "predictions": predictions, "labels": labels}
    )
    df.to_csv(file, index=False)


def print_logs(
    file, info, epoch, val_loss, mac_val_prec, mac_val_rec, mac_val_f1, accuracy
):
    logs = []
    s = " ".join((info + " epoch", str(epoch), "Total loss %.4f" % (val_loss), "\n"))
    logs.append(s)
    print(s)
    s = " ".join((info + " epoch", str(epoch), "Prec", str(mac_val_prec), "\n"))
    logs.append(s)
    print(s)
    s = " ".join((info + " epoch", str(epoch), "Recall", str(mac_val_rec), "\n"))
    logs.append(s)
    print(s)
    s = " ".join((info + " epoch", str(epoch), "F1", str(mac_val_f1), "\n"))
    logs.append(s)
    print(s)
    s = " ".join((info + " epoch", str(epoch), "Accuracy", str(accuracy), "\n"))
    logs.append(s)
    print(s)
    #     print("epoch",epoch,"MICRO val precision %.4f, recall %.4f, f1 %.4f,"%(mic_val_prec,mic_val_rec,mic_val_f1))
    print()
    logs.append("\n")
    if file is not None:
        f = open(file, "a")
        f.writelines(logs)
        f.close()


class EarlyStopping(object):
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(
        self,
        score_at_min1=0,
        patience=100,
        verbose=False,
        delta=0,
        path="checkpoint.pt",
        trace_func=print,
        save_epochwise=False,
    ):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = score_at_min1
        self.early_stop = False
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
        self.state_dict_list = [None] * patience
        self.improved = 0
        self.stop_update = 0
        self.save_model_counter = 0
        self.save_epochwise = save_epochwise
        self.times_improved = 0
        self.activated = False

    def activate(self, s1):
        if not self.activated and s1 > 0:
            self.activated = True

    def __call__(self, score, epoch, model):
        if not self.activated:
            return None
        self.save_model_counter = (self.save_model_counter + 1) % 4
        if not self.stop_update:
            if self.verbose:
                self.trace_func(
                    f"\033[91m The val score  of epoch {epoch} is {score:.4f} \033[0m"
                )
            if score < self.best_score + self.delta:
                self.counter += 1
                self.trace_func(
                    f"\033[93m EarlyStopping counter: {self.counter} out of {self.patience} \033[0m"
                )
                if self.counter >= self.patience:
                    self.early_stop = True
                self.improved = 0
            else:
                self.save_checkpoint(score, model, epoch)
                self.best_score = score
                self.counter = 0
                self.improved = 1
        else:
            self.improved = 0  # not needed though

    def save_checkpoint(self, score, model, epoch):
        """Saves model when validation loss decrease."""
        # if self.verbose:
        self.times_improved += 1
        self.trace_func(
            f"\033[92m Validation score improved ({self.best_score:.4f} --> {score:.4f}). \033[0m"
        )
        if self.save_epochwise:
            path = self.path + "_" + str(self.times_improved) + "_" + str(epoch)
        else:
            path = self.path
        torch.save(model.state_dict(), path)


def evaluate(dl, model_new=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assert model_new is not None
    loader = tqdm(dl, total=len(dl), unit="batches")
    total_len = 0
    model_new.eval()
    model_new = model_new.to(device)
    with torch.no_grad():
        total_loss = 0
        # tts = 0
        y_pred = []
        y_true = []
        for batch in loader:
            input_ids = batch["input_ids"]
            attn_mask = batch["attention_mask"]
            y = batch["label"]
            input_ids = input_ids.to(device)
            attn_mask = attn_mask.to(device)
            y = y.to(device)
            classfn_out, loss = model_new(
                input_ids, attn_mask, y, use_decoder=False, use_classfn=1
            )
            #             print(classfn_out.detach().cpu())
            if classfn_out.ndim == 1:
                predict = torch.zeros_like(y)
                predict[classfn_out > 0] = 1
            else:
                predict = torch.argmax(classfn_out, dim=1)

            y_pred.append(predict.cpu().numpy())
            #             y_pred.append(torch.zeros_like(y).numpy())
            y_true.append(y.cpu().numpy())
            total_loss += len(input_ids) * loss[0].item()
            total_len += len(input_ids)
        #             torch.cuda.empty_cache()
        total_loss = total_loss / total_len
        mac_prec, mac_recall, mac_f1_score, _ = precision_recall_fscore_support(
            np.concatenate(y_true), np.concatenate(y_pred), average="weighted"
        )
        accuracy = accuracy_score(np.concatenate(y_true), np.concatenate(y_pred))
        print(f"LABELS: {np.unique(np.concatenate(y_true))}")
        print(
            f"classification_report:\n{classification_report(np.concatenate(y_true),np.concatenate(y_pred), labels=np.unique(np.concatenate(y_true)), digits = 3)}"
        )

    return (
        total_loss,
        mac_prec,
        mac_recall,
        mac_f1_score,
        accuracy,
        np.concatenate(y_true),
        np.concatenate(y_pred),
    )


# Functions for analyzing prototypes


def get_best_k_protos_for_batch(
    dataset_info,
    dataset,
    model_new=None,
    model_path=None,
    model_class=None,
    topk=None,
    do_all=False,
):
    """
    get the best k protos for that a fraction of test data where each element has a specific true label.
    the "best" is in the sense that it has (or is one of those who has) the minimal distance
    from the encoded representation of the sentence.
    """
    assert (model_new is not None) ^ (model_path is not None)
    dl = torch.utils.data.DataLoader(
        dataset,
        batch_size=128,
        shuffle=False,
        collate_fn=lambda batch: {
            "input_ids": torch.LongTensor([i["input_ids"] for i in batch]),
            "attention_mask": torch.Tensor([i["attention_mask"] for i in batch]),
            "label": torch.LongTensor([i["label"] for i in batch]),
        },
    )
    loader = tqdm(dl, total=len(dl), unit="batches")
    model_new.eval()
    with torch.no_grad():
        # Updated for negative prototypes

        all_protos = model_new.prototypes

        best_protos = []
        best_protos_dists = []
        for batch in loader:
            input_ids = batch["input_ids"]
            attn_mask = batch["attention_mask"]
            y = batch["label"]
            batch_size = input_ids.size(0)
            last_hidden_state = model_new.bart_model.base_model.encoder(
                input_ids.cuda(),
                attn_mask.cuda(),
                output_attentions=False,
                output_hidden_states=False,
            ).last_hidden_state
            if not model_new.dobatchnorm:
                input_for_classfn = torch.cdist(
                    last_hidden_state.view(batch_size, -1),
                    all_protos.view(model_new.num_protos, -1),
                )
            else:
                input_for_classfn = torch.cdist(
                    last_hidden_state.view(batch_size, -1),
                    all_protos.view(model_new.num_protos, -1),
                )
                input_for_classfn = torch.nn.functional.instance_norm(
                    input_for_classfn.view(batch_size, 1, model_new.num_protos)
                ).view(batch_size, model_new.num_protos)

            if do_all:
                temp = torch.topk(input_for_classfn, dim=1, k=topk, largest=False)
            else:
                predicted = torch.argmax(
                    model_new.classfn_model(input_for_classfn).view(
                        batch_size, dataset_info.num_classes
                    ),
                    dim=1,
                )
                concerned_idxs = torch.nonzero((predicted == y.cuda())).view(-1)
                temp = torch.topk(
                    input_for_classfn[concerned_idxs], dim=1, k=topk, largest=False
                )
            best_protos.append(temp[1].cpu())
            best_protos_dists.append(temp[0].cpu())
        #             best_protos.append((torch.topk(input_for_classfn,dim=1,
        #                                               k=topk,largest=False)[1]).cpu())
        best_protos = torch.cat(best_protos, dim=0)
        best_protos_dists = torch.cat(best_protos_dists, dim=0)
    return best_protos, best_protos_dists


def get_bestk_train_data_for_every_proto(
    dataset_info, train_dataset_eval, model_new=None, top_k=3
):
    """
    for every prototype find out k best similar training examples
    """
    batch_size = 128
    dl = torch.utils.data.DataLoader(
        train_dataset_eval,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda batch: {
            "input_ids": torch.LongTensor([i["input_ids"] for i in batch]),
            "attention_mask": torch.Tensor([i["attention_mask"] for i in batch]),
            "label": torch.LongTensor([i["label"] for i in batch]),
        },
    )
    #     dl=torch.utils.data.DataLoader(test_dataset,batch_size=batch_size,shuffle=False,
    #                                      collate_fn=test_dataset.collate_fn)
    loader = tqdm(dl, total=len(dl), unit="batches")
    model_new.eval()
    #     model_new=model_new.cpu()
    with torch.no_grad():
        best_train_egs = []
        best_train_egs_values = []
        all_distances = torch.tensor([])
        predict_all = torch.tensor([])
        true_all = torch.tensor([])
        # Updated for negative prototypes

        all_protos = model_new.prototypes
        for batch_index, batch in enumerate(loader):
            input_ids = batch["input_ids"]
            attn_mask = batch["attention_mask"]
            y = batch["label"]
            batch_size = input_ids.size(0)
            last_hidden_state = model_new.bart_model.base_model.encoder(
                input_ids.cuda(),
                attn_mask.cuda(),
                output_attentions=False,
                output_hidden_states=False,
            ).last_hidden_state
            if not model_new.dobatchnorm:
                input_for_classfn = torch.cdist(
                    last_hidden_state.view(batch_size, -1),
                    all_protos.view(model_new.num_protos, -1),
                )
            else:
                input_for_classfn = torch.cdist(
                    last_hidden_state.view(batch_size, -1),
                    all_protos.view(model_new.num_protos, -1),
                )
                input_for_classfn = torch.nn.functional.instance_norm(
                    input_for_classfn.view(batch_size, 1, model_new.num_protos)
                ).view(batch_size, model_new.num_protos)
            predicted = torch.argmax(
                model_new.classfn_model(input_for_classfn).view(
                    batch_size, dataset_info.num_classes
                ),
                dim=1,
            )
            concerned_idxs = torch.nonzero((predicted == y.cuda())).view(-1)
            #             concerned_idxs=torch.nonzero((predicted==y)).view(-1)
            input_for_classfn = input_for_classfn[concerned_idxs]
            #             predict_all=torch.cat((predict_all,predicted.cpu()),dim=0)
            #             true_all=torch.cat((true_all,y.cpu()),dim=0)
            if top_k is None:
                all_distances = torch.cat(
                    (all_distances, input_for_classfn.cpu()), dim=0
                )
            else:
                best = torch.topk(input_for_classfn, dim=0, k=top_k, largest=False)
                best_train_egs.append(best[1] + batch_index * batch_size)
                best_train_egs_values.append(best[0])
    if top_k is None:
        return torch.cat(
            (true_all.view(-1, 1), predict_all.view(-1, 1), all_distances), dim=1
        )
    else:
        best_train_egs = torch.cat(best_train_egs, dim=0)
        best_train_egs_values = torch.cat(best_train_egs_values, dim=0)
        best_of_all_examples_for_each_prototype = torch.topk(
            best_train_egs_values, dim=0, k=top_k, largest=False
        )
        topk_idxs = best_of_all_examples_for_each_prototype[1]
        final_concerned_idxs = []
        for i in range(best_train_egs.size(1)):
            concerned_idxs = best_train_egs[topk_idxs[:, i], i]
            final_concerned_idxs.append(concerned_idxs)
        #         true_all=torch.cat(true_all,dim=0)
        #         predict_all=torch.cat(predict_all,dim=0)
        return (
            torch.stack(final_concerned_idxs, dim=0).cpu().numpy(),
            best_of_all_examples_for_each_prototype[0].cpu().numpy().T,
        )


def best_protos_for_test(test_dataset, model_new=None, top_k=5):
    batch_size = 60
    dl = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda batch: {
            "input_ids": torch.LongTensor([i["input_ids"] for i in batch]),
            "attention_mask": torch.Tensor([i["attention_mask"] for i in batch]),
            "label": torch.LongTensor([i["label"] for i in batch]),
        },
    )
    #     loader = tqdm(dl, total=len(dl), unit="batches")
    all_protos = model_new.prototypes
    batch = next(iter(dl))
    input_ids = batch["input_ids"]
    attn_mask = batch["attention_mask"]
    y = batch["label"]
    with torch.no_grad():
        last_hidden_state = model_new.bart_model.base_model.encoder(
            input_ids.cuda(),
            attn_mask.cuda(),
            output_attentions=False,
            output_hidden_states=False,
        ).last_hidden_state
        input_for_classfn = torch.cdist(
            last_hidden_state.view(batch_size, -1),
            all_protos.view(model_new.num_protos, -1),
        )
        predicted = torch.argmax(model_new.classfn_model(input_for_classfn), dim=1)
        proper_idxs_pos = (
            torch.nonzero(torch.logical_and(predicted == y, y == 1)).view(-1)
        )[:15]

        pos_best_protos = torch.topk(
            input_for_classfn[proper_idxs_pos], dim=1, k=top_k, largest=False
        )[1]

    return input_ids[proper_idxs_pos], pos_best_protos
