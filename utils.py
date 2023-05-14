import joblib
import numpy as np
import collections
import os
from sklearn.metrics import (
    precision_recall_fscore_support,
    accuracy_score,
    classification_report,
)
import torch.nn as nn
from sklearn.utils.class_weight import compute_class_weight
import torch
import pandas as pd
from tqdm import tqdm
from preprocess import CustomNonBinaryClassDataset
from args import args
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


class DatasetInfo:
    def __new__(cls):
        if not hasattr(cls, "instance"):
            cls.instance = super(DatasetInfo, cls).__new__(cls)
        return cls.instance

    @property
    def data(self):
        return self.dataset_info

    def __init__(self) -> None:
        self.load_dataset_info()

    def load_dataset_info(self):
        with open(os.path.join(args.data_dir, "info.json"), "r") as f:
            self.dataset_info = json.load(f)

    @property
    def batch_size(self):
        return DatasetInfo().data["batch_size"]

    @property
    def dataset_type(self):
        return DatasetInfo().data["dataset_type"]

    @property
    def dataset_classes_dict(self):
        classes = self.dataset_info["features"]["label"]["names"]
        return dict(zip(classes, range(len(classes))))

    @property
    def max_length(self):
        return (
            self.dataset_info["sentence_max_length"]
            if ("sentence_max_length" in self.dataset_info and args.use_max_length)
            else 128
        )

    @property
    def num_classes(self):
        return len(self.dataset_info["features"]["label"]["names"])


# Compute class weights
def get_class_weights(train_labels):
    class_weight_vect = compute_class_weight(
        "balanced", classes=np.unique(train_labels), y=train_labels
    )
    print(f"Class weight vectors: {class_weight_vect}")
    return class_weight_vect


def load_dataset(tokenizer):
    train_df = pd.read_csv(os.path.join(args.data_dir, "train.csv"))
    val_df = pd.read_csv(os.path.join(args.data_dir, "val.csv"))
    test_df = pd.read_csv(os.path.join(args.data_dir, "test.csv"))

    # shuffle train dataframe
    train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)

    if DatasetInfo().dataset_type == "classification":
        return load_classification_dataset(train_df, val_df, test_df, tokenizer)
    elif DatasetInfo().dataset_type == "nli":
        return load_nli_dataset(train_df, val_df, test_df, tokenizer)

    else:
        raise Exception("Dataset type not supported")


def load_nli_dataset(train_df, val_df, test_df, tokenizer):
    train_sentences1 = train_df["sentence1"].tolist()
    train_sentences2 = train_df["sentence2"].tolist()
    train_labels = train_df["label"].tolist()

    val_sentences1 = val_df["sentence1"].tolist()
    val_sentences2 = val_df["sentence2"].tolist()
    val_labels = val_df["label"].tolist()

    test_sentences1 = test_df["sentence1"].tolist()
    test_sentences2 = test_df["sentence2"].tolist()
    test_labels = test_df["label"].tolist()

    train_sentences = (train_sentences1, train_sentences2)
    val_sentences = (val_sentences1, val_sentences2)
    test_sentences = (test_sentences1, test_sentences2)

    train_dataset = CustomNonBinaryClassDataset(
        sentences=train_sentences,
        labels=train_labels,
        tokenizer=tokenizer,
        max_length=DatasetInfo().max_length,
        dataset_type=DatasetInfo().dataset_type,
    )
    val_dataset = CustomNonBinaryClassDataset(
        sentences=val_sentences,
        labels=val_labels,
        tokenizer=tokenizer,
        max_length=DatasetInfo().max_length,
        dataset_type=DatasetInfo().dataset_type,
    )
    test_dataset = CustomNonBinaryClassDataset(
        sentences=test_sentences,
        labels=test_labels,
        tokenizer=tokenizer,
        max_length=DatasetInfo().max_length,
        dataset_type=DatasetInfo().dataset_type,
    )
    return train_dataset, val_dataset, test_dataset


def load_classification_dataset(train_df, val_df, test_df, tokenizer):
    train_sentences = train_df["sentence"].tolist()
    train_labels = train_df["label"].tolist()

    val_sentences = val_df["sentence"].tolist()
    val_labels = val_df["label"].tolist()

    test_sentences = test_df["sentence"].tolist()
    test_labels = test_df["label"].tolist()

    train_dataset = CustomNonBinaryClassDataset(
        sentences=train_sentences,
        labels=train_labels,
        tokenizer=tokenizer,
        max_length=DatasetInfo().max_length,
        dataset_type=DatasetInfo().dataset_type,
    )
    val_dataset = CustomNonBinaryClassDataset(
        sentences=val_sentences,
        labels=val_labels,
        tokenizer=tokenizer,
        max_length=DatasetInfo().max_length,
        dataset_type=DatasetInfo().dataset_type,
    )
    test_dataset = CustomNonBinaryClassDataset(
        sentences=test_sentences,
        labels=test_labels,
        tokenizer=tokenizer,
        max_length=DatasetInfo().max_length,
        dataset_type=DatasetInfo().dataset_type,
    )

    return train_dataset, val_dataset, test_dataset


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
            input_ids, attn_mask, y = batch
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
            f"classification_report:\n{classification_report(np.concatenate(y_true),np.concatenate(y_pred), labels=np.unique(np.concatenate(y_true)))}"
        )

    return total_loss, mac_prec, mac_recall, mac_f1_score, accuracy


# Functions for analyzing prototypes


def get_best_k_protos_for_batch(
    dataset,
    specific_label,
    tokenizer,
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
    if model_new is None:
        print("creating new model")
        model_new = model_class().cuda()
        model_new.load_state_dict(torch.load(model_path))
    dl = torch.utils.data.DataLoader(
        dataset, batch_size=128, shuffle=False, collate_fn=dataset.collate_fn
    )
    loader = tqdm(dl, total=len(dl), unit="batches")
    model_new.eval()
    with torch.no_grad():
        # Updated for negative prototypes
        if model_new.num_neg_protos > 0:
            all_protos = torch.cat(
                (model_new.neg_prototypes, model_new.pos_prototypes), dim=0
            )
        else:
            all_protos = model_new.pos_prototypes

        best_protos = []
        best_protos_dists = []
        indices = []
        for batch in loader:
            input_ids, attn_mask, y = batch
            #             print(y)
            batch_size = input_ids.size(0)
            last_hidden_state = model_new.bart_model.base_model.encoder(
                input_ids.cuda(),
                attn_mask.cuda(),
                output_attentions=False,
                output_hidden_states=False,
            ).last_hidden_state
            if not model_new.dobatchnorm:
                input_for_classfn = model_new.one_by_sqrt_bartoutdim * torch.cdist(
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
                    batch_size, DatasetInfo().num_classes
                ),
                dim=1,
            )
            if do_all:
                temp = torch.topk(input_for_classfn, dim=1, k=topk, largest=False)
            else:
                concerned_idxs = torch.nonzero((predicted == y.cuda())).view(-1)
                temp = torch.topk(
                    input_for_classfn[concerned_idxs], dim=1, k=topk, largest=False
                )
            best_protos.append(temp[1].cpu())
            best_protos_dists.append(
                (
                    temp[0] * torch.sqrt(torch.tensor(model_new.bart_out_dim).float())
                ).cpu()
            )
        #             best_protos.append((torch.topk(input_for_classfn,dim=1,
        #                                               k=topk,largest=False)[1]).cpu())
        best_protos = torch.cat(best_protos, dim=0)
        best_protos_dists = torch.cat(best_protos_dists, dim=0)
    return best_protos, best_protos_dists


def get_bestk_train_data_for_every_proto(
    train_dataset_eval, model_new=None, top_k=3, return_distances=True
):
    """
    for every prototype find out k best similar training examples
    """
    batch_size = 128
    dl = torch.utils.data.DataLoader(
        train_dataset_eval,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=train_dataset_eval.collate_fn,
    )
    #     dl=torch.utils.data.DataLoader(test_dataset,batch_size=batch_size,shuffle=False,
    #                                      collate_fn=test_dataset.collate_fn)
    loader = tqdm(dl, total=len(dl), unit="batches")
    model_new.eval()
    #     model_new=model_new.cpu()
    count = 0
    with torch.no_grad():
        best_train_egs = []
        best_train_egs_values = []
        all_distances = torch.tensor([])
        predict_all = torch.tensor([])
        true_all = torch.tensor([])
        # Updated for negative prototypes
        if model_new.num_neg_protos > 0:
            all_protos = torch.cat(
                (model_new.neg_prototypes, model_new.pos_prototypes), dim=0
            )
        else:
            all_protos = model_new.pos_prototypes
        for batch in loader:
            input_ids, attn_mask, y = batch
            batch_size = input_ids.size(0)
            last_hidden_state = model_new.bart_model.base_model.encoder(
                input_ids.cuda(),
                attn_mask.cuda(),
                output_attentions=False,
                output_hidden_states=False,
            ).last_hidden_state
            if not model_new.dobatchnorm:
                input_for_classfn = model_new.one_by_sqrt_bartoutdim * torch.cdist(
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
                    batch_size, DatasetInfo().num_classes
                ),
                dim=1,
            )
            concerned_idxs = torch.nonzero((predicted == y.cuda())).view(-1)
            #             concerned_idxs=torch.nonzero((predicted==y)).view(-1)
            input_for_classfn = input_for_classfn[concerned_idxs] * torch.sqrt(
                torch.tensor(model_new.bart_out_dim).float()
            )
            #             predict_all=torch.cat((predict_all,predicted.cpu()),dim=0)
            #             true_all=torch.cat((true_all,y.cpu()),dim=0)
            if top_k is None:
                all_distances = torch.cat(
                    (all_distances, input_for_classfn.cpu()), dim=0
                )
            else:
                best = torch.topk(input_for_classfn, dim=0, k=top_k, largest=False)
                best_train_egs.append((best[1] + count * batch_size))
                count += 1
                best_train_egs_values.append(best[0])
    if top_k is None:
        return torch.cat(
            (true_all.view(-1, 1), predict_all.view(-1, 1), all_distances), dim=1
        )
    else:
        best_train_egs = torch.cat(best_train_egs, dim=0)
        best_train_egs_values = torch.cat(best_train_egs_values, dim=0)
        temp = torch.topk(best_train_egs_values, dim=0, k=top_k, largest=False)
        topk_idxs = temp[1]
        final_concerned_idxs = []
        for i in range(best_train_egs.size(1)):
            concerned_idxs = best_train_egs[topk_idxs[:, i], i]
            final_concerned_idxs.append(concerned_idxs)
        #         true_all=torch.cat(true_all,dim=0)
        #         predict_all=torch.cat(predict_all,dim=0)
        return (
            torch.stack(final_concerned_idxs, dim=0).cpu().numpy(),
            temp[0].cpu().numpy(),
        )


def get_distances_for_rdm(train_dataset_eval, model_new=None, return_distances=True):
    """
    for every prototype find out k best similar training examples
    """
    dl = torch.utils.data.DataLoader(
        train_dataset_eval,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=train_dataset_eval.collate_fn,
    )
    #     dl=torch.utils.data.DataLoader(test_dataset,batch_size=batch_size,shuffle=False,
    #                                      collate_fn=test_dataset.collate_fn)
    loader = tqdm(dl, total=len(dl), unit="batches")
    model_new.eval()
    count = 0
    with torch.no_grad():
        best_train_egs = []
        best_train_egs_values = []
        all_distances = torch.tensor([])
        predict_all = torch.tensor([])
        true_all = torch.tensor([])
        all_protos = model_new.pos_prototypes.view(model_new.num_protos, -1)
        for batch in loader:
            input_ids, attn_mask, y = batch
            batch_size = input_ids.size(0)
            last_hidden_state = model_new.bart_model.base_model.encoder(
                input_ids.cuda(),
                attn_mask.cuda(),
                output_attentions=False,
                output_hidden_states=False,
            ).last_hidden_state
            input_for_classfn = model_new.one_by_sqrt_bartoutdim * torch.cdist(
                last_hidden_state.view(batch_size, -1), all_protos
            )
            predicted = torch.argmax(
                model_new.classfn_model(input_for_classfn).view(
                    batch_size, DatasetInfo().num_classes
                ),
                dim=1,
            )
            concerned_idxs = torch.nonzero(
                torch.logical_and(predicted == y.cuda(), y.cuda() == 1)
            ).view(-1)
            #             concerned_idxs=torch.nonzero((predicted==y)).view(-1)
            input_for_classfn = input_for_classfn[concerned_idxs]
            print(torch.sort(input_for_classfn, descending=False, dim=1)[0].cpu())
            break
        return


def print_protos(train_dataset, tokenizer, train_ls, which_protos, protos_train_table):
    df = np.zeros(
        [
            DatasetInfo().num_classes,
            DatasetInfo().num_classes,
        ]
    )
    first_prototypes = collections.defaultdict(list)
    for proto_index in which_protos:
        train_ys = []
        print("@@@" * 10)
        print(f"prototype {proto_index}")
        concerned_idxs = protos_train_table[proto_index]
        sents = tokenizer.batch_decode(
            [train_dataset.x[j] for j in concerned_idxs],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        for index, (sent, y) in enumerate(
            zip(sents, [train_ls[j] for j in concerned_idxs])
        ):
            if index == 0:
                first_prototypes[y].append((sent, proto_index))
            print(sent, y)
            train_ys.append(y)
        for i in range(len(train_ys)):
            for j in range(i + 1, len(train_ys)):
                df[
                    DatasetInfo().dataset_classes_dict[train_ys[i]],
                    DatasetInfo().dataset_classes_dict[train_ys[j]],
                ] += 1
                df[
                    DatasetInfo().dataset_classes_dict[train_ys[j]],
                    DatasetInfo().dataset_classes_dict[train_ys[i]],
                ] += 1
        print(collections.Counter(train_ys))

        print()

    print("all the labels in the prototypes")
    print("cooccurence matrix")
    from matplotlib import pyplot as plt
    import seaborn as sns

    df = pd.DataFrame(
        df,
        index=list(DatasetInfo().dataset_classes_dict.keys()),
        columns=list(DatasetInfo().dataset_classes_dict.keys()),
    )
    _ = plt.figure(figsize=(15, 15))
    sns.heatmap(df, cmap="RdYlGn", linewidths=0.30, annot=True)
    plt.savefig("artifacts/cooc_matrix.png")
    df.to_csv("artifacts/cooccurence_matrix.csv")
    joblib.dump(first_prototypes, "artifacts/first_prototypes.pkl")


def best_protos_for_test(test_dataset, model_new=None, top_k=5):
    batch_size = 60
    dl = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=test_dataset.collate_fn,
    )
    #     loader = tqdm(dl, total=len(dl), unit="batches")
    all_protos = model_new.pos_prototypes
    input_ids, attn_mask, y = next(iter(dl))
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
