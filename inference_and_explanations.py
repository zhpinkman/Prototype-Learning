import torch
from transformers import AutoTokenizer
from preprocess import *

# Load all the functions for analyzing prototypes
from args import args
import utils
import joblib
from models import ProtoTEx


model = ProtoTEx(
    args.num_prototypes,
    bias=False,
    dropout=False,
    special_classfn=True,  # special_classfn=False, # apply dropout only on bias
    p=1,  # p=0.75,
    batchnormlp1=True,
)


model_path = "Models/finegrained_nli_bart_prototex"

print(f"Loading model checkpoint: {model_path}")
pretrained_dict = torch.load(model_path)
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

device = torch.device("cuda")
model.to(device)
tokenizer = AutoTokenizer.from_pretrained("ModelTC/bart-base-mnli")

# Load data
train_dataset, test_dataset, train_labels, test_labels = utils.load_dataset(tokenizer)

best_protos_per_testeg = utils.get_best_k_protos_for_batch(
    dataset=test_dataset,
    specific_label=None,
    model_new=model,
    tokenizer=tokenizer,
    topk=5,
    do_all=True,
)
best_protos_per_traineg = utils.get_best_k_protos_for_batch(
    dataset=train_dataset,
    specific_label=None,
    model_new=model,
    tokenizer=tokenizer,
    topk=5,
    do_all=True,
)
bestk_train_data_per_proto = utils.get_bestk_train_data_for_every_proto(
    train_dataset, model_new=model, top_k=5
)


joblib.dump(bestk_train_data_per_proto, "artifacts/bestk_train_data_per_proto.joblib")
joblib.dump(best_protos_per_testeg, "artifacts/best_protos_per_testeg.joblib")
joblib.dump(best_protos_per_traineg, "artifacts/best_protos_per_traineg.joblib")


all_protos = model.prototypes
torch.save(all_protos, "artifacts/all_protos.pt")

utils.print_protos(
    train_dataset=train_dataset,
    tokenizer=tokenizer,
    train_ls=train_labels,
    which_protos=list(range(args.num_prototypes)),
    protos_train_table=bestk_train_data_per_proto[0],
)

# train_sents_joined = train_sentences
# test_sents_joined = test_sentences

"""
distances generation
test true labels and pred labels 
"""
# loader = tqdm(test_dl, total=len(test_dl), unit="batches")
# model.eval()
# with torch.no_grad():
#     test_true=[]
#     test_pred=[]
#     for batch in loader:
#         input_ids,attn_mask,y=batch
#         classfn_out,_=model(input_ids,attn_mask,y,use_decoder=False,use_classfn=1)
#         predict=torch.argmax(classfn_out,dim=1)
# #         correct_idxs.append(torch.nonzero((predicted==y.cuda())).view(-1)
#         test_pred.append(predict.cpu().numpy())
#         test_true.append(y.cpu().numpy())
# test_true=np.concatenate(test_true)
# test_pred=np.concatenate(test_pred)

"""
distances generation
csv generation
"""
# import csv

# fields = ["S.No.", "Test Sentence","Predicted","Actual","Actual Prop or NonProp"]
# num_protos_per_test=5
# num_train_per_proto=5
# for i in range(num_protos_per_test):
#     for j in range(num_train_per_proto):
#         fields.append(f"Prototype_{i}_wieght0")
#         fields.append(f"Prototype_{i}_wieght1")
#         fields.append(f"Prototype_{i}_Nearest_train_eg_{j}")
#         fields.append(f"Prototype_{i}_Nearest_train_eg_{j}_actuallabel")
#         fields.append(f"Prototype_{i}_Nearest_train_eg_{j}_distance")

# filename = f"{model_path[len('Models/'):]}_nearest.csv"
# weights=model.classfn_model.weight.detach().cpu().numpy()
# with open(filename, 'w') as csvfile:
#     csvwriter = csv.writer(csvfile)
#     csvwriter.writerow(fields)
#     for i in range(len(test_sents_joined)):
# #     for i in range(100):
#         row=[i,test_sents_joined[i],test_pred[i],test_labels[i],test_true[i]]
#         for j in range(num_protos_per_test):
#             proto_idx=best_protos_per_testeg[0][i][j]
#             for k in range(num_train_per_proto):
# #                 print(i,j,k)
#                 row.append(weights[0][proto_idx])
#                 row.append(weights[1][proto_idx])
#                 row.append(train_sents_joined[bestk_train_data_per_proto[0][proto_idx][k]])
#                 row.append(train_labels[bestk_train_data_per_proto[0][proto_idx][k]])
#                 row.append(bestk_train_data_per_proto[1][k][proto_idx])

#         csvwriter.writerow(row)
