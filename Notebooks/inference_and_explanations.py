#!/usr/bin/env python
# coding: utf-8

# In[2]:


import os 
from importlib import reload  
import numpy as np
import torch,time
from transformers import BartModel,BartConfig,BartForConditionalGeneration,BartForCausalLM, BartTokenizer
from tqdm.notebook import tqdm
from torch import nn


# In[3]:


import sys

MOD_FOLDER = '../'
# setting path to enable import from the parent directory
sys.path.append(MOD_FOLDER)
print(sys.path)


# In[4]:


from models import SimpleProtoTex
from models import ProtoTEx


# In[5]:


num_prototypes = 36
num_pos_prototypes = 36
model = ProtoTEx(num_prototypes, 
                 num_pos_prototypes,
                 bias=False, 
                 dropout=False, 
                 special_classfn=True, # special_classfn=False, ## apply dropouonly on bias 
                 p=1, #p=0.75,
                 batchnormlp1=True)
# model


# In[20]:


model_path = "Models/0408_NegProtoBart_protos_xavier_large_bs20_20_woRat_noReco_g2d_nobias_nodrop_cu1_PosUp_normed"


# In[21]:


model.load_state_dict(torch.load(MOD_FOLDER + model_path))


# In[13]:


device = torch.device('cuda:0')
model.to(device)


# In[7]:


tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')


# In[8]:


## Load all the functions for analyzing prototypes
from utils import *


# In[9]:


## Load all the data classes
from preprocess import *
import pandas as pd


# In[10]:


## Load data

train_df = pd.read_csv("../data/logical_fallacy/edu_train.csv")
dev_df = pd.read_csv("../data/logical_fallacy/edu_dev.csv")
test_df = pd.read_csv("../data/logical_fallacy/edu_test.csv")

train_df = train_df[train_df['updated_label'] != 'equivocation']
dev_df = dev_df[dev_df['updated_label'] != 'equivocation']
test_df = test_df[test_df['updated_label'] != 'equivocation']

train_sentences = train_df['source_article'].tolist()
dev_sentences = dev_df['source_article'].tolist()
test_sentences = test_df['source_article'].tolist()

train_labels = train_df['updated_label'].tolist()
dev_labels = dev_df['updated_label'].tolist()
test_labels = test_df['updated_label'].tolist()




train_dataset = CustomNonBinaryClassDataset(
    sentences = train_sentences,
    labels = train_labels,
    tokenizer = tokenizer
)
dev_dataset = CustomNonBinaryClassDataset(
    sentences = dev_sentences,
    labels = dev_labels,
    tokenizer=tokenizer
)
test_dataset = CustomNonBinaryClassDataset(
    sentences = test_sentences,
    labels = test_labels,
    tokenizer = tokenizer
)

train_dl=torch.utils.data.DataLoader(train_dataset,batch_size=20,shuffle=True,
                                 collate_fn=train_dataset.collate_fn)
val_dl=torch.utils.data.DataLoader(dev_dataset,batch_size=128,shuffle=False,
                                 collate_fn=dev_dataset.collate_fn)
test_dl=torch.utils.data.DataLoader(test_dataset,batch_size=128,shuffle=False,
                                 collate_fn=test_dataset.collate_fn)


# In[15]:

test_sents = test_sentences

best_protos_per_testeg = get_best_k_protos_for_batch(
    dataset = test_dataset,
    specific_label=None, 
    model_new=model, 
    tokenizer=tokenizer, 
    topk= 5, 
    do_all=True
)



# In[16]:


train_sents_joined = train_sentences
test_sents_joined = test_sentences

# train_sents_joined=[" ".join(i) for i in train_sents]
# test_sents_joined=[" ".join(i) for i in test_sents]


# In[18]:


"""
distances generation
test true labels and pred labels 
"""
loader = tqdm(test_dl, total=len(test_dl), unit="batches")
model.eval()    
with torch.no_grad():
    test_true=[]
    test_pred=[]
    for batch in loader:
        input_ids,attn_mask,y=batch
        classfn_out,_=model(input_ids,attn_mask,y,use_decoder=False,use_classfn=1)
        predict=torch.argmax(classfn_out,dim=1)
#         correct_idxs.append(torch.nonzero((predicted==y.cuda())).view(-1)
        test_pred.append(predict.cpu().numpy())
        test_true.append(y.cpu().numpy())
test_true=np.concatenate(test_true)
test_pred=np.concatenate(test_pred)


# In[40]:


bestk_train_data_per_proto=get_bestk_train_data_for_every_proto(train_dataset, 
                                                   model_new=model, top_k=5)



print_protos(
    train_dataset = train_dataset, 
    tokenizer = tokenizer, 
    train_ls = train_labels, 
    which_protos=list(range(num_prototypes)), 
    protos_train_table=bestk_train_data_per_proto[0]
)

"""
distances generation
csv generation
"""
import csv

fields = ["S.No.", "Test Sentence","Predicted","Actual","Actual Prop or NonProp"]
num_protos_per_test=5
num_train_per_proto=5
for i in range(num_protos_per_test):
    for j in range(num_train_per_proto):
        fields.append(f"Prototype_{i}_wieght0")
        fields.append(f"Prototype_{i}_wieght1")
        fields.append(f"Prototype_{i}_Nearest_train_eg_{j}")
        fields.append(f"Prototype_{i}_Nearest_train_eg_{j}_actuallabel")
        fields.append(f"Prototype_{i}_Nearest_train_eg_{j}_distance")
        
filename = f"{model_path[len('Models/'):]}_nearest.csv"
weights=model.classfn_model.weight.detach().cpu().numpy()
with open(filename, 'w') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(fields)
    for i in range(len(test_sents_joined)):
#     for i in range(100):
        row=[i,test_sents_joined[i],test_pred[i],test_labels[i],test_true[i]]
        for j in range(num_protos_per_test):
            proto_idx=best_protos_per_testeg[0][i][j]
            for k in range(num_train_per_proto):
#                 print(i,j,k)
                row.append(weights[0][proto_idx])
                row.append(weights[1][proto_idx])
                row.append(train_sents_joined[bestk_train_data_per_proto[0][proto_idx][k]])
                row.append(train_labels[bestk_train_data_per_proto[0][proto_idx][k]])
                row.append(bestk_train_data_per_proto[1][k][proto_idx])

        csvwriter.writerow(row)

