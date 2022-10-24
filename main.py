import os
# os.environ['TRANSFORMERS_CACHE'] = '/mnt/infonas/data/baekgupta/cache/'
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"]="2" 
from importlib import reload  
import numpy as np
import torch,time
from transformers import BartModel,BartConfig,BartForConditionalGeneration
from transformers import BartTokenizer
from tqdm.notebook import tqdm
import pathlib
from args import args
import pandas as pd

## Custom modules
from preprocess import CustomNonBinaryClassDataset, make_dataset
from preprocess import make_bert_dataset,make_bert_testset
from preprocess import create_labels, labels_set 
from preprocess import BinaryClassDataset

from training import train_simple_ProtoTEx, train_simple_ProtoTEx_adv, train_ProtoTEx_w_neg

## Set cuda 
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def main():    
    ## preprocess the propaganda dataset loaded in the data folder. Original dataset can be found here
    ## https://propaganda.math.unipd.it/fine-grained-propaganda-emnlp.html 

    tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')

    
    train_df = pd.read_csv("data/logical_fallacy/edu_train.csv")
    dev_df = pd.read_csv("data/logical_fallacy/edu_dev.csv")
    test_df = pd.read_csv("data/logical_fallacy/edu_test.csv")
    
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
    # train_dl_eval=torch.utils.data.DataLoader(train_dataset_eval,batch_size=20,shuffle=False,
    #                                  collate_fn=train_dataset_eval.collate_fn)


    if args.model == "ProtoTEx":
        print("ProtoTEx best model: {0}, {1}".format(args.num_prototypes, args.num_pos_prototypes))
        train_ProtoTEx_w_neg(
            train_dl =  train_dl,
            val_dl = val_dl,
            test_dl = test_dl,
            num_prototypes=args.num_prototypes, 
            num_pos_prototypes=args.num_pos_prototypes
        )
    # SimpleProtoTEx can be trained in two different ways. In one case it is by reusing the ProtoTEx class definition 
    # and the other way is to use a dedicated SimpleProtoTEx class definition. Both of the implementations are available below. 
    # The dedicated SimpleProtoTEx class definition shall reproduce the results mentioned in the paper. 
    
    elif args.model == "SimpleProtoTExAdv":
        print("Use ProtoTEx Class definition for Simple ProtoTEx")
        train_simple_ProtoTEx_adv(
            train_dl = train_dl,
            val_dl = val_dl,
            test_dl = test_dl,
            train_dataset_len = len(train_dataset),
            num_prototypes=args.num_prototypes, 
            num_pos_prototypes=args.num_pos_prototypes
        ) 
    
    elif args.model == "SimpleProtoTEx":
        print("Dedicated simple prototex")
        train_simple_ProtoTEx(
             train_dl, 
             val_dl, 
             test_dl,
             train_dataset_len = len(train_dataset),
             modelname="0406_simpleprotobart_onlyclass_lp1_lp2_fntrained_20_train_nomask_protos",
             num_prototypes=args.num_prototypes 
        )


if __name__ == '__main__':
    main()