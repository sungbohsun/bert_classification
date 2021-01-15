import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import json
import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import torch # the main pytorch library
import torch.nn.functional as f # the sub-library containing different functions for manipulating with tensors
from transformers import BertModel, BertTokenizer


from utils import *
from transformers import BertTokenizer, BertForSequenceClassification
from torchtext.data import Field, LabelField, TabularDataset, BucketIterator, Iterator


class BERT(nn.Module):

    def __init__(self):
        super(BERT, self).__init__()
        
        options_name = "bert-base-chinese"
        #BertForSequenceClassification.config_class = 14
        self.encoder = BertForSequenceClassification.from_pretrained(options_name,num_labels = 14)
    def forward(self, text, label):
        loss, text_fea = self.encoder(text, labels=label)[:2]
        #print(self.encoder(text, labels=label))
        return loss, text_fea
    
model_inten = BERT().to(device)
load_checkpoint('inten_best_model.pt', model_inten)

class BERT(nn.Module):

    def __init__(self):
        super(BERT, self).__init__()
        
        options_name = "bert-base-chinese"
        #BertForSequenceClassification.config_class = 14
        self.encoder = BertForSequenceClassification.from_pretrained(options_name,num_labels = 15)
    def forward(self, text, label):
        loss, text_fea = self.encoder(text, labels=label)[:2]
        #print(self.encoder(text, labels=label))
        return loss, text_fea
    
def QA(Qu):
    qus = np.array(tokenizer.encode(Qu))
    qus_ = torch.tensor([np.pad(qus, (0, 128-len(qus)), 'constant')]).to(device)
    L = torch.tensor([0]).to(device)
    _, output_inten  = model_inten(qus_, L)
    _, output_depart = model_depart(qus_, L)
    pred_inten  = torch.argmax(output_inten, 1).tolist()
    pred_depart = torch.argmax(output_depart, 1).tolist()
    return depart_dic[pred_depart[0]],inten_dic[pred_inten[0]],float(output_depart.max()),float(output_inten.max())

class Ans_setup():
    
    def __init__(self,depart_threshold,inten_threshold,qus_threshold):
        self.depart_threshold = depart_threshold
        self.inten_threshold = inten_threshold
        self.qus_threshold = qus_threshold
        
    def Ans(self,qu):
        if qu[-1] != '?':
            qu = qu+'?'
        df = pd.read_csv('QA_dataset_v4_1222.csv')
        depart,inten,depart_prob,inten_porb = QA(qu)
        df = df[df['商管學院單位'] == depart]
        df = df[df['意圖'] == inten].reset_index()
        texts = df['問題 (Question)'].values.tolist()

        texts += [qu]

        encodings = tokenizer(
            texts, # the texts to be tokenized
            padding=True, # pad the texts to the maximum length (so that all outputs have the same length)
            return_tensors='pt' # return the tensors (not lists)
        )

        encodings = encodings.to(device)

        # disable gradient calculations
        with torch.no_grad():
            # get the model embeddings
            embeds = model(**encodings)

        embeds = embeds[0]
        CLSs = embeds[:, 0, :]

        # normalize the CLS token embeddings
        normalized = f.normalize(CLSs, p=2, dim=1)
        # calculate the cosine similarity
        cls_dist = normalized.matmul(normalized.T)
        ans = int(cls_dist[-1][:-1].argmax())
        qus_prob = float(cls_dist[-1][:-1].max())

        if self.depart_threshold < depart_prob:
            print('depart :',depart)
            print('depart_prob :',depart_prob)
        else:
            print('can not identify depart')

        if self.inten_threshold < inten_porb:
            print('inten :',inten)
            print('inten_porb :',inten_porb)
        else:
            print('can not identify inten')

        if self.depart_threshold < depart_prob and self.inten_threshold < inten_porb:
            if self.qus_threshold < qus_prob:
                print('similarity qus:',df['問題 (Question)'][ans])
                print('similarity ans:',df['答案 (Answer)'][ans])
                print('prob :',qus_prob)
            else:
                print('not match qustion')
    
model_depart = BERT().to(device)
load_checkpoint('depart_best_model.pt', model_depart)

with open('inten_dic.json') as json_file:
    dic = json.load(json_file)
inten_dic = {v:k for k,v in zip(dic.keys(),dic.values())}

with open('depart_dic.json') as json_file:
    dic = json.load(json_file)
depart_dic = {v:k for k,v in zip(dic.keys(),dic.values())}

os.environ["CUDA_VISIBLE_DEVICES"]="0"
torch.cuda.get_device_name(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
MAX_SEQ_LEN = 128
PAD_INDEX = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
UNK_INDEX = tokenizer.convert_tokens_to_ids(tokenizer.unk_token)

label_field = Field(sequential=False, tokenize=y_tokenize, use_vocab=False, batch_first=True)
text_field  = Field(use_vocab=False, tokenize=tokenizer.encode, lower=False, include_lengths=False, batch_first=True,
                   fix_length=MAX_SEQ_LEN, pad_token=PAD_INDEX, unk_token=UNK_INDEX)
fields = [('label', label_field),('titletext', text_field)]

bert_version = 'bert-base-chinese'
tokenizer = BertTokenizer.from_pretrained(bert_version)
model = BertModel.from_pretrained(bert_version)

model = model.eval()
model = model.to(device)

