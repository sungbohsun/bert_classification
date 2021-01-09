import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns

from torchtext.data import Field, LabelField, TabularDataset, BucketIterator, Iterator
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from utils import *

class BERT(nn.Module):

    def __init__(self):
        super(BERT, self).__init__()
        
        options_name = "bert-base-chinese"
        self.encoder = BertForSequenceClassification.from_pretrained(options_name,num_labels = 14)
    def forward(self, text, label):
        loss, text_fea = self.encoder(text, labels=label)[:2]
        return loss, text_fea
    

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

# TabularDataset

train,valid = TabularDataset.splits(path='./', train='data/inten_train.csv',validation='data/inten_valid.csv', format='CSV', fields=fields, skip_header=True)

train_iter = BucketIterator(train, batch_size=64, sort_key=lambda x: len(x.titletext),
                            device=device, train=True, sort=True, sort_within_batch=True)

valid_iter = BucketIterator(valid, batch_size=64, sort_key=lambda x: len(x.titletext),
                            device=device, train=True, sort=True, sort_within_batch=True)

model = BERT().to(device)
optimizer = optim.Adam(model.parameters(), lr=2e-5)

# Training Function

criterion = nn.BCELoss()
train_loader = train_iter
valid_loader = valid_iter
num_epochs = 30
eval_every = 10
file_path = './'
best_valid_loss = float("Inf")

# initialize running values
running_loss = 0.0
valid_running_loss = 0.0
global_step = 0
train_loss_list = []
valid_loss_list = []
global_steps_list = []
# training loop
model.train()
for epoch in range(num_epochs):
    for i in train_loader:
        labels = i.label
        titletext = i.titletext
        labels = labels.type(torch.LongTensor)           
        labels = labels.to(device)
        titletext = titletext.type(torch.LongTensor)  
        titletext = titletext.to(device)
        output = model(titletext, labels)
        loss, _ = output

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # update running values
        running_loss += loss.item()
        global_step += 1

        # evaluation step
        if global_step % eval_every == 0:
            model.eval()
            with torch.no_grad():                    

                # validation loop
                for i in valid_loader:
                    labels = i.label
                    titletext = i.titletext
                    labels = labels.type(torch.LongTensor)           
                    labels = labels.to(device)
                    titletext = titletext.type(torch.LongTensor)  
                    titletext = titletext.to(device)
                    output = model(titletext, labels)
                    loss, _ = output

                    valid_running_loss += loss.item()

            # evaluation
            average_train_loss = running_loss / eval_every
            average_valid_loss = valid_running_loss / len(valid_loader)
            train_loss_list.append(average_train_loss)
            valid_loss_list.append(average_valid_loss)
            global_steps_list.append(global_step)

            # resetting running values
            running_loss = 0.0                
            valid_running_loss = 0.0
            model.train()

            # print progress
            print('Epoch [{}/{}], Step [{}/{}], Train Loss: {:.4f}, Valid Loss: {:.4f}'
                  .format(epoch+1, num_epochs, global_step, num_epochs*len(train_loader),
                          average_train_loss, average_valid_loss))

            # checkpoint
            if best_valid_loss > average_valid_loss:
                best_valid_loss = average_valid_loss
                save_checkpoint(file_path + 'inten_best_model.pt', model, best_valid_loss)

save_metrics(file_path + 'inten_best_metrics.pt', train_loss_list, valid_loss_list, global_steps_list)
print('Finished Training!')

train_loss_list, valid_loss_list, global_steps_list = load_metrics( 'inten_best_metrics.pt')
plt.plot(global_steps_list, train_loss_list, label='Train')
plt.plot(global_steps_list, valid_loss_list, label='Valid')
plt.xlabel('Global Steps')
plt.ylabel('Loss')
plt.legend()
plt.show() 
plt.savefig('png/inten_metrics.png')
# Evaluation Function
#def evaluate(model, test_loader):

best_model = BERT().to(device)
load_checkpoint('inten_best_model.pt', best_model)

y_pred = []
y_true = []

model.eval()
with torch.no_grad():
    for i in valid_iter:
            labels = i.label
            titletext = i.titletext
            labels = labels.type(torch.LongTensor)           
            labels = labels.to(device)
            titletext = titletext.type(torch.LongTensor)  
            titletext = titletext.to(device)
            output = model(titletext, labels)

            loss, output = output
            y_pred.extend(torch.argmax(output, 1).tolist())
            y_true.extend(labels.tolist())

print('Classification Report:')
print(classification_report(y_true, y_pred, digits=4))

cm = confusion_matrix(y_true, y_pred, list(range(14)))
fig = plt.figure(figsize=(10,10),dpi=80)
ax = fig.add_subplot(2,1,1)
sns.heatmap(cm, annot=True, ax = ax, cmap='Blues', fmt="d")

ax.set_title('Confusion Matrix')

ax.set_xlabel('Predicted Labels')
ax.set_ylabel('True Labels')
fig.savefig('png/inten_confusion_matrix.png')
