from sklearn.metrics import classification_report
# from training_phobert import model, tokenizer, tags_set, device
import torch
import json
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from transformers import AutoModel, AutoTokenizer, AdamW
tokenizer = AutoTokenizer.from_pretrained('vinai/phobert-base')

# Process the train dataset:
tags = []
X = []
y = []

# Load train and validation dataset
with open('content.json', 'r', encoding="utf-8") as c:
    contents = json.load(c)
with open('val_content.json', 'r', encoding="utf-8") as v:
    val_contents = json.load(v)
# Load model PhoBERT and its tokenizer
phobert = AutoModel.from_pretrained('vinai/phobert-base')
tokenizer = AutoTokenizer.from_pretrained('vinai/phobert-base')
for content in contents['intents']:
    tag = content['tag']
    for pattern in content['patterns']:
        X.append(pattern)
        tags.append(tag)

tags_set = sorted(set(tags))

with open('test_content.json', 'r', encoding='utf-8') as c:
    contents = json.load(c)

tags_test = []
X_test = []
y_test = []

for content in contents['intents']:
    tag = content['tag']
    for pattern in content['patterns']:
        X_test.append(pattern)
        tags_test.append(tag)

for tag in tags_test:
    label = tags_set.index(tag)
    y_test.append(label)
token_test = {}
token_test = tokenizer.batch_encode_plus(
    X_test,
    max_length=13,
    padding='max_length',
    truncation=True
)
X_test_mask = torch.tensor(token_test['attention_mask'])
X_test = torch.tensor(token_test['input_ids'])
y_test = torch.tensor(y_test)

path = 'saved_weights.pth'
model.load_state_dict(torch.load(path))
with torch.no_grad():
    preds = model(X_test.to(device), X_test_mask.to(device))
    preds = preds.detach().cpu().numpy()

preds = np.argmax(preds, axis=1)
print(classification_report(y_test, preds))
