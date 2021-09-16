import numpy as np
from scipy.special import softmax
import csv
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
import GPUtil
import torch
from GPUtil import showUtilization as gpu_usage
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import urllib
import json
import glob
import os
import fastparquet as fp
import pickle
import shutil
import time

# Maximum sequence length for tokenizer is 500. Anything larger than 514 will throw an error for the model
# Maximum single GPU batch size is 50, anything larger will result in gpu usage overflow

print("Device ",torch.cuda.get_device_name())
print("Devices available ",torch.cuda.device_count())

task='offensive'

MODEL = f"cardiffnlp/twitter-roberta-base-{task}"

def get_all_files():
        s3_path = "sentiment/*.parquet"
        all_paths_from_s3 = glob.glob(s3_path)
        print(all_paths_from_s3)
        return(all_paths_from_s3)

class TextLoader(Dataset):
    def __init__(self, file=None, transform=None, target_transform=None, tokenizer=None):
        df = pd.read_parquet(file)
        self.file = df
        print('File name ',file)
        print('Number of records in file ',len(self.file))
        self.file = tokenizer(list(self.file['text']), padding=True, truncation=True, max_length=500, return_tensors='pt')	
        self.file = self.file['input_ids']
        print(self.file.shape)
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.file)

    def __getitem__(self, idx):
        data = self.file[idx]
        return(data)

     
class RobertaModel(nn.Module):
    # Our model

    def __init__(self):
        super(RobertaModel, self).__init__()
        #print("------------------- Initializing once ------------------")
        self.fc = AutoModelForSequenceClassification.from_pretrained(MODEL)

    def forward(self, input):
        output = self.fc(input)
        #print("\tIn Model: input size", input.size())
        return output

pd.set_option('display.max_colwidth', None)

mapping_link = f"https://raw.githubusercontent.com/cardiffnlp/tweeteval/main/datasets/{task}/mapping.txt"
with urllib.request.urlopen(mapping_link) as f:
     html = f.read().decode('utf-8').split("\n")
     csvreader = csv.reader(html, delimiter='\t')

labels = [row[1] for row in csvreader if len(row) > 1]

device = torch.device('cuda')
device_staging = 'cuda:0'
model = RobertaModel()
if(torch.cuda.device_count() > 1):
  print("Parallelizing the model")
  model = nn.DataParallel(model) 
model = model.to(device_staging)

tokenizer = AutoTokenizer.from_pretrained(MODEL)
all_files = get_all_files()

for file in all_files:
	data = TextLoader(file=file, tokenizer=tokenizer)
	train_dataloader = DataLoader(data, batch_size=200, shuffle=True)
	gpu_usage()
	out = []
	t0 = time.time()
	for i,data in enumerate(train_dataloader):
		#gpu_usage()
		input = data.to(device_staging)
		#print(i,input.shape)
		res = model(input)
		#print(res['logits'].shape)
		#out.append(res['logits'].cpu().data)

	print("Prediction time ", time.time() - t0)
	gpu_usage()
	filename = file.split('/')[1]
	output_file = 'sentimentres/results/' + filename + '.npy'
	with open(output_file, 'wb') as f:
		f.write(pickle.dumps(out))

	shutil.move(file, 'sentimentres/processed/' + file.split('/')[1])
