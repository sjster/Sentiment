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
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import torch.multiprocessing as mp
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

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def get_all_files():
        s3_path = "sentiment/*.parquet"
        all_paths_from_s3 = glob.glob(s3_path)
        print(all_paths_from_s3)
        return(all_paths_from_s3)

class TextLoader(Dataset):
    def __init__(self, file=None, transform=None, target_transform=None, tokenizer=None):
        df = pd.read_parquet(file)
        self.file = df
        print('File name from textloader ',file)
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

    def forward(self, input, rank):
        output = self.fc(input)
        #print("\tIn Model: input size", input.size(), rank)
        return(output)


def rank_inference(rank, world_size, args, use_cuda):

     #--------------- Setup -------------#
    print(f"Running basic DDP example on rank {rank}.")
    setup(rank, world_size)
    #-----------------------------------#

    pd.set_option('display.max_colwidth', None)

    mapping_link = f"https://raw.githubusercontent.com/cardiffnlp/tweeteval/main/datasets/{task}/mapping.txt"
    with urllib.request.urlopen(mapping_link) as f:
         html = f.read().decode('utf-8').split("\n")
         csvreader = csv.reader(html, delimiter='\t')

    labels = [row[1] for row in csvreader if len(row) > 1]

    device = torch.device('cuda')
    device_staging = 'cuda:0'
    model = RobertaModel().to(rank)
    model = DDP(model)

    tokenizer = AutoTokenizer.from_pretrained(MODEL)

    all_files = args['files']
    num_files = len(all_files)
    work_size = int(num_files / world_size)
    if(rank == 0):
        index_start = 0
        index_end = work_size
    elif(rank == 1):
        index_start = work_size
        index_end = num_files
    files = args['files'][index_start: index_end]

    print('Rank from inference and files --  ',rank,files)   
    
    tstart = time.time()
    for file in files:
        data = TextLoader(file=file, tokenizer=tokenizer)
        train_dataloader = DataLoader(data, batch_size=50, shuffle=False)
        gpu_usage()
        out = []
        t0 = time.time()
        for i,data in enumerate(train_dataloader):
            #gpu_usage()
            input = data.to(rank)
            #print(i,input.shape, rank)
            res = model(input, rank)
            #print(res['logits'].shape)
            out.append(res['logits'].cpu().data)

        #print(res['logits'].cpu().data, rank)
        print("Prediction time ", rank, time.time() - t0)
        gpu_usage()
        filename = file.split('/')[1]
        output_file = 'sentimentres/results/' + filename + '.npy'
        with open(output_file, 'wb') as f:
            f.write(pickle.dumps(out))

        shutil.move(file, 'sentimentres/processed/' + file.split('/')[1])

    cleanup()
    print("Total execution time ",time.time() - t0)


if __name__ == "__main__":

	world_size = torch.cuda.device_count()
	use_cuda = True
	args = {}
	all_files = get_all_files()
	args['files'] = all_files
	print(args)
	mp.spawn(rank_inference, args=(world_size, args, use_cuda), nprocs=world_size, join=True)

