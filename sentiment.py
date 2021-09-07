#from tensorflow.python.client import device_lib
#device_lib.list_local_devices()

import numpy as np
from scipy.special import softmax
import csv
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
import pandas as pd
import tensorflow as tf
import urllib
import json
import glob
import os
import fastparquet as fp
import pickle
import shutil

task='offensive'
MODEL = f"cardiffnlp/twitter-roberta-base-{task}"
PATH = '/home/vt/extra_storage/Production/temporary_files/df_sampled.json/'
PATH = '/home/vt/extra_storage/Production/output/tweets_tokenized_sentiment.json/'
pd.set_option('display.max_colwidth', None)

tokenizer = AutoTokenizer.from_pretrained(MODEL)

mapping_link = f"https://raw.githubusercontent.com/cardiffnlp/tweeteval/main/datasets/{task}/mapping.txt"
with urllib.request.urlopen(mapping_link) as f:
     html = f.read().decode('utf-8').split("\n")
     csvreader = csv.reader(html, delimiter='\t')

labels = [row[1] for row in csvreader if len(row) > 1]

model = TFAutoModelForSequenceClassification.from_pretrained(MODEL)

s3_path = "sentiment/*.parquet"
all_paths_from_s3 = glob.glob(s3_path)
for file in all_paths_from_s3:
  df = pd.read_parquet(file)
  print(file)
  tokenized = tokenizer(list(df['text']), padding=True, return_tensors='tf')
  res = model.predict(tokenized['input_ids'], batch_size=100, use_multiprocessing=True)
  output_file = 'sentimentres/results/' + file.split('/')[1].split('.')[0] + '.npy'
  print(output_file)
  with open(output_file, 'wb') as f:
        f.write(pickle.dumps(res['logits'])) 
  shutil.move(file, 'sentimentres/processed/' + file.split('/')[1])
