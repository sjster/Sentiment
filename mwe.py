import numpy as np
from scipy.special import softmax
import csv
import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
import json
import glob
import os
import pickle
import shutil
import time
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
	tf.config.experimental.set_memory_growth(gpu, True)
task='offensive'
MODEL = f"cardiffnlp/twitter-roberta-base-{task}"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = TFAutoModelForSequenceClassification.from_pretrained(MODEL)
tokenized = tokenizer(["Hello there", "Howdy hhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhh"],padding=True, return_tensors='tf')
res = model.predict(tokenized['input_ids'], batch_size=100, use_multiprocessing=True)
print(res)
