
## About

This repo contains the code to run the sentiment analysis pipeline

1. sentiment_pytorch.py   - Pytorch code using DataParallel
2. sentiment_ddp.py       - Pytorch code using Distributed Data Parallel
3. sentiment.py           - Tensorflow MultiGPU

This takes as input the folder containing the input files, located in sentiment/ containing the text data. The processed files, and the results files are written to the folders as shown below.

sentiment/
	*.parquet
	...

sentimentres/
	processed/
	results/

### Copying files

The files can be copied from the Wasabi S3 buckets using the following scripts 

1. copy_files_from_wasabi.sh 
2. copy_files_to_wasabi.sh

### Environment

Use the file env_minimal.yml to setup a Python environment
