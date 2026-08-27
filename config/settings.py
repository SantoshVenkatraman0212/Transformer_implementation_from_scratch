'''
This file provides variables (aliases) for parameters
'''
# Importing necessary libraries
import torch
# Batch size for the Parquet reader (Avoids loading 4.5M samples at once to handle RAM bottleneck)
DATA_PREP_BATCH_SIZE = 100000
# Vocab size
VOCAB_SIZE = 37000 # Vocab size from Attention is All you Need 2017
# Min frequency (BPE Merge)
MIN_FREQUENCY = 2
# Batch size
TRAIN_BATCH_SIZE = 32
# Compute device
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
# Transformer model params
D_MODEL = 512
N_HEADS = 8
N_BLOCKS = 6
DROPOUT = 0.1
SRC_VOCAB_SIZE = 37000
TGT_VOCAB_SIZE = 37000
SRC_SEQ_LEN = 256
TGT_SEQ_LEN = 256
# Training params
WARMUP_STEPS = 4000
N_EPOCHS = 1
PATIENCE = 3
# Memory & Compute optimization params
# No of CPU cores being used in parallel
NUM_WORKERS = 4 
# Boolean variable for loading the data tensors from disk to host memory
PIN_MEMORY = True 
# CPU workers created initially will be used by every other epoch rather than creating from scratch
PERSISTENT_WORKERS = True 
# This boolean variables controls tensor movement to compute device asynchronously
NON_BLOCKING = True
# This boolean controls how many dataloader batches will be fetched preemptively while GPU is processing
PREFETCH_FACTOR = 2
