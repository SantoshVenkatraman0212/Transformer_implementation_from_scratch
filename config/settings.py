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
