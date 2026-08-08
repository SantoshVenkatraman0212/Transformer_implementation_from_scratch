'''
This file provides variables (aliases) for parameters
'''
# Batch size for the Parquet reader (Avoids loading 4.5M samples at once to handle RAM bottleneck)
DATA_PREP_BATCH_SIZE = 100000
# Vocab size
VOCAB_SIZE = 37000 # Vocab size from Attention is All you Need 2017
# Min frequency (BPE Merge)
MIN_FREQUENCY = 2