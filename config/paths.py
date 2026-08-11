'''
This file is for handling and resolving all the file paths, so that these paths work irrespective of where the code is run from
'''
# Importing necessary libraries
from pathlib import Path

# Getting the overall parent dir of paths.py file
PROJECT_DIR = Path(__file__).resolve().parent.parent

# Paths to the dataset
DATA_DIR = PROJECT_DIR / 'data'
RAW_DATA_DIR = DATA_DIR / 'raw'
PROCESSED_DATA_DIR = DATA_DIR / 'processed'
RAW_TRAIN_DIR = RAW_DATA_DIR / 'train.parquet'
RAW_VAL_DIR = RAW_DATA_DIR / 'validation.parquet'
RAW_TEST_DIR = RAW_DATA_DIR / 'test.parquet'
PROCESSED_TRAIN = PROCESSED_DATA_DIR / 'train.parquet'
PROCESSED_VAL = PROCESSED_DATA_DIR / 'val.parquet'
PROCESSED_TEST = PROCESSED_DATA_DIR / 'test.parquet'
# Paths to ARTIFACTS
ARTIFACTS_DIR = PROJECT_DIR / 'artifacts'
TOKENIZER_PATH = ARTIFACTS_DIR / 'tokenizer_en_de_37k.json'
# Raw data dict with split mapping
RAW_DATA_DICT = {'TRAIN': RAW_TRAIN_DIR, 'VAL': RAW_VAL_DIR, 'TEST': RAW_TEST_DIR}
PROCESSED_DATA_DICT = {'TRAIN': PROCESSED_TRAIN, 'VAL': PROCESSED_VAL, 'TEST': PROCESSED_TEST}