'''
This file downloads the WMT 2014 English-German dataset from Hugging Face,
then extracts the train, test, and val sets inside /data/raw dir
'''
# Importing necessary libraries
from datasets import load_dataset
import pandas as pd
from config.paths import RAW_DATA_DIR


def download_dataset() -> None:
    '''
    This file downloads the WMT-2014 dataset from Hugging Face then flattens it, and saves the dataset files
    '''
    # Creating the data dir if it doens't exist
    RAW_DATA_DIR.mkdir(parents = True, exist_ok = True)
    # Downloading the dataset from Hugging Face
    dataset = load_dataset('wmt14', 'de-en')

    # Iterating through the HF dataset (nested dict)
    for split_type, split in dataset.items():
        df = split.to_pandas()
        translations = pd.DataFrame(df['translation'].to_list())

        # Saving translation cols
        translations.to_parquet(RAW_DATA_DIR / f'{split_type}.parquet', index = False)

def main() -> None:
    '''
    Main function for running the download, and extraction script
    '''
    download_dataset()

if __name__ == '__main__':
    main()

    
