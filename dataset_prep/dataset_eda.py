'''
This file performs in-depth EDA to clean the text by dataset schema check, null value handling, dedup etc
on train, val and test sets
'''
# Importing necessary libraries
import pandas as pd
import numpy as np
from config.paths import RAW_DATA_DICT

def perform_eda(file_path: str) -> None:
    '''
    This function analyzes the input dataset to extract cols, dtype, 
    null and empty string handling, and gives overall base statistics
    Args:
        file_path: str
            train/val/test path
    Returns:
        None
    '''
    df = pd.read_parquet(file_path)
    # Getting the basic dataset statistics
    print('---------- Dataset exploration report ----------\n')
    print(f'Dataset preview (1st 5 samples)\n{df.head()}\n')
    print(f'Columns\n{df.columns}\n')
    print(f'Dataset shape:\n{df.shape}\n')
    print('Column data types\n')
    print(f'{df.info()}\n')
    # Checking for empty cells in each col
    print(f'Null values\n{df.isnull().sum()}\n')
    # Checking for empty string in each col
    print(f'Empty string values\n{df.apply(lambda col: col.str.strip().eq('').sum())}\n')
    # No of duplicated samples
    print(f'Duplicate rows\n{df.duplicated().sum()}\n\n')
    # Getting the mean, median, and standard deviation of the sequence lengths in english and german 
    de_str_len = [len(s) for s in df['de'].str.strip()]
    en_str_len = [len(s) for s in df['en'].str.strip()]
    print('----- String length stats -----')
    print(f'Deutsch\nmax: {np.max(de_str_len)} | min: {np.min(de_str_len)} | mean: {np.mean(de_str_len)} | median: {np.median(de_str_len)} | standard deviation: {np.std(de_str_len)}\n')
    print(f'English\nmax: {np.max(en_str_len)} | min: {np.min(en_str_len)} | mean: {np.mean(en_str_len)} | median: {np.median(en_str_len)} | standard deviation: {np.std(en_str_len)}')
    
def main() -> None:
    '''
    Main function for orchestration of EDA
    '''
    for split, path in RAW_DATA_DICT.items():
        print(f'{split} SET')
        perform_eda(path)
        print('\n\n')

if __name__ == '__main__':
    main()


