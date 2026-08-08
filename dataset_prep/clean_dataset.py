'''
This file handles cleaning, and removing inconsistencies from train/val/test sets
'''
# Importing necessary libraries
import math
from pathlib import Path
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm
from config.paths import RAW_DATA_DICT
from config.paths import PROCESSED_DATA_DIR
from config.settings import DATA_PREP_BATCH_SIZE

def data_preprocess(df: pd.DataFrame) -> pd.DataFrame:
    '''
    The function implements the following preprocessing steps:
    1. Removal of rows with null values
    2. Removing leading and trailing whitespaces from sequences 
    3. Removal of rows with empty strings 
    4. Normalizing inter-word multi-whitespaces to single whitespace
    5. Removing duplicated rows

    Args:
        pd.DataFrame
            Pandas dataframe
    
    Returns:
        pd.DataFrame
            Pandas dataframe
    '''
    # Dropping the null rows (rows where any col has null value)
    df = df.dropna()
    # Removing leading and trailing whitespaces
    df = df.apply(lambda col: col.str.strip())
    # Getting the indices of empty strings
    indices = df['de'].eq('') | df['en'].eq('')
    # Only having samples where none of the cols have empty strings
    df = df[~indices]
    # Having only one whitespace between words in deustch and english cols
    df = df.apply(lambda col: col.str.split().str.join(' '))
    # Dropping the duplicated rows (retaining only the 1st one)
    df = df.drop_duplicates(keep = 'first')
    
    return df

def batched_preprocess(in_file_path: Path, out_file_path: Path) -> None:
    '''
    This function performs the following operations:
    1. Creates ParquetFile object for each input file 
    2. Iterates over the entire dataset in batches
    3. Each pyarrow RecordBatch is converted to pandas DataFrame for preprocessing
    4. Converts the processed batch DF to pyarrow table
    5. Creates a ParquetWriter object, appends to the table in batches

    Args:
        in_file_path: Path
            Input file path
        out_file_path: Path
            Output file path
    
    Returns:
        None

    '''
    # Creation of ParquetFile object for the input file using it's absolute path
    parquet_file_obj = pq.ParquetFile(in_file_path)
    # Initializing the parquet writer as None
    # parquet_write is writes the data from pyarrow table to parquet file in batches by appending to the same file by keeping it open
    parquet_writer = None
    try: 
        # Getting total number of batches for each train/val/test split
        # Using math.ceil to account for spillover samples
        total_batches = math.ceil(parquet_file_obj.metadata.num_rows / DATA_PREP_BATCH_SIZE)
        # Only runs when there are batches to be written 
        # Iterating through each batch of size 100k (safe default) of input file parquet object
        # in_file_path.stem here gives name of the dataset split near the progress bar
        for batch in tqdm(parquet_file_obj.iter_batches(batch_size = DATA_PREP_BATCH_SIZE), total = total_batches, desc = in_file_path.stem):
            # Converting each batch from pyarrow RecordBatch to pandas DF to call data_preprocess function
            batch_df = batch.to_pandas()
            # Dataset cleaning (clean_batch_df is a dataframe with shape (BATCH_SIZE, 2))
            clean_batch_df = data_preprocess(batch_df)
            # Converting the cleaned batch DF to pyarrow table for writing the batches
            # Removing the indices as the pyarrow table and the writer already handle them
            clean_batch_table = pa.Table.from_pandas(clean_batch_df, preserve_index = False)
            # If the writer is None then create a ParquetWriter object with output file path, and schema of the pyarrow batch table
            if parquet_writer is None:
                # Here schema refers to col names and dtypes 
                # This is essential as the writer opens the file, keeps writing by checking if the schema of the next batch matches 
                parquet_writer = pq.ParquetWriter(out_file_path, clean_batch_table.schema)
            # Each batch is written as pyarrow table and is converted to parquet
            parquet_writer.write_table(clean_batch_table)
    finally:
        # If the batch iteration fails i.e. if all records have been cleaned and written to PROCESSED_DATA_DIR, and writer isn't None, then the writer is closed
        # This finishes the file write operation
        if parquet_writer is not None:
            parquet_writer.close()



def main() -> None:
    '''
    This function calls the data_preprocess function for preprocessing
    '''
    for split, dataset in RAW_DATA_DICT.items():
        # Writing the processed dataframe as parquet tile to PROCESSED_DATA_DIR
        processed_df_path = f'{PROCESSED_DATA_DIR}/{split.lower()}.parquet'
        batched_preprocess(dataset, processed_df_path)
        print(f'Processed {split} set written to: {processed_df_path}')

if __name__ == '__main__':
    main()





    
        
    

