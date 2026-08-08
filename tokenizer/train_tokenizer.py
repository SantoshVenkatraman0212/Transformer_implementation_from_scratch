'''
This file streams batches of processed training data, and uses Hugging Face's Byte Pair Encoding (BPE) Tokenizer.
Tokenization is performed only on the training set
'''
# Importing necessry libraries
import math
from pathlib import Path
import pyarrow.parquet as pq
from tqdm import tqdm
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Metaspace
from tokenizers.decoders import Metaspace as MetaspaceDecoder
from tokenizers.trainers import BpeTrainer
from config.settings import DATA_PREP_BATCH_SIZE, VOCAB_SIZE, MIN_FREQUENCY
from config.paths import PROCESSED_DATA_DICT, ARTIFACTS_DIR

def train_batch_streamer(file_path: Path) -> None:
    '''
    This function creates a ParquetFile object and streams batches of processed train parquet file as pyarrow RecordBatches.
    From each batch, the function retrieves the deutsch, and english sequences

    Args:
        file_path: Path
            Absolute POSIX file path for the train file
    Returns:
        None
    '''
    # parquet file object
    parquet_file_obj = pq.ParquetFile(file_path)
    # Getting the batch size for the processed train file
    total_batches = math.ceil(parquet_file_obj.metadata.num_rows / DATA_PREP_BATCH_SIZE)
    # Iterating through the batches
    for batch in tqdm(parquet_file_obj.iter_batches(DATA_PREP_BATCH_SIZE), total = total_batches, desc = file_path.stem):
        # Converting each pyarrow RecordBatch to pandas DF
        batch_processed_train_df = batch.to_pandas()
        # Getting the sequences from de and en columns
        for seq in batch_processed_train_df['de']:
            yield seq

        for seq in batch_processed_train_df['en']:
            yield seq 

def train_batch_tokenizer(file_path: Path, vocab_size: int, min_frequency: int) -> Tokenizer:
    '''
    This function creates a BPE tokenizer for tokenizing the input train sequence
    Args:
        file_path: Path
            Absolute POSIX file path 
        vocab_size: int
            No of unique tokens in the transformer vocabulary
        min_frequency: int
            Min no of pair occurrences required for byte merge
    
    Returns:
        Tokenizer
            A BPE tokenizer trained on the processed train sequence
    '''
    # Creating an empty BPE Tokenizer instance with '<unk>' token for encoding the token that the tokenizer doesn't have in it's vocab
    tokenizer = Tokenizer(BPE(unk_token = '<unk>'))
    # Pretokenizer instance, splitting the sequences into words 
    tokenizer.pre_tokenizer = Metaspace(replacement = '_', prepend_scheme = 'always')
    # Tokenizer Decoder instance
    tokenizer.decoder = MetaspaceDecoder(replacement = '_', prepend_scheme = 'always')
    # Creating a tokenizer trainer instance
    tokenizer_trainer = BpeTrainer(vocab_size = vocab_size, min_frequency = min_frequency, special_tokens = ['<pad>', '<unk>', '<bos>', '<eos>'])
    # Training the BPE tokenizer on the streamed batches
    tokenizer.train_from_iterator(train_batch_streamer(file_path = file_path), trainer = tokenizer_trainer)

    return tokenizer

def verify_tokenizer(tokenizer: Tokenizer) -> None:
    '''
    This function verifies if the tokenizer is able to encode sample sequences
    Args:
        tokenizer: Tokenizer
            Trained BPE tokenizer
    Returns:
        None
    '''
    sample_sentences = [ 'Ich liebe maschinelles Lernen.', 'I love machine learning.', 'Donaudampfschifffahrtsgesellschaftskapitän']
    # Printing the tokenizer vocabulary size
    print(f'Tokenizer vocab size: {tokenizer.get_vocab_size()}')
    # Iterating through the sample sentences for getting the token IDs
    for seq in sample_sentences:
        print(f'Sample sequence: {seq}')
        encoding = tokenizer.encode(seq)
        print(f'Tokens: {encoding.tokens}')
        print(f'Token IDs: {encoding.ids}')
        print(f'Decoded sequence: {tokenizer.decode(encoding.ids)}')

def main() -> None:
    '''
    The main function orchestrates the processed train data streaming, tokenizer training, and verification
    '''
    # Training the BPE tokenizer on the streamed processed train batches
    tokenizer = train_batch_tokenizer(file_path = PROCESSED_DATA_DICT['TRAIN'], vocab_size = VOCAB_SIZE, min_frequency = MIN_FREQUENCY)
    # Saving the tokenizer
    ARTIFACTS_DIR.mkdir(exist_ok = True, parents = True)
    tokenizer.save(str(f'{ARTIFACTS_DIR}/tokenizer_en_de_37k.json'))
    # Checking if the tokenizer works
    verify_tokenizer(tokenizer = tokenizer)

if __name__ == '__main__':
    main()

