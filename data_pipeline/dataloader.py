'''
This file craetes PyTorch DataLoader objects which are batched tokenized and padded tensors
'''
# Importing necessary libraries
from torch.utils.data import DataLoader
from config.paths import PROCESSED_DATA_DICT, TOKENIZER_PATH
from config.settings import TRAIN_BATCH_SIZE, NUM_WORKERS, PIN_MEMORY, PERSISTENT_WORKERS, PREFETCH_FACTOR
from .translation_dataset import TranslationDataset
from .collate import collate_batches

# Function for creating PyTorch DataLoaders for train/val/test sets
def create_dataloaders() -> tuple[DataLoader, DataLoader, DataLoader]:
    '''
    This function creates iterable PyTorch dataloaders for batched iteration through the train/val/test sets
    The batch-wise iteration happens during training
    '''
    # Translation Dataset instance for train/val/test sets
    train_dataset = TranslationDataset(file_path = PROCESSED_DATA_DICT['TRAIN'], tokenizer_path = TOKENIZER_PATH)
    val_dataset = TranslationDataset(file_path = PROCESSED_DATA_DICT['VAL'], tokenizer_path = TOKENIZER_PATH)
    test_dataset = TranslationDataset(file_path = PROCESSED_DATA_DICT['TEST'], tokenizer_path = TOKENIZER_PATH)

    train_dataloader = DataLoader(dataset = train_dataset, batch_size = TRAIN_BATCH_SIZE, shuffle = True, 
                                  collate_fn = lambda batch: collate_batches(batch, pad_id = train_dataset.pad_id), 
                                  num_workers = NUM_WORKERS, pin_memory = PIN_MEMORY, prefetch_factor = PREFETCH_FACTOR, 
                                  persistent_workers = PERSISTENT_WORKERS)

    val_dataloader = DataLoader(dataset = val_dataset, batch_size = TRAIN_BATCH_SIZE, shuffle = False, 
                                  collate_fn = lambda batch: collate_batches(batch, pad_id = val_dataset.pad_id), 
                                  num_workers = NUM_WORKERS, pin_memory = PIN_MEMORY, prefetch_factor = PREFETCH_FACTOR, 
                                  persistent_workers = PERSISTENT_WORKERS)

    test_dataloader = DataLoader(dataset = test_dataset, batch_size = TRAIN_BATCH_SIZE, shuffle = False, 
                                  collate_fn = lambda batch: collate_batches(batch, pad_id = test_dataset.pad_id), 
                                  num_workers = NUM_WORKERS, pin_memory = PIN_MEMORY, prefetch_factor = PREFETCH_FACTOR, 
                                  persistent_workers = PERSISTENT_WORKERS)

    return train_dataloader, val_dataloader, test_dataloader