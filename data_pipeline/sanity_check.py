'''
This code ensures that the entire data_pipeline works as intended i.e.
The processed dataset gets loaded, tokenized and shifted, then it is properly batched, 
collated, and converts to a batched iterable PyTorch DataLoader
'''
# Importing necessary libraries
import torch
from config.settings import TRAIN_BATCH_SIZE
from .dataloader import create_dataloaders

def sanity_checks() -> None:
    '''
    This function checks if the overall data_pipeline right from TranslationDataset creation, 
    tokenization, special tokens placement, batching, collation and PyTorch DataLoader creation work.
    The function checks the following:
    1. No of samples i.e. translation pairs in the train/val/test sets
    2. Checks the dtype, and shape for encoder_input, decoder_input, and the label
    3. Checks the special tokens (eos, bos, and pad) placement 
    4. Iterable dataloaders' shape, and data type, and batch_size check
    5. Collation, and padding check
    '''
    # Getting the train, val and test dataloaders
    train_dataloader, val_dataloader, test_dataloader = create_dataloaders()
    # Getting the train/val/test datasets
    train_dataset = train_dataloader.dataset
    val_dataset = val_dataloader.dataset
    test_dataset = test_dataloader.dataset
    print('---------- Dataset Statistics ----------')
    print(f'No of train dataset samples: {len(train_dataset)}')
    print(f'No of val dataset samples: {len(val_dataset)}')
    print(f'No of test dataset samples: {len(test_dataset)}')
    # Retrieving one sample of each dataset
    train_sample = train_dataset[0]
    val_sample = val_dataset[0]
    test_sample = test_dataset[0]
    # Dict for sample analysis
    sample_dict = {'train': train_sample, 'val': val_sample, 'test': test_sample}
    print('\n---------- Sample Specific Statistics ----------')
    for split, sample in sample_dict.items():
        print(f'----- {split} set -----')
        for key, value in sample.items():
            print(f'{key}\nShape: {value.shape}\nData type: {value.dtype}')
            # Ensuring the encoder_input, decoder_input, and decoder_output all have torch.long dtype
            assert sample['encoder_input'].dtype == torch.long
            assert sample['decoder_input'].dtype == torch.long
            assert sample['label'].dtype == torch.long
            # Special token checks
            if split == 'train':
                dataset = train_dataset
            elif split == 'val':
                dataset = val_dataset
            else:
                dataset = test_dataset
            # Encoder input should end with eos token ID
            assert sample['encoder_input'][-1].item() == dataset.eos_id
            # Decoder input should begin with bos token ID
            assert sample['decoder_input'][0].item() == dataset.bos_id
            # Label should end with eos token ID
            assert sample['label'][-1].item() == dataset.eos_id

            # Ensuring autoregressive shifting is working
            assert torch.equal(sample['decoder_input'][1: ], sample['label'][: -1])
        # Checking sample token IDs, and special tokens overall structure
        print('----- Overall tokenized sequence structural check -----')
        print(f'{split} set')
        print(f'Encoder input: {sample['encoder_input'].tolist()}')
        print(f'Decoder input: {sample['decoder_input'].tolist()}')
        print(f'Label: {sample['label'].tolist()}')
        print(f"'<pad>' token ID: {dataset.pad_id}")
        print(f"'<bos>' token ID: {dataset.bos_id}")
        print(f"'<eos> token ID: {dataset.eos_id}'")
            
    # train/val/tet dataloader checks
    # Checking the first batch
    train_batch = next(iter(train_dataloader))
    val_batch = next(iter(val_dataloader))
    test_batch = next(iter(test_dataloader))
    print('---------- Dataloader Statistics ----------')
    dataloader_batch_dict = {'train': train_batch, 'val': val_batch, 'test': test_batch}
    for split, batch in dataloader_batch_dict.items():
        print(f'\n----- {split} batch -----')
        for key, value in batch.items():
            print(f'{key}\nShape: {value.shape}\nData type: {value.dtype}')
            # Asserting the data type are long
            assert value.dtype == torch.long
            # Asserting decoder input and labels have the same shape
            assert batch['decoder_input'].shape == batch['label'].shape
            # Asserting encoder batch size matches with the TRAIN_BATCH_SIZE
            assert batch['encoder_input'].shape[0] == TRAIN_BATCH_SIZE

            if split == 'train':
                dataset = train_dataset
            elif split == 'val':
                dataset = val_dataset
            else:
                dataset = test_dataset

            pad_id = dataset.pad_id

        # Padding config
        print('----- Padding Info -----')
        print(f'Encoder input padding: {(batch['encoder_input'] == pad_id).sum().item()}')
        print(f'Decoder input padding: {(batch['decoder_input'] == pad_id).sum().item()}')
        print(f'Label padding: {(batch['label'] == pad_id).sum().item()}')
        
    print('********** All Data pipeline sanity checks passed **********')

def main() -> None:
    '''
    This function acts as the orchestrator for the sanity_checks function 
    '''
    sanity_checks()

if __name__ == '__main__':
    main()


    
        