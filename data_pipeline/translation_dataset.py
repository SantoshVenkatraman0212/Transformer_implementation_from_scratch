'''
This file loads the preprocessed data, and tokenizes it, and performs autoregressive shifting for making it transformer compatible
'''
# Importing necessary libraries
from pathlib import Path
import torch
from datasets import load_dataset
from torch.utils.data import Dataset
from tokenizers import Tokenizer
from config.settings import SRC_SEQ_LEN, TGT_SEQ_LEN
class TranslationDataset(Dataset):
    '''
    This class acts as a PyTorch Dataset class that instantiates dataset, tokenizer, and special tokens
    tokenizes the samples and returns the data in the format, that can be used by the downstream PyTorch DataLoader
    '''
    def __init__(self, file_path: Path, tokenizer_path: Path) -> None:
        # HF Dataset instance
        self.dataset = load_dataset('parquet', data_files = str(file_path), split = 'train')
        # Tokenizer instance
        self.tokenizer = Tokenizer.from_file(str(tokenizer_path))
        # Instaces for special tokens
        # Pad token ID
        self.pad_id = self.tokenizer.token_to_id('<pad>')
        # Beginning of Sequence token ID
        self.bos_id = self.tokenizer.token_to_id('<bos>')
        # End of Sequence token ID
        self.eos_id = self.tokenizer.token_to_id('<eos>')
        # Ensuring that all the special tokens are there in the tokenizer's vocabulary
        assert self.pad_id is not None
        assert self.bos_id is not None
        assert self.eos_id is not None

    def __len__(self) -> int:
        '''
        The result of this function is returned when len(TranslationDataset) instance is called
        Args:
            self: 
                Object of the class
        Returns:
            int: Length of the input Dataset
        '''

        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, torch.tensor]:
        '''
        This function is called internally when we try to index on a TranslationDataset instance
        for accessing samples by their indices.
        1. The function accesses Dataset samples by indices
        2. Tokenizes the source (de) and target (en) text
        3. Creates and returns encoder_input, decoder_input, and label

        Args:
            self: 
                Object of the class
            index: int
                sample index
        Returns:
            dict[str, torch.Tensor]
                Dict of encoder_input, decoder_input, and label with their long(dtype) tensors

        '''
        # Retrieving each dataset sample by its index
        sample = self.dataset[index]
        # Getting the source and target text
        source_text = sample['de']
        target_text = sample['en']
        # Tokenizing the source and the target text
        # This returns Hugging Face tokenizers object
        source_enc = self.tokenizer.encode(source_text)
        target_enc = self.tokenizer.encode(target_text)
        # Getting the token IDs of the source and target encodings from Hugging Face tokenizers object
        # Here we hard lock the source and target encodings to match the source and target sequence lengths
        # i.e. The encoder_input tensor will only have token IDs for 0 to SRC_SEQ_LEN - 1 th token
        # Similarly the decoder_input tensor will only have token IDs for 0 to TGT_SEQ_LEN - 1 th token.
        # This is because the model won't have positional encodings for positions above the source and target context lengths
        # However the seq lengths currently being used account for about 98-99% of the dataset, thus causing truncation only in extreme rare outliers
        source_id = source_enc.ids[: SRC_SEQ_LEN - 1]
        target_id = target_enc.ids[: TGT_SEQ_LEN - 1]
        # Creating the encoder input list
        # Encoder receives the input embeddings and creates rich contextual representations
        # No autoregressive prediction like the decoder
        # Therefore encoder only needs an EOS token
        encoder_input = [*source_id, self.eos_id]
        # Decoder input
        # The decoder gets the encoder output from cross attention, and shifted sequence as inputs
        # Decoder performs autoregressive predictions
        # Therefore it needs start and end of sequence tokens to know when to start and stop predicting tokens
        # Here the preceeding * indicates that all of the IDs (iterable list) should be put here as a whole list rather than nesting
        decoder_input = [self.bos_id, *target_id]
        label = [*target_id, self.eos_id]

        # Dictionary of encoder input, decoder input and decoder output
        return {'encoder_input': torch.tensor(encoder_input, dtype = torch.long),
                'decoder_input': torch.tensor(decoder_input, dtype = torch.long),
                'label': torch.tensor(label, dtype = torch.long)}