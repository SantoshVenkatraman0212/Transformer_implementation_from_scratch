'''
The main objective of collate.py is to iterate through each batch of tokenized 
data and pad the sequences to match the max seq len in that batch
for stacking the tensors
'''
# Import necessary libraries
import torch
from torch.nn.utils.rnn import pad_sequence

def collate_batches(batch: list[dict[str, torch.Tensor]], pad_id: int) -> dict[str, torch.Tensor]:
    '''
    This function iterates through each batch of encoder_input, decoder_input, and label,
    appends pad tokens to the end of the sequences for all sequences to match the max sequence length for that batch

    Args:
        batch: list[dict[str, torch.Tensor]]
            A list / batch of sequences
        pad_id: int
            ID of the pad token

    Returns:
        dict[str, torch.Tensor]
            Dict of batched encoder_input, decoder_input, and label
    '''
    # Getting the tensors
    encoder_input = [sample['encoder_input'] for sample in batch]
    decoder_input = [sample['decoder_input'] for sample in batch]
    label = [sample['label'] for sample in batch]

    # Padding the tensors
    encoder_input = pad_sequence(sequences = encoder_input, batch_first = True, padding_value = pad_id)
    decoder_input = pad_sequence(sequences = decoder_input, batch_first = True, padding_value = pad_id)
    label = pad_sequence(sequences = label, batch_first = True, padding_value = pad_id)

    # This function returns a dict of encoder_input, decoder_input, and label batched tensors
    # For instance, for each batch encoder_input will be a tensor of multiple padded tokenized sequences
    return {'encoder_input': encoder_input, 'decoder_input': decoder_input, 'label': label}
    