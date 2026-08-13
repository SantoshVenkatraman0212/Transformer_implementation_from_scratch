'''
This file implements TranslationLoss class for computing cross entropy loss on the decoder's autoregressive predictions
'''
# Importing necessary libraries
import torch 
import torch.nn as nn

class TranslationLoss(nn.Module):
    '''
    This class computes cross entropy loss between decoder predicted tokens, and true labels
    '''
    def __init__(self, pad_id: int) -> None:
        super().__init__()
        self.loss = nn.CrossEntropyLoss(ignore_index = pad_id)

    def forward(self, logits: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        '''
        The forward function does the following:
        1. Gets the dimensions of logits, and labels
        2. Reshapes logits and labels by flattening their batch, and sequence length dimensions
        3. Computes and returns cross entropy loss

        Args:
            self:
                object of the class
            logits: torch.Tensor
                Final forward pass output of the transformer
            label: torch.Tensor
                True labels for the tokens
        '''
        # Logits is of shape (batch_size, seq_len, vocab_size)
        batch_size, seq_len, vocab_size = logits.size()
        # Flattening logits shape to (batch_size * seq_len, vocab_size)
        logits = logits.reshape(batch_size * seq_len, vocab_size)
        # Flattening labels shape to (batch_size * seq_len)
        label = label.reshape(batch_size * seq_len)
        # Computing cross entropy loss
        loss = self.loss(logits, label)

        return loss