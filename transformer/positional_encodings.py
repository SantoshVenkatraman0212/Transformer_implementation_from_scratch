'''
This file creates sinusoidal positional encodings for all the positions and for each embedding dim
'''
# Importing necessary libraries
import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    '''
    This class generates fixed positional encodings for each of the possible positions in a sequence.
    Input shape: (batch_size, seq_len, embed_dim)
    Output shape: (batch_size, seq_len, embed_dim)
    '''
    def __init__(self, max_seq_len: int, d_model: int, dropout: float):
        '''
        The constructor creates positional encodings for the max sequence length and registers them in a buffer
        Args:
            max_seq_len: No of tokens in the largest sequence in the input text
            d_model: embedding dimension
            dropout: dropout probability value to avoid overfitting
        '''
        super().__init__()
        # Dropout for avoiding overfitting
        self.dropout = nn.Dropout(dropout)
        # Creating a placeholder (0s Tensor) for all the positions and embedding dims
        # For each token in max_seq_len, and for each dim in d_model, a pos enc gets added
        # Therfore the shape of the 0s tensor is (max_seq_len, d_model)
        pos_enc = torch.zeros(max_seq_len, d_model)
        # This positions tensor will have int corresponding to every position from 0 to max_seq_len - 1
        # Now by default it's shape will be (max_seq_len, ) 
        # But each position is supposed to correspond to a token embedding vector; every row should have a position int
        # Therefore we add a dimension at the end of this positions tensor to give the shape (max_seq_len, 1)
        pos = torch.arange(max_seq_len).unsqueeze(1)
        # The div term is basically 1 / (10000 ^ ((2 * i) / d_model)))
        # But 1 / math.pow is slower and computationally inefficient compared to log and exp
        # Therefore replacing with e ^ (-log(10000) * ([0, 2, 4......d_model-1] / d_model))
        div_term = torch.exp(-math.log(10000) * (torch.arange(0, d_model, 2, dtype = torch.float32) / d_model))
        # Here the order of traversal is this:
        # First for a single row in pos_enc tensor (single position in the sequence) pos remains the same while the div_term keeps moving
        # This is because, here it's element-wise product of 2 tensors; The div_term is broadcasted across the positions tensor
        # Even embed dims get sin pos enc while odd positions get cos pos enc
        pos_enc[:, :: 2] = torch.sin(pos * div_term)
        pos_enc[:, 1 :: 2] = torch.cos(pos * div_term)

        # Adds batch dim to the positional encoding tensor
        pos_enc = pos_enc.unsqueeze(0)

        self.register_buffer('pe', pos_enc)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        This function takes in (batch_size, seq_len, d_model) input, adds positional encoding and returns the tensor
        '''
        # Here we should add positional encodings only for the length of the current input sequence (positions)
        # So for all batches we add positional encodings for that specific sequence length for all the embedding dims
        # Using size instead of shape is a common PyTorch convention
        x = x + self.pe[:, : x.size(1), :]
        # Dropout for setting some of the values(dropout probability based) randomly to 0
        x = self.dropout(x)

        return x


        
            
                    
            
