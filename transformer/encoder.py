'''
This file depicts building the whole transformer encoder by combining N encoder blocks that we've already created
'''
# Importing necessary libraries
import torch
import torch.nn as nn
from .encoder_block import EncoderBlock

class Encoder(nn.Module):
    '''
    This class builds the full transformer encoder by stacking N encoder blocks sequentially
    '''
    def __init__(self, n_heads: int, n_blocks: int, d_model: int, dropout: float):
        super().__init__()
        # Using nn.ModulesList for creating a list of N encoder blocks, as separate class instances that will be registered by PyTorch
        # Separate instances are required, as the blocks downstream should get receive outputs of the preceeding blocks as input, and not same initial input for all the blocks
        self.encoder_blocks = nn.ModuleList([EncoderBlock(n_heads = n_heads, d_model = d_model, dropout = dropout) for _ in range(n_blocks)])

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        '''
        This function iterates through the entire encoder blocks ModuleList consisting of each encoder block
        At every stage the output will become the input of the next encoder block
        '''
        for encoder_block in self.encoder_blocks:
            x = encoder_block(x, padding_mask = padding_mask)

        return x
