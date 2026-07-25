'''
This file depicts the entire transformer decoder module by stacking N decoder blocks 
'''
# Importing necessary libraries
import torch
import torch.nn as nn
from .decoder_block import DecoderBlock

class Decoder(nn.Module):
    '''
    This decoder class builds the full transformer decoder module by sequential stacking of decoder blocks
    The class uses PyTorch's ModuleList so that separate DecoderBlock instances are created and are registered by PyTorch
    '''
    def __init__(self, n_heads: int, n_blocks: int, d_model: int, dropout: float):
        super().__init__()
        self.decoder_blocks = nn.ModuleList([DecoderBlock(n_heads = n_heads, d_model = d_model, dropout = dropout) for _ in range(n_blocks)])

    def forward(self, x: torch.Tensor, cross_x: torch.Tensor) -> torch.Tensor:
        '''
        The forward function controls flow of embeddings across the decoder blocks by iterating through each of them.
        Here cross_x i.e. encoder output is the same, as the encoder's final output (rich contextual representation) is used by decoder blocks
        However x will have the input for the next decoder block i.e. output from the previous decoder block
        '''
        for block in self.decoder_blocks:
            x = block(x, cross_x)

        return x
