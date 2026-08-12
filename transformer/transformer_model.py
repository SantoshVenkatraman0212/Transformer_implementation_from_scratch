'''
This file builds the entire transformer architecture combining all the modules created in this order:
1. Input -> Token embeddings
2. Positional Encoding
3. Encoder
4. Decoder
5. Final projection layer
6. Output
'''
# Importing necessary libraries
import torch
import torch.nn as nn
from .token_embeddings import TokenEmbeddings
from .positional_encodings import PositionalEncoding
from .encoder import Encoder
from .decoder import Decoder

class Transformer(nn.Module):
    '''
    This class builds the entire transformer model by combining all the modules right from token embeddings to decoder, projection and final token prediction layer
    '''
    def __init__(self, src_vocab_size: int, tgt_vocab_size: int, d_model: int, n_heads: int,  n_blocks: int, 
                 src_max_seq_len: int, tgt_max_seq_len: int, dropout: float):
        super().__init__()
        # Source Token embeddings a look up table of shape (src_vocab_size, d_model)
        self.src_token_embeddings = TokenEmbeddings(vocab_size = src_vocab_size, d_model = d_model)
        # Target Token embeddings a look up table of shape (tgt_vocab_size, d_model)
        self.tgt_token_embeddings = TokenEmbeddings(vocab_size = tgt_vocab_size, d_model = d_model)
        # Source Positional encoding for all possible positions in the current sequence (src_vocab_size, d_model)
        self.src_positional_encodings = PositionalEncoding(max_seq_len = src_max_seq_len, d_model = d_model, dropout = dropout)
        # Target Positional encoding for all possible positions in the current sequence (tgt_vocab_size, d_model)
        self.tgt_positional_encodings = PositionalEncoding(max_seq_len = tgt_max_seq_len, d_model = d_model, dropout = dropout)
        # Encoder (embeddings -> rich contextual representations)
        self.encoder = Encoder(n_heads = n_heads, n_blocks = n_blocks, d_model = d_model, dropout = dropout)
        # Decoder (rich contextual representation -> token prediction)
        self.decoder = Decoder(n_heads = n_heads, n_blocks = n_blocks, d_model = d_model, dropout = dropout)
        # Final Linear projection layer before softmax of shape: (d_model, tgt_vocab_size)
        self.proj = nn.Linear(d_model, tgt_vocab_size)

    def forward(self, src_x: torch.Tensor, tgt_x: torch.Tensor, src_padding_mask: torch.Tensor | None = None, 
                tgt_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        '''
        This forward function controls the flow of input token IDs tensor through the entire transformer network
        '''
        # From Input token IDs (batch_size, src_seq_len) to (batch_size, src_seq_len, d_model)
        src_x = self.src_token_embeddings(src_x)
        # Pos enc added to input token embeddings (batch_size, src_seq_len, d_model)
        src_x = self.src_positional_encodings(src_x)
        # Target token embeddings (batch_size, seq_len) to (batch_size, tgt_seq_len, d_model)
        tgt_x = self.tgt_token_embeddings(tgt_x)
        # Target positional encodings (batch_size, tgt_seq_len, d_model)
        tgt_x = self.tgt_positional_encodings(tgt_x)
        # Input token embeddings to rich contextual representations (batch_size, src_seq_len, d_model)
        src_x = self.encoder(src_x, padding_mask = src_padding_mask)
        # Rich contextual representations to target side contextual representation (decoder sequence) (batch_size, tgt_seq_len, d_model)
        x = self.decoder(tgt_x, src_x, src_padding_mask = src_padding_mask, tgt_padding_mask = tgt_padding_mask)
        # Linear projection layer to match vocabulary size (batch_size, decoder_seq_len, d_model) -> (batch_size, tgt_seq_len, vocab_size) 
        x = self.proj(x)
        
        return x

