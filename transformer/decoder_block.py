'''
This file depicts full from scratch build and implementation of the decoder block
The decoder block consists of the following modules:
1. Masked Multi-head Attention (MHA / Causal self attention) for autoregressive generation
2. Residual Connection-1 (Input + Masked MHA Output)
3. LayerNorm-1
4. Cross Attention (Q from Decoder, Key, and value from encoder output)
5. Residual Connection-2 (LayerNorm-1 + Cross Attention Output)
6. LayerNorm-2
7. FeedForward 
8. Residual Connection-2 (LayerNorm-2 + FeedForward Output)
9. LayerNorm-3
10. Output
'''
# Importing necessary libraries
import torch
import torch.nn as nn
from .multi_head_attention import MultiHeadAttention
from .layernorm import LayerNorm
from .feed_forward_network import FeedForward


class DecoderBlock(nn.Module):
    '''
    This class implements the entire decoder block responsible to generate output tokens autoregressively
    '''
    def __init__(self, n_heads: int, d_model: int, dropout: float):
        super().__init__()
        # Causal Self Attention 
        self.causal_attention = MultiHeadAttention(n_heads = n_heads, d_model = d_model, dropout = dropout, mask = True)
        # Cross Attention
        self.cross_attention = MultiHeadAttention(n_heads = n_heads, d_model = d_model, dropout = dropout, mask = False)
        # LayerNorm-1
        self.layer_norm_1 = LayerNorm(d_model = d_model)
        # LayerNorm-2
        self.layer_norm_2 = LayerNorm(d_model = d_model)
        # LayerNorm-2 
        self.layer_norm_3 = LayerNorm(d_model = d_model)
        # FeedForward
        self.feed_forward = FeedForward(d_model = d_model, dropout = dropout)

    def forward(self, x: torch.Tensor, cross_x: torch.Tensor, src_padding_mask: torch.Tensor | None = None, 
                tgt_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        '''
        The forward function orchestrates the flow of input from masked self attention to final output
        '''
        # Input residual
        residual = x
        # Masked self attention ensures the decoder block doesn't attend to the future tokens
        # Since it predicts the target, it gets target padding mask
        x = self.causal_attention(x, padding_mask = tgt_padding_mask)
        # Residual Connection-1 (Input residual + Causal attention)
        x = x + residual
        # LayerNorm-1
        x = self.layer_norm_1(x)
        # LayerNorm-1 residual
        residual = x
        # Cross Attention (x: Decoder input, cross_x: Encoder output)
        # Here it's K, and V are from encoder; Therefore it gets source padding mask
        x = self.cross_attention(x, cross_x, padding_mask = src_padding_mask)
        # Residual Connection-2 (LayerNorm-1 residual + Cross Attention output)
        x = x + residual
        # LayerNorm-2
        x = self.layer_norm_2(x)
        # LayerNorm-2 residual
        residual = x
        # FeedForward
        x = self.feed_forward(x)
        # Residual Connection-3 (LayerNorm-2 residual + FeedForward Output)
        x = x + residual
        # LayerNorm-3 
        x = self.layer_norm_3(x)

        return x
