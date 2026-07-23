'''
This file depicts a full from scratch implementation of an encoder block (adhering to 2017 Attention is all you need paper)
The encoder block consists of the following
1. Input (embeddings)
2. Multi-head Attention block
3. Residual connection - 1 (Input + Multi-head Attention output)
4. Layer Norm-1
5. Feed Forward Network
6. Residual connection - 2 (Stage-4 output + Stage-5 output)
7. Layer Norm-2
8. Output 
'''
# Importing necessary libraries
import torch
import torch.nn as nn
# Importing the classes defined for the encoder block (Multi-head attention, layernorm, and feed forward)
from .multi_head_attention import MultiHeadAttention
from .layernorm import LayerNorm
from .feed_forward_network import FeedForward

class EncoderBlock(nn.Module):
    '''
    This class implements an encoder block whose task is to encode input tokens into a rich (multi-dim) contextual representation
    1. The encoder block begins with multi-head attention (MHA) that converts embeddings into rich contextual representations
    2. Then the input is added to the output of MHA (residual connection) to account for the required correction to prevent model performance degradation
    3. Then Layer norm normalizes this output to stabilize downstream processing, and provides embedding scale stability during feed forward
    4. Then we have feed forward network, that enables each token to transform it's features (contextual representation after attention) to give a much richer self representation 
    5. Then we have another layer norm that takes feed forward output + residual from feed forward input
    6. Final output of the encoder block
    '''
    def __init__(self, n_heads: int, d_model: int, dropout: float):
        super().__init__()
        # Multi-head self attention (mask is False as encoder doesn't require causal mask)
        self.attention = MultiHeadAttention(n_heads = n_heads, d_model = d_model, dropout = dropout, mask = False)
        # LayerNorm-1
        self.layer_norm_1 = LayerNorm(d_model = d_model)
        # LayerNorm-2
        self.layer_norm_2 = LayerNorm(d_model)
        # FeedForward
        self.feed_forward = FeedForward(d_model = d_model, dropout = dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        This function orchestrates the flow of tensor through the entire encoder block
        '''
        # Input residual
        residual = x
        # Self attention
        x = self.attention(x)
        # Input + MHA output
        x = x + residual
        # LayerNorm-1
        x = self.layer_norm_1(x)
        # Residual layer norm output
        residual = x
        # FeedForward Network
        x = self.feed_forward(x)
        # FeedForward Output + LayerNorm-1 residual
        x = x + residual
        # LayerNorm-2
        x = self.layer_norm_2(x)

        return x

