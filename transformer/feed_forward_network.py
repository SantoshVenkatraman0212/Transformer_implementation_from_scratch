'''
This file handles the implementation of Feed Forward Network (FFN), which is a Multi Layer Perceptron (MLP)
that performs the following functionalities:
1. Higher dimensional richer representation of individual tokens
2. Projecting the learned features back to token embeddings' own dimensions (d_model)
'''
# Import necessary libraries
import torch
import torch.nn as nn

class FeedForward(nn.Module):
    '''
    This class implements a fully connected feedforward neural network consisting of 2 layers:
    1. Layer-1: Projects the input tensor (Combined output from the multi-head attention block) to 4x higher dimension
    2. Layer-2: Maps the high dimensional learned representation back into token embeddings' own dimensions (d_model)

    The main purpose of this module network is convert each token having contextual representation of every other
    token in sequence during attention to richer high dimensional representation individually. 
    '''
    def __init__(self, d_model: int, dropout: float):
        super().__init__()
        # Dropout layer
        self.dropout = nn.Dropout(dropout)
        # The richer 4x dimensional projection layer for multi-head attention output tensor
        self.ffn_1 = nn.Linear(d_model, 4 * d_model) # The paper attention is all you need explicitly does 512 -> 2048 expansion
        # ReLU introduces non-linearity in the network, and clips negative values
        self.activation = nn.ReLU()
        # The down projection layer that maps the expanded tensor back to token's embedding dim (d_model)
        # This layer makes the individual contextual token representation even richer
        # There is no ReLU here, as clipping -ve values from this tensor, destroys many meaningful -ve float representations
        self.ffn_2 = nn.Linear(4 * d_model, d_model)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        The forward function handles the forward pass through the feed-forward network
        1. Input
        2. Higher dim projection (4 x d_model)
        3. Non-Linearity(ReLU)
        4. Dropout (regularization)
        5. Mapping learned features back to original token embedding dims (d_model)
        '''
        x = self.ffn_1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.ffn_2(x)

        return x