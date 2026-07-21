'''
This file demonstrates Layer Normalization implementation in the transformer architecture
In addition, it also provides methods to address training instability and math error by using scaling factors
'''
# Importing necessary libraries
import torch
import torch.nn as nn

class LayerNorm(nn.Module):
    '''
    This class defines all of the 3 scaling factors for layer norm, and normalizes
    the input tensor, to have uniform token embeddings ranges, for training stability downstream
    1. epsilon: This additive factor prevents zero divide error for the tensors where mean, and std = 0
    2. beta: This additive factor helps preserve richness of the contextual representation after normalization
    3. gamma: This multiplicative factor also helps in maintaining the scale of the embeddings tensor, and for preserving embedding representation richness

    Note: Every embedding dim should have it's own gamma, and beta and it's token independent
    '''
    def __init__(self, d_model: int):
        super().__init__()
        # Default epsilon value used in Attention is all you need
        self.epsilon = 1.0e-5 # Industry standard scaling factors(epsilon) range from 10^-6 - 10^-5
        # This is learnable scaling factor and that's why it is nn.Parameter; also initializing as 0 (additive)
        self.beta = nn.Parameter(torch.zeros(d_model))
        # This learnable param is initialized as 1, as it's multiplicative, and 0 would make just the beta to be present in the output
        self.gamma = nn.Parameter(torch.ones(d_model))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        This function performs stable normalization of the input.
        The order of operation is as follows:
        1. Mean of each token across it's embedding dim
        2. Variance of each token across it's embedding dim
        3. Subtracting the mean from each value in the token embedding vector; dividing by sqrt(variance + epsilon)
        4. Multiplying the normalized tensor with gamma, and adding beta for preserving feature scales and offsets due to normalization
        '''
        # Computing mean across the embedding dimension
        # Here dim = -1, means the mean and var are being computed across d_model
        # keepdim = True is for tensor broadcasting, as taking mean and var changes x.shape (batch_size, seq_len, d_model) -> (batch_size, seq_len,)
        # So keepdim ensures that the shape is (batch_size, seq_len, 1) so that tensor broadcasting happens automatically
        mean = x.mean(dim = -1, keepdim = True)
        var = x.var(dim = -1, keepdim = True, unbiased = False)
        x = (x - mean) / torch.sqrt(var + self.epsilon)
        x = (self.gamma * x) + self.beta

        return x
