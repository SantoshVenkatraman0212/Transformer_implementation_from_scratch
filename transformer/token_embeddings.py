'''
This file is aimed at implementing the class for converting token Is to trainable token embeddings 

'''
# Importing necessary libraries
import math
import torch
import torch.nn as nn

# Class for input token embeddings
class TokenEmbeddings(nn.Module):
    '''
    This class converts input token IDs to scaled embedding tensor and returns it
    The class inherits from PyTorch's nn.Module for the following reasons:
        1. nn.Embeddings belongs to nn.Module class, and it has to be inherited 
        2. This makes the components defined by the class to be registered by PyTorch 
        3. It also ensures that the embeddings are trainable as this triggers nn.Parameter that has (requires_grad = True)
    '''
    def __init__(self, vocab_size: int, d_model: int):
        '''
        The constructor determines whenever TokenEmbeddings class instance is created, what all variables or members are also invoked
        Args:
            self: object of the class TokenEnbeddings
            vocab_size: Total no of unique tokens in our transformer vocabulary
            d_model: No of dims / len of embedding vectors associated with the token IDs 
        '''
        super().__init__() # The super constructor ensures that, the base class's constructor i.e. nn.Module's constructor is also invoked
        # Token embeddings here are essentially lookup tables where each row corresponds to a token ID and each col is a dim of the embedding vectors
        self.token_embeddings = nn.Embedding(vocab_size, d_model)
        # Embedding vectors dims
        self.d_model = d_model
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Forward function is a staple for most classes implementing PyTorch logic.
        It's a special function that is called whenever the class is involved in the forward pass logic
        This function defines the control flow for the class
        Here it determines how the token IDs are turned to embeddings
        Args:
            x: Input token IDs
        '''
        # The forward pass of input token IDs through the token embeddings basically creates a token embeddings tensor of (bath_size, vocab_size, d_model) dims
        # Basically whatever the input tensor is, each of the integer IDs will be mapped to token embeddings vectors
        x = self.token_embeddings(x)
        # The embedding tensor is scaled up by square root of d_model, so that the positional encodings don't dominate over token representations
        x = x * math.sqrt(self.d_model)

        return x