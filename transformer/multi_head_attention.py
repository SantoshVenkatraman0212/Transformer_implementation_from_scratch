'''
This file has the code for multi-head attention implementation from scratch
The Multi-head attention implementation includes the following:
1. Projection of Input embeddings into query, key and value tensors
2. Propagation of query, key and values across attention heads
3. Causal masking for autoregression
4. Cross attention for decoder block (Query from Decoder, Key and Value from Encoder)
4. Multi-head attention output computation, concatenation, and projection

This is the core Transformers
'''
# Importing necessary libraries
import math
import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    '''
    This class implements the full multi-head attention mechanism.
    It receives input embeddings (token emb + pos enc), performs self attention (normal and masked) and cross attention, and returns combined output
    '''
    def __init__(self, n_heads: int, d_model: int, dropout: float, mask: bool):
        super().__init__()
        # The embedding dim should be exactly divisible by attention head
        assert d_model % n_heads == 0, "Embedding dimension not divisible by number of attention heads"
        # Craeting members for attention heads and emb dims
        self.n_heads = n_heads
        self.d_model = d_model
        # class member for dropout
        self.dropout = nn.Dropout(dropout)
         # No of embedding dims each attention head will take
        self.d_k = self.d_model // self.n_heads
        # Defining query, key, and value weight tensors
        # These tensors get us the query, key and value representations upon dot product with the input
        # 3 trainable weight tensors for computing query (Q), key (K), and value (V) of dims (d_model, d_model)
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        # Projection layer after combining the multi-head attention outputs
        self.o_proj = nn.Linear(d_model, d_model)
        # causal mask
        # Masking is required for decoder module (the model shouldn't see the future tokens)
        self.mask = mask
    
    def forward(self, x: torch.Tensor, cross_x: torch.Tensor = None, padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        '''
        This function takes input embeddings (encoder, and decoder), and implements self attention
        Args:
            x: Input embeddings (batch_size, seq_len, d_model)
            cross_x: Encoder output embeddings tensor for cross attention in decoder
        
        Returns:
            proj_outputs: Combined final multi-head attention output tensor ((batch_size, seq_len, d_model))
        '''
        # Getting the batch size from the input embeddings
        batch_size, seq_len = x.size(0), x.size(1)
        # Query is forward pass (dot product of x and w_q) 
        query = self.w_q(x) # shape: (batch_size, seq_len, d_model)
        # Propagating Q, K, and V across the multi-attention heads
        # ---------- Self attention (normal and masked) ----------
        # Q, K and V are projected from (batch_size, seq_len, d_model) -> (batch_size, seq_len, n_heads, d_k) i.e. each head sees d_k dims of the tokens
        # From (batch_size, seq_len, n_heads, d_k) they are changed to (batch_size, n_heads, seq_len, n_heads, d_k)
        # This is because for each batch each attention head sees the entire sequence, but a part of the token representation (d_model)
        query = query.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        if cross_x is None: # If it's self attention in encoder or masked self attention in decoder
            # Key is dot product x and w_k
            key = self.w_k(x) # shape: (batch_size, seq_len, d_model)
            # Value is dot product of x and w_v
            value = self.w_v(x) # shape: (batch_size, seq_len, d_model)
            key = key.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
            value = value.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        else: # if it's cross attention in decoder
            # Cross attention doesn't accept causal mask
            assert self.mask is not True, "Masked cross attention is not possible"
            # Ensuring the batch_size is the same for decoder input, and also the encoder output
            assert x.size(0) == cross_x.size(0), "Batch size mismatch"
            # seq_len for the encoder output
            # Note: The sequence length for decoder might not be the same as that of encoder, as the generated seq can be of any length
            cross_x_seq_len = cross_x.size(1)
            # Getting the key, and value from encoder 
            # Here the decoder asks "for whatever I'm trying to generate, which tokens are relevant given encoder's rich contextual representation"
            # This is why decoder does the querying part while the key, and values are obtained from the encoder
            key = self.w_k(cross_x)
            value = self.w_v(cross_x)
            # ---------- Cross attention ----------
            # Q, K and V are projected from (batch_size, cross_x_seq_len, d_model) -> (batch_size, cross_x_seq_len, n_heads, d_k) i.e. each head sees d_k dims of the tokens
            # From (batch_size, cross_x_seq_len, n_heads, d_k) they are changed to (batch_size, n_heads, cross_x_seq_len, n_heads, d_k)
            key = key.view(batch_size, cross_x_seq_len, self.n_heads, self.d_k).transpose(1, 2)
            value = value.view(batch_size, cross_x_seq_len, self.n_heads, self.d_k).transpose(1, 2)

        # Relevance score (attention scores)
        # Shapes of query, key, and values will be (batch_size, n_heads, seq_len, d_k) or (batch_size, n_heads, cross_x_seq_len, d_k)
        # This depends on self or cross attention
        # Therefore for dot product, the shape of key is changed from (batch_size, n_heads, seq_len, d_k) -> (batch_size, n_heads, d_k, seq_len)
        # For cross attention the shape of key is changed from (batch_size, n_heads, cross_x_seq_len, d_k) -> (batch_size, n_heads, d_k, cross_x_seq_len)
        attn_sc = query @ key.transpose(2, 3) # self attention -> Shape: (batch_size, n_heads, seq_len, seq_len) | cross -> Shape: (batch_size, n_heads, seq_len, cross_x_seq_len)
        # Scaling the attention scores
        attn_sc = attn_sc / math.sqrt(self.d_k)

        # Decide whether we require causal masking or no
        # Causal masking makes the attention scores for the successive tokens to -inf
        # This ensures that the model doesn't attend to the new tokens during autoregressive next token prediction
        # This is required for decoder module where the model shouldn't see the attention score for the next tokens it's supposed to predict
        if self.mask:
            # Runs when the mask is True
            # Creating a full 1s tensor of shape (seq_len, seq_len), and moving it to GPU explicitly as it's not auto registered by PyTorch 
            mask_tensor = torch.ones(seq_len, seq_len).to(x.device)
            # Converting the 1s tensor to upper triangular tensor where the upper right diagonal and next values are set to 1
            tri_tensor = torch.triu(mask_tensor, 1)
            # Getting the coords of upper triangle elements, and setting them to -inf for masking
            mask_coords = tri_tensor == 1
            tri_tensor[mask_coords] = -torch.inf
            # Adding the mask to attention scores
            attn_sc = attn_sc + tri_tensor

        # Applying pad token mask to the attn_sc
        # Padding mask will be of dim (batch_size, seq_len)
        # It checks for each sequence in a batch, which tokens are real, and which ones are pad tokens
        if padding_mask is not None:
            # Doing sanity checks on the padding_mask to verify its tensor shapes are compatible
            # Checking if its just a 2-D Tensor
            assert padding_mask.dim() == 2
            # Checking the 0th dim of padding_mask is batch_size, and last dim is seq_len or cross_x_seq_len 
            # This dims have to be compatible with the Attention scores
            assert padding_mask.size(0) == batch_size
            assert padding_mask.size(-1) == attn_sc.size(-1)

            padding_mask = padding_mask.view(batch_size, 1, 1, attn_sc.size(-1))
            # Here wherever padding_mask is True, it means they are real tokens to attend to
            # ~padding_mask gives the coords of the pad tokens i.e. not-real tokens
            # They're convered to -infinity
            attn_sc = attn_sc.masked_fill(~padding_mask, -torch.inf)
        # Computing attention weights
        attn_wt = torch.softmax(attn_sc, dim = -1)
        # Dropout on attenion weight
        attn_wt = self.dropout(attn_wt)
        # Multi-head attention output
        # Current atten_wt shape -> (batch_size, n_heads, seq_len, seq_len)
        mha_output = attn_wt @ value
        # Concatenating outputs of multi-head attention
        # So for us to combine the outputs from multiple heads, which were projected from d_model to n_heads, d_k
        # The mha_output shape should be brought from: 
        # self attention: (batch_size, n_heads, seq_len, d_k) -> (batch_size, seq_len, n_heads, d_k)
        # cross attention: (batch_size, n_heads, cross_x_seq_len, d_k) -> (batch_size, cross_x_seq_len, n_heads, d_k)
        # .contiguous ensures that the output is a contiguous tensor (continuous memory locations)
        attn_output = mha_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        # Final linear projection
        proj_output = self.o_proj(attn_output)

        return proj_output


        

