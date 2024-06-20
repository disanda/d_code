## How to build multi-head attention using Pytorch?
# A transformer is a deep learning model that is designed to process sequential input data using self-attention mechanisms. 
# It consists of an encoder and a decoder, both of which are composed of multiple layers of self-attention and feedforward neural networks.
# Here is an example of how you can implement a transformer using PyTorch:

import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, input_dim, output_dim):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        self.query_projections = nn.ModuleList([nn.Linear(input_dim, output_dim) for _ in range(num_heads)])
        self.key_projections = nn.ModuleList([nn.Linear(input_dim, output_dim) for _ in range(num_heads)])
        self.value_projections = nn.ModuleList([nn.Linear(input_dim, output_dim) for _ in range(num_heads)])
        self.output_projection = nn.Linear(num_heads * output_dim, output_dim)

    def forward(self, query, key, value, mask=None):
        outputs = []
        for i in range(self.num_heads):
            query_projection = self.query_projections[i](query)
            key_projection = self.key_projections[i](key)
            value_projection = self.value_projections[i](value)
            
            dot_product = torch.matmul(query_projection, key_projection.transpose(1, 2))
            if mask is not None:
                dot_product = dot_product.masked_fill(mask == 0, -1e9)
            attention_weights = torch.softmax(dot_product, dim=-1)
            output = torch.matmul(attention_weights, value_projection)
            outputs.append(output)
            
        concatenated_outputs = torch.cat(outputs, dim=-1)
        final_output = self.output_projection(concatenated_outputs)
        return final_output

class TransformerLayer(nn.Module):
    def __init__(self, hidden_size, num_heads, dropout):
        super(TransformerLayer, self).__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.dropout = dropout

        # Multi-head attention module
        self.attention = MultiHeadAttention(num_heads=num_heads, input_dim=hidden_size, output_dim=hidden_size)
        # Dropout and residual connection after the attention module
        self.attention_dropout = nn.Dropout(dropout)
        self.attention_norm = nn.LayerNorm(hidden_size)

        # Feedforward neural network
        self.feedforward = nn.Sequential(
            nn.Linear(hidden_size, 4 * hidden_size),
            nn.ReLU(),
            nn.Linear(4 * hidden_size, hidden_size)
        )
        # Dropout and residual connection after the feedforward network
        self.feedforward_dropout = nn.Dropout(dropout)
        self.feedforward_norm = nn.LayerNorm(hidden_size)

    def forward(self, input, mask):
        # Multi-head attention
        attention_output = self.attention(input, input, input, mask)
        attention_output = self.attention_dropout(attention_output)
        # Add the residual connection and apply layer normalization
        attention_output = self.attention_norm(input + attention_output)

        # Feedforward neural network
        feedforward_output = self.feedforward(attention_output)
        feedforward_output = self.feedforward_dropout(feedforward_output)
        # Add the residual connection and apply layer normalization
        feedforward_output = self.feedforward_norm(attention_output + feedforward_output)

        return feedforward_output

class Transformer(nn.Module):
    def __init__(self, num_layers=6, hidden_size=512, num_heads=8, dropout=0.1):
        super(Transformer, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.dropout = dropout

        # Encoder layers
        self.encoder_layers = nn.ModuleList([TransformerLayer(hidden_size, num_heads, dropout) for _ in range(num_layers)])
        self.encoder_norm = nn.LayerNorm(hidden_size)

        # Decoder layers
        self.decoder_layers = nn.ModuleList([TransformerLayer(hidden_size, num_heads, dropout) for _ in range(num_layers)])
        self.decoder_norm = nn.LayerNorm(hidden_size)

        # Output projection layer
        self.output_projection = nn.Linear(hidden_size, hidden_size)

    def forward(self, input, mask):
        # Pass the input through the encoder layers
        for layer in self.encoder_layers:
            input = layer(input, mask)

        # Apply the encoder layer normalization
        input = self.encoder_norm(input)

        # Pass the encoded input through the decoder layers
        for layer in self.decoder_layers:
            input = layer(input, mask)

        # Apply the decoder layer normalization
        input = self.decoder_norm(input)

        # Apply the output projection layer
        output = self.output_projection(input)

        return output

# To use this transformer model, you can pass it the input data and a mask indicating 
# which positions in the input should be ignored as input, and it will return the output of the transformer.
# For example:

# Define the transformer model
transformer = Transformer(num_layers=6, hidden_size=512, num_heads=8, dropout=0.1)

# Input data and mask
batch_size = 10
sequence_length = 16
hidden_size = 512
input = torch.randn(batch_size, sequence_length, hidden_size)
mask = torch.rand(batch_size, sequence_length, sequence_length) > 0.5

# Compute the transformer output
output = transformer(input, mask)



