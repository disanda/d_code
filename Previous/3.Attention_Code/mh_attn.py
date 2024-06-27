## How to build multi-head attention using Pytorch?
# Multi-head attention is an extension of the attention mechanism that 
# allows the model to attend to multiple different parts of the input simultaneously. 
# It does this by using multiple attention heads, each of which attends to a different part of the input and produces its own attended output. 
# These attended outputs are then concatenated and transformed to obtain the final attended output.
# Here is an example of how you can implement multi-head attention using PyTorch:


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

# Define the multi-head attention module
attention = MultiHeadAttention(num_heads=8, input_dim=512, output_dim=64)

# Define the input tensors
query = torch.randn(32, 16, 512)
key = torch.randn(32, 16, 512)
value = torch.randn(32, 16, 512)
mask = torch.zeros(32, 16, 16)

# Apply the attention module to the input tensors
output = attention(query, key, value, mask=mask)