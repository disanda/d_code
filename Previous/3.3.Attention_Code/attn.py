##How to build attention using Pytorch
#In PyTorch, you can build an attention mechanism by using the dot or cosine similarity functions to compute the attention weights, 
#and then applying those weights to the input to obtain the attended output.
#Here is an example of how you can implement attention using PyTorch:

import torch
import torch.nn as nn

class Attention(nn.Module):
    def __init__(self, attention_type='dot', hidden_size=256):
        super(Attention, self).__init__()
        self.attention_type = attention_type
        self.hidden_size = hidden_size

        # Linear layer to transform the query (decoder hidden state)
        self.query = nn.Linear(hidden_size, hidden_size, bias=False)
        # Linear layer to transform the key (encoder hidden state)
        self.key = nn.Linear(hidden_size, hidden_size, bias=False)
        # Linear layer to transform the value (encoder hidden state)
        self.value = nn.Linear(hidden_size, hidden_size, bias=False)

        # Softmax layer to compute the attention weights
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, query, keys, values):
        # Transform the query
        query = self.query(query).unsqueeze(1)
        # Transform the keys
        keys = self.key(keys)
        # Transform the values
        values = self.value(values)

        # Compute the attention weights
        if self.attention_type == 'dot':
            # dot product attention
            attention_weights = torch.bmm(query, keys.transpose(1, 2))
        elif self.attention_type == 'cosine':
            # cosine similarity attention
            query = query / query.norm(dim=-1, keepdim=True)
            keys = keys / keys.norm(dim=-1, keepdim=True)
            attention_weights = torch.bmm(query, keys.transpose(1, 2))
        else:
            raise ValueError(f"Invalid attention type: {self.attention_type}")

        # Normalize the attention weights
        attention_weights = self.softmax(attention_weights)

        # Apply the attention weights to the values to obtain the attended output
        attended_output = torch.bmm(attention_weights, values)

        return attended_output, attention_weights

#To use this attention module, you can pass it the query (decoder hidden state), keys (encoder hidden states), and values (encoder hidden states) 
#as input, and it will return the attended output and the attention weights.
#For example:

# Define the attention module
attention = Attention(attention_type='dot', hidden_size=256)

# Inputs to the attention module
batch_size = 10
hidden_size = 256
sequence_length = 12
query = torch.randn(batch_size, hidden_size)
keys = torch.randn(batch_size, sequence_length, hidden_size)
values = torch.randn(batch_size, sequence_length, hidden_size)

# Compute the attended output and attention weights
attended_output, attention_weights = attention(query, keys, values)
print(attended_output.shape)
print(attention_weights.shape)