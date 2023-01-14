'''
In PyTorch, nn.RNN is a class that represents a recurrent neural network (RNN) layer. 
RNNs are a type of neural network that are designed to process sequential data, such as time series or natural language. 
They are able to "remember" information from previous time steps and use it to process the current time step, 
which makes them well-suited for tasks such as language translation or language modeling.
'''

import torch
import torch.nn as nn

num_layers = 2 # hidden tensor (num_layers, batch_size, hidden_size)
seq_len = 10 # sequence  tensor  (seq_len, batch_size, input_size)
input_size = 300 
batch_size = 32
hidden_size = 128

# print('------------- Define the RNN Cell --------------------')
# rnn_cell = nn.RNNCell(input_size, hidden_size, bias=True, nonlinearity='tanh', device=None, dtype=None)

# # Print the weights of cell // y = wx + b
# print(rnn_cell.weight_ih.shape)  # (128, 300) input2hidden  
# print(rnn_cell.weight_hh.shape)  # (128, 128) hidden2hidden
# print(rnn_cell.bias_ih.shape)  # (128,)
# print(rnn_cell.bias_hh.shape)  # (128,)

# print('---')

# # Initialize the hidden state of the RNN
# input_sequences = torch.randn(seq_len, batch_size, input_size) # (10, 5, 12)
# hidden = torch.zeros(batch_size, hidden_size)  

# # Forward pass
# f = 0
# outputs = 0
# for i in input_sequences:
#     h_n = rnn_cell(i, hidden) # input:(batch_size, input_size) output: h_n
#     if f == 0:
#         outputs = h_n.unsqueeze(0)
#         f = 1
#     else:
#         outputs = torch.cat((outputs,h_n.unsqueeze(0)))
#     hidden = h_n
#     print(h_n.shape)
#     print(outputs.shape)  # output.shape torch.Size([10, 32, 128]) / (seq_len, batch_size, hidden_size) 

# print('----------- RNN layers-----------')
# rnn = nn.RNN(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers)

# # # Print the weights of the first layer
# # print(rnn.weight_ih_l0.shape)  # (300, 128) input2hidden
# # print(rnn.weight_hh_l0.shape)  # (128, 128) hidden2hidden
# # print(rnn.bias_ih_l0.shape)  # (128,)
# # print(rnn.bias_hh_l0.shape)  # (128,)

# # # Print the weights of the second layer
# # print(rnn.weight_ih_l1.shape)  # (128, 128)
# # print(rnn.weight_hh_l1.shape)  # (128, 128)
# # print(rnn.bias_ih_l1.shape)  # (128,)
# # print(rnn.bias_hh_l1.shape)  # (128,)

# # Each sequence is a tensor of shape (seq_len, batch_size, input_size)
# input_sequences = torch.randn(seq_len, batch_size, input_size)

# # Initialize the hidden state of the RNN
# h_0 = torch.zeros(num_layers, batch_size, hidden_size)  #  h0 (num_layers, batch_size, hidden_size)

# # Forward pass
# outputs, h_n = rnn(input_sequences, h_0) # output h_n
# print(outputs.shape)  # output.shape torch.Size([10, 32, 128]) / (seq_len, batch_size, hidden_size) 
# print(h_n.shape)  # hidden.shape torch.Size([2, 32, 128]) /(num_layers, batch_size, hidden_size)
# print(outputs[-1] == h_n[-1])


# print('------------------ Define the GRU cell ------------------------')
# gru_cell = nn.GRUCell(input_size, hidden_size)

# '''Note that the GRU cell has 3 sets of weights and biases: 
# one for the Update Gate, (previous information (prior time steps) that needs to be passed along the next state.)
# one for the Reset Gate (the past information is needed to neglect), 
# one for the Hidden State, (RNN)
# '''

# # Print the weights of cell
# print(gru_cell.state_dict().keys())  # dict_keys(['weight_ih', 'weight_hh', 'bias_ih', 'bias_hh'])
# print(gru_cell.weight_ih.shape)  # (3 * hidden_size, 300) input2hidden
# print(gru_cell.weight_hh.shape)  # (3 * hidden_size, 128) hidden2hidden
# print(gru_cell.bias_ih.shape)  # (3 * hidden_size,)
# print(gru_cell.bias_hh.shape)  # (3 * hidden_size,)

# input_sequence = torch.randn(seq_len, batch_size, input_size)
# hidden = torch.randn(batch_size, hidden_size)
# output = []
# for i in range(6):
#     hx = gru_cell(input_sequence[i], hidden)
#     output.append(hx)
#     print(hx.shape)

# print('------------------ Define the GRU layers ------------------------')
# gru = nn.GRU(input_size, hidden_size, num_layers)

# hidden = torch.randn(num_layers, batch_size, hidden_size)
# output, hidden = gru(input_sequence,hidden)

print('------------------- Define the LSTM cell -----------------------')
lstm_cell = nn.LSTMCell(input_size, hidden_size)

'''Note that the LSTM cell has four sets of weights and biases: 
one for the input gate, (4 * hidden_size, input_size)
one for the forget gate, (4 * hidden_size, input_size)
one for the output gate, (4 * hidden_size,)
one for the hidden state.  (4 * hidden_size,)
'''

# Print the weights of the LSTM cell
print(lstm_cell.weight_ih.shape)
print(lstm_cell.weight_hh.shape)
print(lstm_cell.bias_ih.shape)
print(lstm_cell.bias_hh.shape)

input_sequence = torch.randn(batch_size, input_size)
hidden = torch.zeros(batch_size, hidden_size)
cell = torch.zeros(batch_size, hidden_size)

# #Pass the input through the LSTM model to get the output and the updated hidden and cell states
hidden, cell = lstm_cell(input_sequence, (hidden, cell))

print('------------------- Define the LSTM layers -----------------------')

lstm = nn.LSTM(input_size, hidden_size, num_layers)

input_sequence = torch.randn(seq_len, batch_size, input_size)
hidden = torch.zeros(num_layers, batch_size, hidden_size)
cell = torch.zeros(num_layers, batch_size, hidden_size)

outputs, (hidden, cell) = lstm(input_sequence, (hidden, cell))

print(outputs.shape)  # output.shape torch.Size([10, 32, 128]) / (seq_len, batch_size, hidden_size) 
print(hidden.shape)  # hidden.shape torch.Size([2, 32, 128]) / (num_layers, batch_size, hidden_size)
print(cell.shape)