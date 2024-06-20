'''
To use a recurrent neural network (RNN) to make predictions on a time sequence in PyTorch, you can follow these steps:

1.Define the RNN layer using nn.RNN. You can specify the input size, hidden size, 
and number of layers of the RNN as well as other parameters such as the nonlinearity and cell type.

2.Define a loss function and an optimizer. You can use a suitable loss function such as mean squared error (MSE) or cross-entropy loss,
and an optimizer such as stochastic gradient descent (SGD) or Adam to update the weights of the RNN during training.

3.Split the time sequence into input and target sequences. 
For example, you can use the first n-1 time steps 
as the input sequences and the last time step as the target sequence.

4.Iterate over the input and target sequences in a loop. 
At each iteration, pass the input sequence through the RNN to get the output 
and compare the output with the target sequence using the loss function. 
Use the optimizer to update the weights of the RNN.

5.After the loop, the RNN should be trained on the time sequence. 
You can then use the trained RNN to make predictions on new, unseen time sequences 
by passing them through the RNN and using the output as the prediction.
'''

import torch
import torch.nn as nn

# Define the RNN layer
dim = 1
num_layers = 1
rnn = nn.RNN(input_size=1, hidden_size=dim, num_layers=num_layers, nonlinearity='relu') # tanh

# Define a loss function and an optimizer
loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(rnn.parameters(), lr=0.01) # Adam

# Split the time sequence into input and target sequences
time_sequence = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])

flag = 0
epoch = 10
for eq in range(epoch):
    print('epoch:'+str(eq))
    a = torch.randn(1)
    input_sequences = time_sequence[:-1]*a  # first n-1 time steps
    target_sequence = time_sequence[1:]*a  # last n-1 time steps
    for input_seq, target_seq in zip(input_sequences, target_sequence):
        # Reset the hidden state of the RNN
        hidden = torch.zeros(num_layers, 1, dim)  # (num_layers, batch_size, hidden_size)

        # Forward pass
        input_seq_pand=input_seq.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        output, hidden = rnn(input_seq_pand, hidden)

        # Compute the loss
        loss = loss_fn(output, target_seq.unsqueeze(0))

        # Zero the gradients
        optimizer.zero_grad()

        # Backward pass
        loss.backward()

        # Update the weights
        optimizer.step()

        print(input_seq_pand)
        print(target_seq)
        # print(flag)
        # flag = flag + 1

# Make predictions on new, unseen time sequences
new_time_sequence = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]) 
input_sequences = new_time_sequence[:-1]
predictions = []

# Iterate over the input sequences
for input_seq in input_sequences:
    # Reset the hidden state of the RNN
    hidden = torch.zeros(num_layers, 1, dim)  # (num_layers, batch_size, hidden_size)

    # Forward pass
    input_seq_pand = input_seq.unsqueeze(0).unsqueeze(0).unsqueeze(0)
    output, hidden = rnn(input_seq_pand, hidden)

    # Get the prediction
    prediction = output.squeeze()

    # Add the prediction to the list
    predictions.append(prediction)

# Print the predictions
print(predictions)