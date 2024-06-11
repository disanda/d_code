
import torch.nn as nn
class Normal_RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1, model_type='RNN'):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        if model_type == 'RNN':
            self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True) # batch_first=True, nonlinearity = 'relu'
        elif model_type == 'GRU':
            self.rnn = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        #self.dropout = nn.Dropout(p=0.2) 效果变差

    def forward(self, x, h0):
        out, hn = self.rnn(x, h0)
        #out = out[:, -1, :]
        #out = self.dropout(out)
        out = self.fc(out[:, -1, :])
        #out = torch.tanh(out)
        return out, hn