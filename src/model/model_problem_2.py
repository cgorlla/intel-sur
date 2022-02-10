import torch
import torch.nn as nn

print(nn.Module)
x = torch.tensor([
    [[1,2,3],
     [4,5,6],
     [7,8,9],
     [10,11,12]],
    [[-1,-2,-3],
     [-4,-5,-6],
     [-7,-8,-9],
     [-10,-11,-12]]
])
print(x.shape)

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0):
        super(RNN, self).__init__()
        
        # defining instance variables
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.dropout = dropout
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, dropout, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        # x: [batch_size, sequence_length, input_size]
        
        # initial hidden state; h0: [num_layers, batch_size, hidden_size]
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size.to(device))
        
        # forward pass
        out, h0 = self.rnn(x, h0.detach())
                         
        # reshaping; out: [batch_size, sequence_length, hidden_size]
        out = out[:, -1, :1]
        
        out = self.fc(out)
        
        return out
    
input_size = 10
hidden_size = 10
num_layers = 2
output_size = 1
model = RNN(input_size,hidden_size,num_layers,output_size,dropout=0)
print(model.parameters)