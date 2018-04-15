import torch.nn as nn
import torch

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers, dropout):
        super(RNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers

        self.encoder = nn.Embedding(input_size, hidden_size)
        self.cells = nn.GRU(hidden_size, hidden_size, n_layers)
        self.decoder = nn.Linear(hidden_size, output_size)

    def forward(self, input, hidden):
        input = self.encoder(input)
        output, hidden = self.cells(input, hidden)
        output = self.decoder(output.view(output.size(0) * output.size(1), output.size(2)))
        return output, hidden

    def create_hidden(self, batch_size):
        # should this be small random instead of zeros
        # should this also be stored in the class rather than being passed around?
        return torch.zeros(self.n_layers, batch_size, self.hidden_size)
