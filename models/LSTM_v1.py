import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTMv1(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, depth, width, num_classes, dropout_prob):
        super(LSTMv1, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.dropout_lstm = nn.Dropout(dropout_prob)
        self.dense_layers = nn.ModuleList()
        self.dropout_layers = nn.ModuleList()

        first_layer = nn.Linear(hidden_size, width)
        self.dense_layers.append(first_layer)
        self.dropout_layers.append(nn.Dropout(dropout_prob))

        for _ in range(1, depth):
            layer = nn.Linear(width, width)
            self.dense_layers.append(layer)
            self.dropout_layers.append(nn.Dropout(dropout_prob))

        self.output_layer = nn.Linear(width, num_classes)

    def forward(self, x):
        h_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        lstm_output, (hidden, cell) = self.lstm(x, (h_0, c_0))
        last_hidden_state = self.dropout_lstm(hidden[-1])

        for dense_layer, dropout_layer in zip(self.dense_layers, self.dropout_layers):
            last_hidden_state = F.leaky_relu(dense_layer(last_hidden_state))
            last_hidden_state = dropout_layer(last_hidden_state)

        out = torch.sigmoid(self.output_layer(last_hidden_state))
        return out
