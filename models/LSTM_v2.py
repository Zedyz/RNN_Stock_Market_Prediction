import torch
import torch.nn as nn
import torch.nn.functional as F


class AdditiveAttention(nn.Module):
    def __init__(self, input_dim, attention_dim):
        super(AdditiveAttention, self).__init__()
        self.attention_net = nn.Linear(input_dim, attention_dim)
        self.context_vector = nn.Linear(attention_dim, 1, bias=False)

    def forward(self, lstm_output):
        attention_input = torch.tanh(self.attention_net(lstm_output))
        attention_scores = self.context_vector(attention_input)
        attention_weights = F.softmax(attention_scores, dim=1)
        context_vector = torch.sum(attention_weights * lstm_output, dim=1)
        return context_vector


class LSTMv2(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, depth, width, num_classes, dropout_prob, attention_dim=6):
        super(LSTMv2, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)

        self.dropout_lstm = nn.Dropout(dropout_prob)

        self.attention = AdditiveAttention(hidden_size * 2, attention_dim)

        self.dense_layers = nn.ModuleList()
        self.dropout_layers = nn.ModuleList()

        first_layer = nn.Linear(hidden_size * 2, width)
        self.dense_layers.append(first_layer)
        self.dropout_layers.append(nn.Dropout(dropout_prob))

        for _ in range(1, depth):
            layer = nn.Linear(width, width)
            self.dense_layers.append(layer)
            self.dropout_layers.append(nn.Dropout(dropout_prob))

        self.output_layer = nn.Linear(width, num_classes)

    def forward(self, x):
        h_0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)
        c_0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)

        lstm_output, _ = self.lstm(x, (h_0, c_0))
        context_vector = self.attention(lstm_output)

        for dense_layer, dropout_layer in zip(self.dense_layers, self.dropout_layers):
            context_vector = F.leaky_relu(dense_layer(context_vector))
            context_vector = dropout_layer(context_vector)

        out = torch.sigmoid(self.output_layer(context_vector))
        return out
