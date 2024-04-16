import torch

import torch.nn as nn


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size

        self.rnn = nn.RNN(input_size, hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, input):
        hidden = torch.zeros(1, input.size(0), self.hidden_size)
        output, _ = self.rnn(input, hidden)
        output = self.fc(output[-1])
        return output


class BiRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(BiRNN, self).__init__()
        self.hidden_size = hidden_size

        self.rnn = nn.RNN(input_size, hidden_size, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, output_size)

    def forward(self, input):
        hidden = torch.zeros(2, input.size(0), self.hidden_size)
        output, _ = self.rnn(input, hidden)
        output = self.fc(output[-1])
        return output


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size

        self.lstm = nn.LSTM(input_size, hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, input):
        hidden = (
            torch.zeros(1, input.size(0), self.hidden_size),
            torch.zeros(1, input.size(0), self.hidden_size),
        )
        output, _ = self.lstm(input, hidden)
        output = self.fc(output[-1])
        return output


class BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(BiLSTM, self).__init__()
        self.hidden_size = hidden_size

        self.lstm = nn.LSTM(input_size, hidden_size, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, output_size)

    def forward(self, input):
        hidden = (
            torch.zeros(2, input.size(0), self.hidden_size),
            torch.zeros(2, input.size(0), self.hidden_size),
        )
        output, _ = self.lstm(input, hidden)
        output = self.fc(output[-1])
        return output
