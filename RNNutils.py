import pandas as pd
import torch
from torch.utils.data import Dataset

import torch.nn as nn


class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleRNN, self).__init__()
        self.hidden_size = hidden_size

        self.rnn = nn.RNN(input_size, hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, input):
        hidden = torch.zeros(1, input.size(0), self.hidden_size)
        output, _ = self.rnn(input, hidden)
        output = self.fc(output[-1])
        return output


class SimpleBiRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleBiRNN, self).__init__()
        self.hidden_size = hidden_size

        self.rnn = nn.RNN(input_size, hidden_size, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, output_size)

    def forward(self, input):
        hidden = torch.zeros(2, input.size(0), self.hidden_size)
        output, _ = self.rnn(input, hidden)
        output = self.fc(output[-1])
        return output


class SimpleLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleLSTM, self).__init__()
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


class SimpleBiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleBiLSTM, self).__init__()
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
