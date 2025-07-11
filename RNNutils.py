import torch
import torch.nn as nn
from sklearn.metrics import precision_recall_fscore_support
import sklearn


class RNN(nn.Module):

    def __init__(
        self,
        input_dim,
        embedding_dim,
        hidden_dim,
        output_dim=1,
        dropout=0,
        device="cpu",
    ) -> None:
        """
        RNN module for sequence classification.

        Args:
            input_dim (int): The size of the input vocabulary.
            embedding_dim (int): The dimension of the word embeddings.
            hidden_dim (int): The dimension of the hidden state of the RNN.
            output_dim (int, optional): The dimension of the output. Defaults to 1.
            dropout (float, optional): The dropout probability. Defaults to 0.
            device (str, optional): The device to run the module on. Defaults to "cpu".
        """
        super().__init__()
        self.device = device
        self.embedding = nn.Embedding(input_dim, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(p=dropout)
        self.activation = nn.Sigmoid()

    def forward(self, x, x_lens):
        """
        Forward pass of the RNN module.

        Args:
            x (torch.Tensor): The input tensor of shape (batch_size, sequence_length).
            x_lens (torch.Tensor): The lengths of the input sequences.

        Returns:
            torch.Tensor: The output tensor of shape (batch_size, output_dim).
        """
        embedded = self.dropout(self.embedding(x))
        # print(embedded.shape)
        # ebedded dim: [ batch size, sentence length, embedding dim]
        packed_embedded = nn.utils.rnn.pack_padded_sequence(
            embedded,
            x_lens,
            batch_first=True,
            enforce_sorted=False,
        )
        packed_output, hidden = self.rnn(packed_embedded)
        # output dim: [ batch size, sentence length, hidden dim]

        # output, output_lens = nn.utils.rnn.pad_packed_sequence(
        #     packed_output, batch_first=True
        # )

        # hidden dim: [1, batch size, hidden dim] -> [batch size, hidden dim]
        output = self.fc(hidden.squeeze(0))
        return self.activation(output)


class BiRNN(nn.Module):

    def __init__(
        self,
        input_dim,
        embedding_dim,
        hidden_dim,
        output_dim=1,
        n_layers=1,
        dropout=0,
        mean_pooling=False,
        device="cpu",
    ) -> None:
        """
        Bidirectional Recurrent Neural Network (BiRNN) module.

        Args:
            input_dim (int): The size of the input vocabulary.
            embedding_dim (int): The dimension of the word embeddings.
            hidden_dim (int): The dimension of the hidden state of the RNN.
            output_dim (int, optional): The dimension of the output. Defaults to 1.
            dropout (float, optional): The dropout probability. Defaults to 0.
            device (str, optional): The device to run the module on. Defaults to "cpu".
            n_layers (int, optional): The number of hidden layers in the feedforward network. Defaults to 1.
        """
        super().__init__()
        self.device = device
        self.embedding = nn.Embedding(input_dim, embedding_dim)
        self.rnn = nn.RNN(
            embedding_dim, hidden_dim, batch_first=True, bidirectional=True
        )
        if n_layers > 1:
            layers = [nn.Linear(hidden_dim * 2, hidden_dim)]
            for _ in range(n_layers - 2):
                layers.append(nn.Dropout(p=dropout))
                layers.append(nn.Sigmoid())
                layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.Dropout(p=dropout))
            layers.append(nn.Sigmoid())
            layers.append(nn.Linear(hidden_dim, output_dim))
            self.fc = nn.Sequential(*layers)
        else:
            self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(p=dropout)
        self.activation = nn.Sigmoid()

        self.mean_pooling = mean_pooling

    def forward(self, x, x_lens):
        """
        Forward pass of the BiRNN module.

        Args:
            x (torch.Tensor): The input tensor of shape [batch size, sentence length].
            x_lens (torch.Tensor): The lengths of the input sentences in the batch.

        Returns:
            torch.Tensor: The output tensor of shape [batch size, output dim].
        """
        embedded = self.dropout(self.embedding(x))
        packed_embedded = nn.utils.rnn.pack_padded_sequence(
            embedded,
            x_lens,
            batch_first=True,
            enforce_sorted=False,
        )
        packed_output, hidden = self.rnn(packed_embedded)

        if self.mean_pooling:
            output, _ = nn.utils.rnn.pad_packed_sequence(
                packed_output, batch_first=True
            )
            output = self.fc(torch.mean(output, dim=1))
        else:
            hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
            output = self.fc(hidden)
        return self.activation(output)


class LSTM(nn.Module):

    def __init__(
        self,
        input_dim,
        embedding_dim,
        hidden_dim,
        output_dim=1,
        dropout=0,
        device="cpu",
    ) -> None:
        """
        Long Short-Term Memory (LSTM) module for sequence classification.

        Args:
            input_dim (int): The size of the input vocabulary.
            embedding_dim (int): The dimension of the word embeddings.
            hidden_dim (int): The dimension of the hidden state of the LSTM.
            output_dim (int, optional): The dimension of the output. Defaults to 1.
            dropout (float, optional): The dropout probability. Defaults to 0.
            device (str, optional): The device to run the module on. Defaults to "cpu".
        """
        super().__init__()
        self.device = device
        self.embedding = nn.Embedding(input_dim, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(p=dropout)
        self.activation = nn.Sigmoid()

    def forward(self, x, x_lens):
        """
        Forward pass of the LSTM module.

        Args:
            x (torch.Tensor): The input tensor of shape (batch_size, sequence_length).
            x_lens (torch.Tensor): The lengths of the input sequences.

        Returns:
            torch.Tensor: The output tensor of shape (batch_size, output_dim).
        """
        embedded = self.dropout(self.embedding(x))
        packed_embedded = nn.utils.rnn.pack_padded_sequence(
            embedded,
            x_lens,
            batch_first=True,
            enforce_sorted=False,
        )
        packed_output, (hidden, cell) = self.lstm(packed_embedded)
        output = self.fc(hidden.squeeze(0))
        return self.activation(output)


class BiLSTM(nn.Module):

    def __init__(
        self,
        input_dim,
        embedding_dim,
        hidden_dim,
        output_dim=1,
        dropout=0,
        n_layers=1,
        mean_pooling=False,
        pretrained_embedding=None,
        device="cpu",
    ) -> None:
        """
        Bidirectional Long Short-Term Memory (BiLSTM) module.

        Args:
            input_dim (int): The size of the input vocabulary.
            embedding_dim (int): The dimension of the word embeddings.
            hidden_dim (int): The dimension of the hidden state of the LSTM.
            output_dim (int, optional): The dimension of the output. Defaults to 1.
            dropout (float, optional): The dropout probability. Defaults to 0.
            pretrained_embedding (torch.Tensor, optional): A tensor of pretrained embeddings. Defaults to None.
            device (str, optional): The device to run the module on. Defaults to "cpu".
        """
        super().__init__()
        self.device = device
        if pretrained_embedding is not None:
            self.embedding = nn.Embedding.from_pretrained(pretrained_embedding)
            self.embedding.requires_grad = False
        else:
            self.embedding = nn.Embedding(input_dim, embedding_dim)
        self.lstm = nn.LSTM(
            embedding_dim, hidden_dim, batch_first=True, bidirectional=True
        )
        # self.fc = nn.Linear(hidden_dim * 2, output_dim)
        if n_layers > 1:
            layers = [nn.Linear(hidden_dim * 2, hidden_dim)]
            for _ in range(n_layers - 2):
                layers.append(nn.Dropout(p=dropout))
                layers.append(nn.Sigmoid())
                layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.Dropout(p=dropout))
            layers.append(nn.Sigmoid())
            layers.append(nn.Linear(hidden_dim, output_dim))
            self.fc = nn.Sequential(*layers)
        else:
            self.fc = nn.Linear(hidden_dim * 2, output_dim)

        self.dropout = nn.Dropout(p=dropout)
        self.activation = nn.Sigmoid()

        self.mean_pooling = mean_pooling

    def forward(self, x, x_lens):
        """
        Forward pass of the BiLSTM module.

        Args:
            x (torch.Tensor): The input tensor of shape [batch size, sentence length].
            x_lens (torch.Tensor): The lengths of the input sentences in the batch.

        Returns:
            torch.Tensor: The output tensor of shape [batch size, output dim].
        """
        embedded = self.dropout(self.embedding(x))
        packed_embedded = nn.utils.rnn.pack_padded_sequence(
            embedded,
            x_lens,
            batch_first=True,
            enforce_sorted=False,
        )
        packed_output, (hidden, cell) = self.lstm(packed_embedded)
        if self.mean_pooling:
            unpacked_output, _ = nn.utils.rnn.pad_packed_sequence(
                packed_output, batch_first=True
            )
            output = self.fc(torch.mean(unpacked_output, dim=1))
        else:
            hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
            output = self.fc(hidden)
        return self.activation(output)


def train_rnn(
    model: nn.Module,
    iterator: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    clip: float,
    device: torch.device = torch.device("cpu"),
):
    """
    Trains the RNN model on the given data iterator.

    Args:
        model (nn.Module): The RNN model to be trained.
        iterator (torch.utils.data.DataLoader): The data iterator containing the training data.
        optimizer (torch.optim.Optimizer): The optimizer used for training.
        criterion (nn.Module): The loss function used for training.
        clip (float): The maximum gradient norm for gradient clipping.
        device (torch.device, optional): The device to run the training on (default: "cpu").

    Returns:
        float: The average loss per epoch.
    """
    model.to(device)
    model.train()

    epoch_loss = 0

    for _, (src, trg, src_len) in enumerate(iterator):
        src, trg = src.to(device), trg.to(device)

        optimizer.zero_grad()

        output = model(src, src_len).to(device)
        # TODO: check if the output is correct

        loss = criterion(output, trg.float())

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(iterator)


def evaluate(
    model: nn.Module,
    iterator: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device = torch.device("cpu"),
):
    """
    Evaluate the model on the given data iterator.

    Args:
        model (nn.Module): The model to be evaluated.
        iterator (torch.utils.data.DataLoader): The data iterator.
        criterion (nn.Module): The loss criterion.
        device (torch.device, optional): The device to run the evaluation on. Defaults to torch.device("cpu").

    Returns:
        float: The average loss over the evaluation data.
    """
    model.eval()
    epoch_loss = 0

    # Lists to store true labels and predicted labels
    true_labels = []
    predicted_labels = []

    with torch.no_grad():
        for _, (src, trg, src_len) in enumerate(iterator):
            src, trg = src.to(device), trg.to(device)

            output = model(src, src_len)
            loss = criterion(output, trg.float())

            epoch_loss += loss.item()

            pred_l = torch.round(output)
            predicted_labels.extend(pred_l.cpu().numpy())
            true_labels.extend(trg.cpu().numpy())

    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels, predicted_labels, average="macro"
    )

    return epoch_loss / len(iterator), precision, recall, f1


def evaluate_confusion_matrix(
    model: nn.Module,
    iterator: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device = torch.device("cpu"),
):
    """
    Evaluate the model on the given data iterator.

    Args:
        model (nn.Module): The model to be evaluated.
        iterator (torch.utils.data.DataLoader): The data iterator.
        criterion (nn.Module): The loss criterion.
        device (torch.device, optional): The device to run the evaluation on. Defaults to torch.device("cpu").

    Returns:
        float: The average loss over the evaluation data.
    """
    model.eval()
    epoch_loss = 0

    # Lists to store true labels and predicted labels
    true_labels = []
    predicted_labels = []

    with torch.no_grad():
        for _, (src, trg, src_len) in enumerate(iterator):
            src, trg = src.to(device), trg.to(device)

            output = model(src, src_len)
            loss = criterion(output, trg.float())

            epoch_loss += loss.item()

            pred_l = torch.round(output)
            predicted_labels.extend(pred_l.cpu().numpy())
            true_labels.extend(trg.cpu().numpy())

    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels, predicted_labels, average="macro"
    )
    return sklearn.metrics.confusion_matrix(true_labels, predicted_labels)
