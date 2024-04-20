import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from pprint import pprint as print
import torch
from torch import nn
from torch.utils.data import DataLoader


class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size=20, output_size=1):
        super(NeuralNetwork, self).__init__()

        self.stack = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.Sigmoid(),
            nn.Linear(hidden_size, output_size),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.stack(x)
        return x


def train(dataloader, model, loss_fn, optimizer, epochs=5):
    size = len(dataloader.dataset)
    model.train()
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}\n-------------------------------")
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(torch.float), y.to(torch.float)

            # Compute prediction error
            pred = model(X)
            loss = loss_fn(pred.to(torch.float), y)

            # Backpropagation
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if batch % 100 == 0:
                loss, current = loss.item(), (batch + 1) * len(X)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def nn_train(X_train_embed, X_val_embed, y_train, y_val):
    tensor_X_train = torch.from_numpy(X_train_embed)
    tensor_X_val = torch.from_numpy(X_val_embed)
    tensor_y_train = torch.from_numpy(y_train)
    tensor_y_val = torch.from_numpy(y_val)
    td = torch.utils.data.TensorDataset(tensor_X_train, tensor_y_train)
    ffnn = NeuralNetwork(X_train_embed.shape[1])
    train(
        DataLoader(td, batch_size=1),
        ffnn,
        nn.MSELoss(),
        torch.optim.SGD(ffnn.parameters(), lr=1e-3),
    )

    predtfidf = ffnn(tensor_X_val.to(torch.float))
    precisiontfidf, recalltfidf, fscoretfidf, _ = (
        precision_recall_fscore_support(
            tensor_y_val.numpy().astype(np.float32),
            torch.where(predtfidf > 0.5, torch.tensor(1), torch.tensor(0))
            .numpy()
            .astype(np.float32),
            average="macro",
        )
    )
    return precisiontfidf, recalltfidf, fscoretfidf
