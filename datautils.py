from collections import Counter
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import torch
from torchtext.vocab import vocab, Vocab
from torch.nn.utils.rnn import pad_sequence


def split_holdout_dataset(path, no_val=False):
    """
    Split the dataset into training, validation, and test sets.

    Args:
        path (str): The path to the dataset file.
        no_val (bool, optional): If True, no validation set will be created. Defaults to False.

    Returns:
        tuple: A tuple containing the training, validation, and test sets.

    Raises:
        None

    """

    csv_r = pd.read_csv(path, sep="\t")

    # only care about text and label
    csv_r = csv_r[["text", "label"]]

    X_dev, X_test, y_dev, y_test = train_test_split(
        csv_r["text"], csv_r["label"], test_size=0.1, random_state=0
    )

    # TODO: check if index must be dropped
    X_test.reset_index(drop=True, inplace=True)
    y_test.reset_index(drop=True, inplace=True)

    if no_val:

        # TODO: check if index must be dropped
        X_dev.reset_index(drop=True, inplace=True)
        y_dev.reset_index(drop=True, inplace=True)

        return X_dev, y_dev, X_test, y_test

    X_train, X_val, y_train, y_val = train_test_split(
        X_dev, y_dev, test_size=0.2, random_state=0
    )

    # TODO: check if index must be dropped
    X_train.reset_index(drop=True, inplace=True)
    y_train.reset_index(drop=True, inplace=True)
    X_val.reset_index(drop=True, inplace=True)
    y_val.reset_index(drop=True, inplace=True)

    return X_train, y_train, X_val, y_val, X_test, y_test


def split_kfold_dataset(path, no_val=False, n_splits=5):
    """
    Split the dataset into train, validation, and test sets using k-fold cross-validation.

    Args:
        path (str): The path to the dataset file.
        no_val (bool, optional): If True, no validation set will be created and only train and test sets will be returned. Defaults to False.
        n_splits (int, optional): The number of folds in the cross-validation. Defaults to 5.

    Returns:
        tuple: A tuple containing the train, validation, and test sets. If no_val is True, the tuple will only contain the train and test sets.

    """
    csv_r = pd.read_csv(path, sep="\t")

    # only care about text and label
    csv_r = csv_r[["text", "label"]]

    X = csv_r["text"]
    y = csv_r["label"]

    X_dev, X_test, y_dev, y_test = train_test_split(
        X, y, test_size=0.1, random_state=0
    )

    # TODO: check if index must be dropped
    X_dev.reset_index(drop=True, inplace=True)
    y_dev.reset_index(drop=True, inplace=True)
    X_test.reset_index(drop=True, inplace=True)
    y_test.reset_index(drop=True, inplace=True)

    # if no validation is specified, return train and test
    if no_val:
        return X_dev, y_dev, X_test, y_test

    kf = KFold(n_splits=n_splits, random_state=0, shuffle=True)

    X_train, X_val, y_train, y_val = [], [], [], []

    for train_indices, val_indices in kf.split(X=X_dev, y=y_dev):
        X_train.append(X_dev[train_indices].reset_index(drop=True))
        y_train.append(y_dev[train_indices].reset_index(drop=True))
        X_val.append(X_dev[val_indices].reset_index(drop=True))
        y_val.append(y_dev[val_indices].reset_index(drop=True))

    return X_train, y_train, X_val, y_val, X_test, y_test


def tf_idf_preprocessing(data):
    """
    Preprocesses the input data using TF-IDF vectorization.
    This function must be applied only on the training set.
    to preprocess the validation set use the returned vectorizer.

    Args:
        data (list): A list of strings representing the input data.

    Returns:
        scipy.sparse.csr_matrix: The TF-IDF transformed data.
        TfidfVectorizer: the vectorizer trained on data.

    """
    vectorizer = TfidfVectorizer(
        sublinear_tf=True,
    )
    return vectorizer.fit_transform(data), vectorizer


def documents_vector(documents, model):
    document_vectors = []
    for doc in documents:

        doc_words = [word for word in doc if word in model.wv.key_to_index]
        if len(doc_words) > 0:
            document_vectors.append(
                np.mean([model.wv[word] for word in doc_words], axis=0)
            )
        else:
            document_vectors.append(np.zeros(model.vector_size))
    return document_vectors


def documents_vector_wv(documents, wv):
    document_vectors = []
    for doc in documents:

        doc_words = [word for word in doc if word in wv.key_to_index]
        if len(doc_words) > 0:
            document_vectors.append(
                np.mean([wv[word] for word in doc_words], axis=0)
            )
        else:
            # All vectors are of the same size
            document_vectors.append(np.zeros(wv[0].shape[0]))
    return document_vectors


# obtain pytorch device
def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")

    elif torch.backends.mps.is_available():
        device = torch.device("mps")

    else:
        device = torch.device("cpu")
    return device


def build_vocab(dataset, tokenizer, min_freq=1):
    """
    Build a vocabulary from a dataset using a tokenizer.

    Args:
        dataset (str): pandas.core.series.Series, the dataset to build the vocabulary from.
        tokenizer (callable): A function that tokenizes a string.

    Returns:
        Vocab: A vocabulary object containing the tokenized words.

    """
    counter = Counter()
    for string_ in dataset:
        counter.update(tokenizer(string_))
    return Vocab(
        vocab(
            counter,
            specials=["<unk>", "<pad>", "<bos>", "<eos>"],
            min_freq=min_freq,
        )
    )


def data_process(dataset, vocab, tokenizer):
    data = []
    for text in dataset:
        tensor_ = torch.tensor(
            [vocab[token] for token in tokenizer(text)],
            dtype=torch.long,
        )
        data.append(tensor_)
    return data


class TextDataset(torch.utils.data.Dataset):
    """
    A custom PyTorch dataset for text data.

    Args:
        X (list): The input data.
        y (list): The target data.
        vocab (list): The vocabulary used for encoding the data.

    Raises:
        ValueError: If the lengths of X and y are not equal.

    Attributes:
        X (list): The input data.
        y (list): The target data.

    Methods:
        __getitem__(self, idx): Returns a single data item and its corresponding target.
        __len__(self): Returns the total number of data items in the dataset.
        generate_batch(data_batch): A helper method to generate a batch of data.

    """

    def __init__(self, X, y, data_vocab):
        if len(X) != len(y):
            raise ValueError("X and y must have the same number of items.")
        self.X = X
        self.y = y
        self.data_vocab = data_vocab

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

    def __len__(self):
        return len(self.X)

    def generate_batch(self, data_batch):
        """
        Generate a batch of padded sequences.

        Args:
            data_batch: A batch of sequences represented as a list of tensors.

        Returns:
            torch.Tensor: A tensor representing the padded batch of sequences.
        """

        (xx, yy) = zip(*data_batch)
        x_lens = [len(x) for x in xx]

        xx = pad_sequence(
            xx,
            batch_first=True,
            padding_value=self.data_vocab["<pad>"],
        )
        yy = torch.tensor(yy, dtype=torch.int64).reshape(-1, 1)
        x_lens = torch.tensor(x_lens, dtype=torch.long)
        return xx, yy, x_lens
