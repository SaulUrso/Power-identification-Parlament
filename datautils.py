import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np


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
        X_train.append(X_dev[train_indices])
        y_train.append(y_dev[train_indices])
        X_val.append(X_dev[val_indices])
        y_val.append(y_dev[val_indices])

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
'''
def documents_vector_pre(documents, word_vectors):
    document_vectors = []
    for doc in documents:

        doc_words = [word for word in doc if word in word_vectors.key_to_index]
        if len(doc_words) > 0:
            document_vectors.append(
                np.mean([word_vectors[word] for word in doc_words], axis=0)
            )
        else:
            # All vectors are of the same size
            document_vectors.append(np.zeros(word_vectors.vectors[0].shape[0]))
    return document_vectors

'''
def documents_vector_pre(documents, model):
    vectors = []
    for i, doc in enumerate(documents):
        word_vectors = [model[word] for word in doc if word in model]
        if word_vectors:
            vec = np.mean(word_vectors, axis=0)
        else:
            vec = np.zeros(model.vector_size)
            print(f"Documento vuoto (indice {i}): {doc}")  # Debug
        vectors.append(vec)
    
    vectors = np.array(vectors)
    
    # Debug: verifica che non ci siano valori NaN nei vettori
    if np.any(np.isnan(vectors)):
        print(f"Vettori contenenti NaN trovati. Indici: {np.where(np.isnan(vectors))}")
        raise ValueError("I vettori contengono NaN.")
    
    return vectors
