import os
import sys
import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import (
    make_scorer,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.model_selection import cross_val_score
from time import process_time

#  firt of all we define a function called data_splitter
#  which split the
#  dataset in test set and training set


def data_splitter(filename):

    texts, text_label = [], dict()

    # open the file and read it
    # with open(filename, "r") as f:

    csv_r = pd.read_csv(filename, sep="\t")

    for index, row in csv_r.iterrows():
        texts.append((row["id"], row["speaker"], row["text"]))
        text_label[row["text"]] = int(row.get("label", -1))

    # First, split the speakers to train/test sets such that
    # there are no overlap of the authors across the split.
    # This is similar to how orientation test set was split.
    text_set = list(text_label.keys())
    labelset = list(text_label.values())

    text_training, text_test, _, _ = train_test_split(
        text_set, labelset, test_size=0.1
    )

    text_test = set(text_test)

    # Now split the speeches based on speakers split above
    # but we consider only the test part

    test_set, label_test = [], []

    for i, (_, spk, text) in enumerate(texts):
        if text in text_test:
            test_set.append(text)
            label_test.append(text_label[text])

    # now we use the training part to split again into training and validation set
    # we create a new dictionary for the speaker and labels but we considerate
    # only the speaker in the training set

    Training_text_label = dict()

    for name in text_training:
        Training_text_label[name] = text_label[name]

    # once we have create this new dictionary we split again the data as we
    # see above  in the code so we use train_test_split of sklearn

    Traininig_textset = list(Training_text_label.keys())
    Training_labelset = list(Training_text_label.values())

    text_training, text_validation, _, _ = train_test_split(
        Traininig_textset, Training_labelset, test_size=0.2
    )

    text_validation = set(text_validation)

    # Now split the speeches based on speakers split above
    # but we consider the training part to split for validation
    # and training
    training_set, validation_set, label_training, label_validation = [], [], [], []

    for i, (_, spk, text) in enumerate(texts):

        if text in text_validation:
            validation_set.append(text)
            label_validation.append(text_label[text])
        else:
            training_set.append(text)
            label_training.append(text_label[text])

    return (
        training_set,
        validation_set,
        test_set,
        label_training,
        label_validation,
        label_test,
    )


def k_fold(filename):

    texts, text_label = [], dict()

    csv_r = pd.read_csv(filename, sep="\t")

    for index, row in csv_r.iterrows():
        texts.append((row["id"], row["speaker"], row["text"]))
        text_label[row["text"]] = int(row.get("label", -1))

    # First, split the speakers to train/test sets such that
    # there are no overlap of the authors across the split.
    # This is similar to how orientation test set was split.
    textset = list(text_label.keys())
    labelset = list(text_label.values())

    text_training, text_test, _, _ = train_test_split(
        textset, labelset, test_size=0.1
    )

    text_test = set(text_test)
    # Now split the speeches based on speakers split above
    # Using held out for training and test set

    training_set, test_set, label_training, label_test = [], [], [], []

    for i, (_, spk, text) in enumerate(texts):
        if text in text_test:
            test_set.append(text)
            label_test.append(text_label[text])
        else:
            training_set.append(text)
            label_training.append(text_label[text])

    # here we are using kfold on training set with 5 splits

    tempo1 = process_time()
    kf = KFold(n_splits=5, random_state=0, shuffle=True)
    m = LogisticRegression(
        class_weight="balanced", max_iter=500, penalty=None
    )
    vec = TfidfVectorizer(sublinear_tf=True, analyzer="char", ngram_range=(1, 3))
    training_set = np.array(training_set)
    label_training = np.array(label_training)
    f1_scores = []

    for i, (train_index, test_index) in enumerate(kf.split(training_set)):

        print(f"Fold {i}:")
        print(f"  Train: index={train_index}")
        print(f"  Test:  index={test_index}")

        X_train, X_val, y_train, y_val = (
            training_set[train_index],
            training_set[test_index],
            label_training[train_index],
            label_training[test_index],
        )
        X_train = vec.fit_transform(X_train)
        print(X_train.get_shape())
        x_val = vec.transform(X_val)
        m.fit(X_train, y_train)
        pred = m.predict(x_val)
        p, r, f, _ = precision_recall_fscore_support(y_val, pred, average="macro")
        f1_scores.append(f)
        print(f)

    tempo2 = process_time()
    f1_scores = np.array(f1_scores)

    print("mean ")
    print(np.mean(f1_scores))
    print("standar deviation ")
    print(np.std(f1_scores))
    print("tempo impiegato")
    print(tempo2 - tempo1)


# the main function


def main():

    path = sys.argv[1]

    if not (os.path.isfile(path)):
        print("this is not a File")
        return

    (
        training_set,
        validation_set,
        test_set,
        label_training,
        label_validation,
        label_test,
     ) = data_splitter(path)
    
    print(training_set)

    #k_fold(path)


if __name__ == "__main__":
    main()
