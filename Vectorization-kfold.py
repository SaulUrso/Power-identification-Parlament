import os
import pandas as pd
import numpy as np
from sklearn import linear_model
import gensim
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import precision_recall_fscore_support
from pprint import pprint as print
from time import process_time
from Dataset_splitter import data_splitter
import csv


def documents_vector(documents, model):
    document_vectors = []
    for doc in documents:
        try:
            doc_words = [word for word in doc if word in model.wv.vocab]
        except (
            Exception
        ):  # Catch KeyError exception when accessing model.wv.vocab
            doc_words = [word for word in doc if word in model.wv.key_to_index]
        if len(doc_words) > 0:
            document_vectors.append(
                np.mean([model.wv[word] for word in doc_words], axis=0)
            )
        else:
            document_vectors.append(np.zeros(model.vector_size))
    return document_vectors


def k_fold_tfidf(
    filename,
    classifier1=LogisticRegression(
        class_weight="balanced", max_iter=500, penalty=None
    ),
):

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

    training_set = np.array(training_set)
    label_training = np.array(label_training)
    f1_scores_tfidf = []
    maxf1 = 0
    maxf1model = None

    for i, (train_index, test_index) in enumerate(kf.split(training_set)):

        print(f"Fold {i}:")
        print(f"  Train: index={train_index}")
        print(f"  Test:  index={test_index}")
        # DATASET SPLITTING
        X_train, X_val, y_train, y_val = (
            training_set[train_index],
            training_set[test_index],
            label_training[train_index],
            label_training[test_index],
        )

        # tfidf vectorization
        model1 = TfidfVectorizer(
            sublinear_tf=True, analyzer="char", ngram_range=(1, 3)
        )
        X_train_tfidf = model1.fit_transform(X_train)
        print(X_train_tfidf.get_shape())
        x_val_tfidf = model1.transform(X_val)
        # classifier fitting
        classifier1.fit(X_train_tfidf, y_train)
        predtfidf = classifier1.predict(x_val_tfidf)
        precisiontfidf, recalltfidf, fscoretfidf, _ = (
            precision_recall_fscore_support(y_val, predtfidf, average="macro")
        )
        f1_scores_tfidf.append(fscoretfidf)
        if maxf1 < fscoretfidf:
            maxf1 = fscoretfidf
            maxf1model = model1
        print("F-score tfidf")
        print(fscoretfidf)

    tempo2 = process_time()

    f1_scores_tfidf = np.array(f1_scores_tfidf)

    print("TF-IDF")
    print("mean ")
    print(np.mean(f1_scores_tfidf))
    print("standar deviation ")
    print(np.std(f1_scores_tfidf))
    print("tempo impiegato")
    print(tempo2 - tempo1)
    return maxf1model, maxf1


def k_fold_w2v(
    filename,
    classifier2=LogisticRegression(
        class_weight="balanced", max_iter=500, penalty=None
    ),
):

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

    training_set = np.array(training_set)
    label_training = np.array(label_training)
    f1_scores_w2v = []
    maxf1 = 0
    maxf1model = None

    for i, (train_index, test_index) in enumerate(kf.split(training_set)):

        print(f"Fold {i}:")
        print(f"  Train: index={train_index}")
        print(f"  Test:  index={test_index}")
        # DATASET SPLITTING
        X_train, X_val, y_train, y_val = (
            training_set[train_index],
            training_set[test_index],
            label_training[train_index],
            label_training[test_index],
        )
        # word2vec vectorization

        preproc_training_set = []
        preproc_validation_set = []
        # documents preprocessing
        for speech in X_train:
            preproc_training_set.append(gensim.utils.simple_preprocess(speech))
        for speech in X_val:
            preproc_validation_set.append(
                gensim.utils.simple_preprocess(speech)
            )
        # w2v model training
        modelw2v = gensim.models.Word2Vec(
            preproc_training_set,
            vector_size=150,
            window=10,
            min_count=2,
            workers=10,
        )
        # documents vectorization
        X_train_w2v = documents_vector(preproc_training_set, modelw2v)
        X_val_w2v = documents_vector(preproc_validation_set, modelw2v)
        # classifier fitting
        classifier2.fit(X_train_w2v, y_train)
        # classifier predicting
        predw2v = classifier2.predict(X_val_w2v)
        precisionw2v, recallw2v, fscorew2v, _ = (
            precision_recall_fscore_support(y_val, predw2v, average="macro")
        )
        f1_scores_w2v.append(fscorew2v)
        if maxf1 < fscorew2v:
            maxf1 = fscorew2v
            maxf1model = modelw2v
        print("F-score w2v")
        print(fscorew2v)

    tempo2 = process_time()
    f1_scores_w2v = np.array(f1_scores_w2v)

    print("Word2vec")
    print("mean ")
    print(np.mean(f1_scores_w2v))
    print("standar deviation ")
    print(np.std(f1_scores_w2v))

    print("tempo impiegato")
    print(tempo2 - tempo1)

    return maxf1model, maxf1


def k_fold_fstxt(
    filename,
    classifier3=LogisticRegression(
        class_weight="balanced", max_iter=500, penalty=None
    ),
):

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

    training_set = np.array(training_set)
    label_training = np.array(label_training)
    f1_scores_fstxt = []
    maxf1 = 0
    maxf1model = None

    for i, (train_index, test_index) in enumerate(kf.split(training_set)):

        print(f"Fold {i}:")
        print(f"  Train: index={train_index}")
        print(f"  Test:  index={test_index}")
        # DATASET SPLITTING
        X_train, X_val, y_train, y_val = (
            training_set[train_index],
            training_set[test_index],
            label_training[train_index],
            label_training[test_index],
        )
        # word2vec vectorization

        preproc_training_set = []
        preproc_validation_set = []
        # documents preprocessing
        for speech in X_train:
            preproc_training_set.append(gensim.utils.simple_preprocess(speech))
        for speech in X_val:
            preproc_validation_set.append(
                gensim.utils.simple_preprocess(speech)
            )
        # fstxt training
        modelfstxt = gensim.models.fasttext.FastText(
            sentences=preproc_training_set,
            vector_size=150,
            window=10,
            min_count=2,
            workers=10,
        )

        # documents vectorization
        X_train_w2v = documents_vector(preproc_training_set, modelfstxt)
        X_val_w2v = documents_vector(preproc_validation_set, modelfstxt)
        # classifier fitting
        classifier3.fit(X_train_w2v, y_train)
        # classifier predicting
        predw2v = classifier3.predict(X_val_w2v)
        precisionw2v, recallw2v, fscorefstxt, _ = (
            precision_recall_fscore_support(y_val, predw2v, average="macro")
        )
        f1_scores_fstxt.append(fscorefstxt)
        if maxf1 < fscorefstxt:
            maxf1 = fscorefstxt
            maxf1model = modelfstxt
        print("F-score fstxt")
        print(fscorefstxt)

    tempo2 = process_time()
    f1_scores_fstxt = np.array(f1_scores_fstxt)

    print("Fasttext")
    print("mean ")
    print(np.mean(f1_scores_fstxt))
    print("standar deviation ")
    print(np.std(f1_scores_fstxt))

    print("tempo impiegato")
    print(tempo2 - tempo1)
    return maxf1model, maxf1


def main():

    path = "./datasets/power/power-gb-train.tsv"

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

    # print(training_set)
    with open("ridge-results.csv", "w", newline="\n") as file:
        writer = csv.writer(file)
        writer.writerow(["Model", "F1-score"])
        model, f1 = k_fold_w2v(path, linear_model.RidgeClassifier())
        writer.writerow(["Word2Vec", f1])
        model, f1 = k_fold_tfidf(path, linear_model.RidgeClassifier())
        writer.writerow(["TF-IDF", f1])
        model, f1 = k_fold_fstxt(path, linear_model.RidgeClassifier())
        writer.writerow(["FastText", f1])


if __name__ == "__main__":
    main()
