import numpy as np


def compute_metrics(clf_metrics):
    return lambda eval_pred: clf_metrics.compute(
        np.round(eval_pred[0]), eval_pred[1]
    )


def tokenize_function(tokenizer):
    return lambda examples: tokenizer(
        examples["text"], padding="max_length", truncation=True
    )
