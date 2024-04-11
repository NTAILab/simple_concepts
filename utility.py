from functools import partial
from sklearn.metrics import accuracy_score, f1_score
import numpy as np

def sep_scorer(y_true, y_score, scorer):
    assert y_true.ndim == y_score.ndim == 2
    assert y_true.shape[1] == y_score.shape[1]
    result = np.zeros(y_true.shape[1])
    for i in range(y_true.shape[1]):
        result[i] = scorer(y_true[:, i], y_score[:, i])
    return result

def acc_sep_scorer(y_true, y_score):
    return sep_scorer(y_true, y_score, accuracy_score)

def f1_sep_scorer(y_true, y_score):
    f1_scorer = partial(f1_score, average='macro')
    return sep_scorer(y_true, y_score, f1_scorer)