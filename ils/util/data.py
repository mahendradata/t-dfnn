#!/usr/bin/env python3
# --------------------------------------------------------------
# Author: Mahendra Data - mahendra.data@dbms.cs.kumamoto-u.ac.jp
# License: BSD 3 clause
# --------------------------------------------------------------

import numpy as np
import pandas as pd


def counter(x: np.ndarray) -> dict:
    unique, counts = np.unique(x, return_counts=True)
    return dict(zip(unique, counts))


def shuffle(x: np.ndarray, y: np.ndarray):
    assert len(x) == len(y)
    p = np.random.permutation(len(x))
    return np.copy(x[p]), np.copy(y[p])


def confusion_matrix(y_pred: np.ndarray, y: np.ndarray) -> pd.DataFrame:
    y_pred_labels = np.unique(y_pred)
    y_labels = np.unique(y)

    matrix = pd.DataFrame(np.zeros((len(y_pred_labels), len(y_labels))),
                          index=y_pred_labels, columns=y_labels, dtype=int)

    for c in y_labels:
        c_pred = np.where(y_pred == c)[0]
        for p, num in counter(y[c_pred]).items():
            matrix.loc[c, p] = num

    return matrix
