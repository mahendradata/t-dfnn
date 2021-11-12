#!/usr/bin/env python3
# --------------------------------------------------------------
# Author: Mahendra Data - mahendra.data@dbms.cs.kumamoto-u.ac.jp
# License: BSD 3 clause
# --------------------------------------------------------------
import numpy as np


class LabelMap(object):
    def __init__(self, y: np.ndarray):
        """Label class initialization."""
        self.names = np.unique(y)
        self.__map__()

    def __map__(self):
        self.transform_index = {value: index for index, value in enumerate(self.names)}  # Label mapping
        self.revert_index = {value: index for index, value in self.transform_index.items()}  # Label mapping
        self.length = self.names.shape[0]  # Count number of classes

    def not_in(self, y: np.ndarray) -> np.ndarray:
        new_label = np.unique(y)
        mask = np.logical_not(np.isin(new_label, self.names))  # Create mask to get new labels
        return new_label[mask]  # Get new labels

    def append(self, y: np.ndarray):
        """Append additional labels."""
        names_new = self.not_in(y)  # Get new labels
        self.names = np.concatenate((self.names, names_new))  # Combine the old and new labels
        self.__map__()

    def transform(self, y: np.ndarray) -> np.ndarray:
        """Transform the original label to match the output index of the model."""
        return np.vectorize(self.transform_index.get)(y)

    def revert(self, y: np.ndarray) -> np.ndarray:
        """Revert the transformed label to the original label."""
        return np.vectorize(self.revert_index.get)(y)


class LayerName(object):

    def __init__(self, name: str):
        self.name = name
        self.sections = 0

    def new(self) -> str:
        self.sections += 1
        return self.__str__()

    def __str__(self):
        return "{}.{}".format(self.name, self.sections)
