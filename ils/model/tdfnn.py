#!/usr/bin/env python3
# --------------------------------------------------------------
# Author: Mahendra Data - mahendra.data@dbms.cs.kumamoto-u.ac.jp
# License: BSD 3 clause
# --------------------------------------------------------------

import numpy as np
import threading

from ils.model.__common__ import LayerName
from ils.model.dfnn import DFNN
from ils.util.timer import Timer
from ils.util.log import Log
from ils.util.data import counter


class Node(object):
    def __init__(self, input_shape: tuple):
        self.model = None
        self.input_shape = input_shape
        self.leaves = dict()

        # Training Thread variable
        self.t_x = None
        self.t_y = None
        self.t_patience = None
        self.t_batch_size = None
        self.t_epochs = None

    def _expand_model(self, x: np.ndarray, y: np.ndarray,
                      patience: int, batch_size: int, epochs: int, name: str):
        y_pred = self.model.predict(x)  # predict the output using current node model

        for leaf in self.model.labels.names:  # loop for every leaf
            # Get the data that classified as this leaf
            mask = y_pred.flatten() == leaf

            if np.all(~mask):  # If all items in mask are False
                continue  # Skip and continue to the next leaf

            x_leaf = x[mask]
            y_leaf = y[mask]

            # Get the possible outputs from this leaf
            outputs = np.unique(y_leaf)  # possible outputs from this node
            mask = np.logical_not(np.isin(outputs, self.model.labels.names)) + np.isin(outputs, leaf)
            outputs = outputs[mask]  # possible outputs, excluding other leaf

            if outputs.shape[0] > 1:  # If more than one possible output
                if leaf not in self.leaves:
                    self.leaves[leaf] = Node(self.input_shape)

                mask = np.isin(y_leaf.flatten(), outputs)  # select data only for this leaf
                self.leaves[leaf].fit(x_leaf[mask], y_leaf[mask], patience, batch_size, epochs, name)

    def fit(self, x: np.ndarray, y: np.ndarray, patience: int, batch_size: int, epochs: int,
            name: str):
        if self.model is None:
            self.model = DFNN(self.input_shape, y, name)  # create new node

            # Set threading class variable
            self.t_x = x
            self.t_y = y
            self.t_patience = patience
            self.t_batch_size = batch_size
            self.t_epochs = epochs

            # Start multithreading
            fit_thread = FitTread(self)
            fit_thread.start()
            # Add this node to _running_threads list
            TDFNN.fit_threads.append(fit_thread)
        else:
            self._expand_model(x, y, patience, batch_size, epochs, name)

    def predict(self, x: np.ndarray, y_pred: np.ndarray, mask):
        timer = Timer()
        y_pred[mask] = self.model.predict(x[mask])  # Predict the output using this model
        Log.console("T-DFNN; il.model.tdfnn.Node.predict; predicting duration; {}; {}".format(
            counter(y_pred[mask]), timer.stop()))

        if len(self.leaves) != 0:  # If there are some leaf nodes
            for leaf, node in self.leaves.items():  # Loop through all leaves
                # Get the data that classified as this leaf
                submask = y_pred.flatten() == leaf
                submask = mask & submask  # merge mask
                if len(y_pred[submask]) > 0:  # If there are predicted output from this leaf
                    predict_thread = PredictThread(node, x, y_pred, submask)
                    predict_thread.start()
                    # Add this node to _running_threads list
                    TDFNN.predict_threads.append(predict_thread)


class FitTread(threading.Thread):
    def __init__(self, node: Node):
        threading.Thread.__init__(self)
        self.node = node

    def run(self):
        timer = Timer()
        history = self.node.model.fit(self.node.t_x, self.node.t_y, self.node.t_patience,
                                      self.node.t_batch_size, self.node.t_epochs)  # training
        Log.console("T-DFNN; il.model.tdfnn.NodeTread.run; training duration; {}; {}; epoch; {}".format(
            counter(self.node.t_y), timer.stop(), len(history.epoch)))


class PredictThread(threading.Thread):
    def __init__(self, node: Node, x: np.ndarray, y_pred: np.ndarray, mask):
        threading.Thread.__init__(self)
        self.node = node
        self.x = x
        self.y_pred = y_pred
        self.mask = mask

    def run(self):
        self.node.predict(self.x, self.y_pred, self.mask)  # Predict using the leaf node


class TDFNN(object):

    fit_threads = []
    predict_threads = []

    def __init__(self, input_shape: tuple, name: str = "IL"):
        self.input_shape = input_shape
        self.layer_name = LayerName(name)
        self.root = Node(self.input_shape)

    def fit(self, x: np.ndarray, y: np.ndarray,
            patience: int, batch_size: int, epochs: int):
        self.root.fit(x, y, patience, batch_size, epochs, self.layer_name.new())

        # Wait for all threads to complete
        for t in TDFNN.fit_threads:
            t.join()

        TDFNN.fit_threads = []

    def predict(self, x: np.ndarray) -> np.ndarray:
        mask = np.ones(len(x), dtype=bool)
        y_pred = np.zeros(len(x))
        self.root.predict(x, y_pred, mask)

        # Wait until all thread finished
        for t in TDFNN.predict_threads:
            t.join()
        TDFNN.predict_threads = []

        return y_pred

    def summary(self, root: Node = None, result: str = "", sep: str = "") -> str:
        if root is None:
            root = self.root

        result += "{}{}\n".format(sep, np.array_str(root.model.labels.names))
        for leaf, node in root.leaves.items():
            result = self.summary(node, result, "{}|{:>2}---".format(sep, leaf))

        return result
