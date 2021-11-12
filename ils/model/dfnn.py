#!/usr/bin/env python3
# --------------------------------------------------------------
# Author: Mahendra Data - mahendra.data@dbms.cs.kumamoto-u.ac.jp
# License: BSD 3 clause
# --------------------------------------------------------------

import numpy as np
import tensorflow as tf

from ils.model.__common__ import LabelMap, LayerName
from tensorflow import keras

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)


class DFNN(object):
    def __init__(self, input_shape: tuple, labels: np.ndarray, name: str = "DFNN"):
        self.input_shape = input_shape  # The model input shape
        self.labels = LabelMap(labels)
        self.model = self.__init_model__(name)

    def __init_model__(self, name: str):
        ln = LayerName(name)

        """Create initial model."""
        inputs = keras.layers.Input(shape=self.input_shape, name=ln.new())
        ly = keras.layers.Dense(120, activation="relu", name=ln.new())(inputs)
        ly = keras.layers.Dense(self.input_shape[0], activation="relu", name=ln.new())(ly)
        outputs = keras.layers.Dense(self.labels.length, activation="softmax", name=ln.new())(ly)

        model = keras.Model(inputs=inputs, outputs=outputs, name=ln.name)

        model.compile(loss="sparse_categorical_crossentropy",
                           metrics=["sparse_categorical_accuracy"],
                           optimizer="adam")

        return model

    def extend(self, labels: np.ndarray, name: str = "DFNN"):
        num_prev_output = self.labels.length
        self.labels.append(labels)
        self.model = self.__extend__((self.labels.length - num_prev_output), name)

    def __extend__(self, new_output: int, name: str):
        ln = LayerName(name)

        """Add new outputs to the existing model."""
        inputs = self.model.inputs
        ly = self.model.layers[2].output
        ly = keras.layers.Dense(new_output, activation="softmax", name=ln.new())(ly)
        outputs = keras.layers.concatenate([self.model.output, ly], name=ln.new())

        model = keras.Model(inputs=inputs, outputs=outputs, name=ln.name)

        model.compile(loss="sparse_categorical_crossentropy",
                      metrics=["sparse_categorical_accuracy"],
                      optimizer="adam")

        return model

    def fit(self, x: np.ndarray, y: np.ndarray,
            batch_size: int, epochs: int, patience: int = 0,
            validation_split: float = 0, verbose: bool = False) -> keras.callbacks.History:
        yr = self.labels.transform(y)  # Relabeling the label data
        cb = [keras.callbacks.EarlyStopping(monitor='loss', patience=patience)] if patience > 0 else None
        history = self.model.fit(x, yr, callbacks=cb, batch_size=batch_size, epochs=epochs,
                                 validation_split=validation_split, verbose=verbose)
        return history

    def predict(self, x: np.ndarray) -> np.ndarray:
        y_pred = self.model.predict(x, verbose=False)
        y_pred = np.argmax(y_pred, axis=1)
        return self.labels.revert(y_pred)
