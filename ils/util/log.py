#!/usr/bin/env python3
# --------------------------------------------------------------
# Author: Mahendra Data - mahendra.data@dbms.cs.kumamoto-u.ac.jp
# License: BSD 3 clause
# --------------------------------------------------------------
import csv
import os
from datetime import datetime

import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report
from tensorflow import keras


class Log(object):
    CONSOLE = "console.log"
    EXPERIMENT = "experiment.log"
    VERBOSE = True

    @staticmethod
    def console(message, timed: bool = False):
        if Log.VERBOSE:
            print(message)

        with open(Log.CONSOLE, "a") as f:
            if timed:
                f.write("[{}] {}\n".format(datetime.now().strftime("%H:%M:%S"), message))
            else:
                f.write("{}\n".format(message))

    @staticmethod
    def experiment(name: str, dtype: str, report_dict: dict):
        # Create header (if needed)
        if not os.path.isfile(Log.EXPERIMENT):
            with open(Log.EXPERIMENT, mode='w') as f:
                f_csv = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                f_csv.writerow(["name", "type", "class", "precision", "recall", "f1-score", "support"])

        # Write experimental data
        with open(Log.EXPERIMENT, mode='a') as f:
            f_csv = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            for k, v in report_dict.items():
                if k == 'accuracy':
                    f_csv.writerow([name, dtype, k, "", "", v, ""])
                else:
                    f_csv.writerow([name, dtype, k, v["precision"], v["recall"], v["f1-score"], v["support"]])


def plot(history: keras.callbacks.History, acc_fig_path: str, loss_fig_path: str):
    # plot history for accuracy
    plt.plot(history.history['sparse_categorical_accuracy'])
    plt.title('Sparse Categorical Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['train'], loc='upper left')

    plt.savefig(acc_fig_path)
    plt.close()

    # plot history for loss
    plt.plot(history.history['loss'])
    plt.title('Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['train'], loc='upper left')

    plt.savefig(loss_fig_path)
    plt.close()


def report(y: np.ndarray, y_pred: np.ndarray, target_names: list = None) -> (str, dict):
    report_str = classification_report(y, y_pred, target_names=target_names)
    report_dict = classification_report(y, y_pred, target_names=target_names, output_dict=True)
    return report_str, report_dict
