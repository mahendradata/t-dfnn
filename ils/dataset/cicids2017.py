#!/usr/bin/env python3
# --------------------------------------------------------------
# Author: Mahendra Data - mahendra.data@dbms.cs.kumamoto-u.ac.jp
# License: BSD 3 clause
# --------------------------------------------------------------
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from ils.util.log import Log
from ils.util.data import shuffle


class CICIDS2017:

    def __init__(self, dataset_path: str, col_label="Label", test_size: float = 0.2, random_state: int = 1):
        df = pd.read_csv(dataset_path, skipinitialspace=True)  # Read dataset

        # Split data & label
        y = df[[col_label]]
        x = df.drop(col_label, axis=1)

        # Data normalization
        scaler = MinMaxScaler()
        x.iloc[:] = scaler.fit_transform(x[:])

        # Convert to numpy
        x = x.to_numpy()
        y = y.to_numpy()

        # Split to train and test dataset
        self.train_x, self.eval_x, self.train_y, self.eval_y = train_test_split(x, y, stratify=y, test_size=test_size,
                                                                                random_state=random_state)

    def get(self, train_labels: list, eval_labels: list):
        mask = np.isin(self.train_y, train_labels).flatten()
        train_x = self.train_x[mask, :]
        train_y = self.train_y[mask, :]

        mask = np.isin(self.eval_y, eval_labels).flatten()
        eval_x = self.eval_x[mask, :]
        eval_y = self.eval_y[mask, :]

        # Get the copy of shuffled data
        train_x, train_y = shuffle(train_x, train_y)
        eval_x, eval_y = shuffle(eval_x, eval_y)

        return train_x, eval_x, train_y, eval_y

    @staticmethod
    def combine(src_dir: str) -> pd.DataFrame:
        # Remove unknown character
        Log.console("Remove unknown character.")
        file_path = os.path.join(src_dir, "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv")

        labels = {"Web Attack � Brute Force": "Web Attack-Brute Force",
                  "Web Attack � XSS": "Web Attack-XSS",
                  "Web Attack � Sql Injection": "Web Attack-Sql Injection"}

        df = pd.read_csv(file_path, skipinitialspace=True)

        for old_label, new_label in labels.items():
            df.Label.replace(old_label, new_label, inplace=True)

        df.to_csv(file_path, index=False)

        # Combine files
        Log.console("Combine files.")
        file_names = ["Monday-WorkingHours.pcap_ISCX.csv",
                      "Tuesday-WorkingHours.pcap_ISCX.csv",
                      "Wednesday-workingHours.pcap_ISCX.csv",
                      "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
                      "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
                      "Friday-WorkingHours-Morning.pcap_ISCX.csv",
                      "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
                      "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"]

        df = [pd.read_csv(os.path.join(src_dir, f), skipinitialspace=True) for f in file_names]
        df = pd.concat(df, ignore_index=True)

        return df

    @staticmethod
    def preprocessing(df: pd.DataFrame, col_label: str = "Label") -> pd.DataFrame:
        # Simplify the class label
        Log.console("Simplify the class label.")
        labels = {"Label": {"BENIGN": 0,
                            "DoS Hulk": 1,
                            "PortScan": 2,
                            "DDoS": 3,
                            "DoS GoldenEye": 4,
                            "FTP-Patator": 5,
                            "SSH-Patator": 6,
                            "DoS slowloris": 7,
                            "DoS Slowhttptest": 8,
                            "Bot": 9,
                            "Web Attack-Brute Force": 10,
                            "Web Attack-XSS": 11,
                            "Infiltration": 12,
                            "Web Attack-Sql Injection": 13,
                            "Heartbleed": 14}}

        for k, v in labels["Label"].items():
            print("{:>2} - {}".format(v, k))

        df.replace(labels, inplace=True)

        # Remove duplicate column
        Log.console("Remove duplicate columns.")
        df = df.drop(columns=['Fwd Header Length.1'])

        #  Fill NaN with median value of each class
        Log.console("Fill NaN in {} rows with median value of each class.".format(df.isna().any(axis=1).sum()))
        y = df[[col_label]]
        df = df.groupby(col_label).transform(lambda x: x.fillna(x.median()))

        # Replace infinite values with twice of maximum value of each class.
        df.replace([np.inf], np.nan, inplace=True)
        Log.console("Replace {} Inf values with twice of maximum value of each class.".format(df.isna().sum().sum()))
        df = pd.concat([df, y], axis=1, sort=False)
        df = df.groupby(col_label).transform(lambda x: x.fillna(x.max() * 2))

        # Merge
        df = pd.concat([df, y], axis=1, sort=False)

        return df
