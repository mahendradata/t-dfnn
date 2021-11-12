#!/usr/bin/env python3
# --------------------------------------------------------------
# This program run the preprocessing steps of CICIDS2017.
#
# Author: Mahendra Data - mahendra.data@dbms.cs.kumamoto-u.ac.jp
# License: BSD 3 clause
#
# Running example: "python3 preprocessing.py -o data/preprocessing.log \
#                      CICIDS2017-MachineLearning/ dataset/"
# --------------------------------------------------------------

import argparse
import pandas as pd

from sys import argv
from os import path
from ils.util.log import Log
from ils.dataset.cicids2017 import CICIDS2017

pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 100)


def summary(df: pd.DataFrame, col_label: str = "Label"):
    Log.console("Dataset shape: {}\n".format(df.shape), False)
    Log.console("Class distribution:\n{}\n".format(df[col_label].value_counts()), False)
    Log.console("Maximum values:\n{}\n".format(df.max()), False)
    Log.console("Minimum values:\n{}\n".format(df.min()), False)
    Log.console("Median values:\n{}\n".format(df.median()), False)
    Log.console("Columns:\n")
    for i, col in enumerate(df.columns):
        Log.console("{:>2} - {}".format(i, col), False)


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(description='CICIDS2017 preprocessing program.')
    parser.add_argument("dir", help="Extracted CICIDS2017 directory.", type=str)
    parser.add_argument("out", help="Output directory of preprocessed CICIDS2017.", type=str)
    parser.add_argument("-q", "--quiet", help="Suppress messages.", action="store_true")
    parser.add_argument("-o", "--log", help="Log file.", type=str,
                        default="{}.log".format(path.splitext(argv[0])[0]))
    args = parser.parse_args()

    Log.CONSOLE = args.log
    Log.VERBOSE = not args.quiet

    Log.console("Reading dataset.")
    dataset = CICIDS2017.combine(args.dir)

    Log.console("Dataset summary (before preprocessing).")
    summary(dataset)

    Log.console("Dataset preprocessing.")
    dataset = CICIDS2017.preprocessing(dataset)

    Log.console("Dataset summary (after preprocessing).")
    summary(dataset)

    Log.console("Saving dataset.")
    dataset = dataset.sample(frac=1)  # Shuffle dataset
    dataset.to_csv(path.join(args.out, "CICIDS2017-MachineLearning.csv"), index=False)
