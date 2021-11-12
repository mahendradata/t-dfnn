#!/usr/bin/env python3
# --------------------------------------------------------------
# Author: Mahendra Data - mahendra.data@dbms.cs.kumamoto-u.ac.jp
# License: BSD 3 clause
#
# Running example: "python3 t-dfnn.py conf/t-dfnn.json 1 2 3"
# --------------------------------------------------------------

import os
import argparse
import sys
import pandas as pd
import pprint

from datetime import datetime
from ils.model.tdfnn import TDFNN
from ils.util.data import counter, confusion_matrix
from ils.util.timer import Timer
from ils.util.log import Log, report
from ils.util.configuration import Configuration, Dict
from ils.dataset.cicids2017 import CICIDS2017

pd.set_option('display.max_columns', 15)
pd.set_option('display.max_rows', 15)

# Global variables
CONF = None
DF = None


def evaluation(model, x, y, name, df_type, data_dir):
    timer = Timer()
    y_pred = model.predict(x)  # Predict the output
    Log.console("{}; main; predicting {} data duration; {}; {}".format(name, df_type, counter(y), timer.stop()))

    report_str, report_dict = report(y, y_pred)
    Log.console("{}; main; report of {} data;\n{}".format(name, df_type, report_str))
    Log.experiment(name, df_type, report_dict)

    matrix = confusion_matrix(y_pred, y)
    Log.console("{}; main; evaluation matrix of {} data; Predicted:row / Actual:col;\n{}".format(name, df_type, matrix))
    matrix.to_csv("{}/evaluation matrix-{}.csv".format(data_dir, df_type), index=True)


def training(model: TDFNN, x, y, name: str, batch: Dict) -> TDFNN:
    # Model initialization
    if model is None:
        input_shape = (x.shape[1],)
        model = TDFNN(input_shape, name)

    # Training
    timer = Timer()
    model.fit(x, y, batch.batch_size, batch.epochs, batch.patience)
    Log.console("{}; main; training duration; {}; {}".format(name, counter(y), timer.stop()))

    Log.console("{}; main; Model summary;\n{}".format(name, model.summary()))

    return model


def experiment(exp_id: str):
    global CONF, DF
    log_name = os.path.splitext(os.path.basename(sys.argv[0]))[0]  # Log filename
    data_dir = "{}/{}/{}".format(CONF.data_dir, log_name, exp_id)  # Log directory
    os.makedirs(data_dir, exist_ok=True)  # Create log directory

    Log.CONSOLE = "{}/{}.out.log".format(data_dir, log_name)
    Log.EXPERIMENT = "{}/{}.exp.log".format(data_dir, log_name)

    Log.console("START; {}".format(datetime.now().strftime("%d/%m/%Y %H:%M:%S")))
    Log.console("CONFIGURATION;\n{}".format(pprint.pformat(CONF)))

    model = None
    for num, batch in enumerate(CONF.batch):
        train_x, eval_x, train_y, eval_y = DF.get(batch.train, batch.eval)
        name = "T-DFNN.{}".format(num)  # Model name and batch number

        Log.console("{}; train_y distribution;\n{}".format(name, pprint.pformat(counter(train_y))))
        Log.console("{}; eval_y distribution;\n{}".format(name, pprint.pformat(counter(eval_y))))

        # Training
        model = training(model, train_x, train_y, name, batch)

        # Evaluate using training data
        evaluation(model, train_x, train_y, name, "training", data_dir)

        # Evaluate using evaluation data
        evaluation(model, eval_x, eval_y, name, "evaluation", data_dir)

    Log.console("FINISH; {}".format(datetime.now().strftime("%d/%m/%Y %H:%M:%S")))


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(description="CICIDS2017 classification using T-DFNN model.")
    parser.add_argument("conf", help="Program configuration.", type=str)
    parser.add_argument("exp", help="Experiment id.", type=str, nargs="+")
    parser.add_argument("-q", "--quiet", help="Suppress messages.", action="store_true")
    args = parser.parse_args()

    Log.VERBOSE = not args.quiet
    CONF = Configuration.read_json(args.conf)

    if CONF.gpu != "all":
        os.environ["CUDA_VISIBLE_DEVICES"] = CONF.gpu  # Limit GPU

    # Read dataset
    DF = CICIDS2017(CONF.dataset)

    for i in args.exp:
        experiment(i)
