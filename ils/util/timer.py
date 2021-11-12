#!/usr/bin/env python3
# --------------------------------------------------------------
# Author: Mahendra Data - mahendra.data@dbms.cs.kumamoto-u.ac.jp
# License: BSD 3 clause
# --------------------------------------------------------------
import time


class Timer(object):
    def __init__(self):
        self.begin = time.time()

    def start(self):
        self.begin = time.time()

    def stop(self) -> float:
        return time.time() - self.begin
