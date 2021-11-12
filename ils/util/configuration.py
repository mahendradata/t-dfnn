#!/usr/bin/env python3
# --------------------------------------------------------------
# Author: Mahendra Data - mahendra.data@dbms.cs.kumamoto-u.ac.jp
# License: BSD 3 clause
# --------------------------------------------------------------
import json


class Dict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class Configuration(object):

    @staticmethod
    def load(data):
        if type(data) is dict:
            return Configuration.__load_dict__(data)
        elif type(data) is list:
            return [Configuration.load(item) for item in data]
        else:
            return data

    @staticmethod
    def __load_dict__(data: dict):
        result = Dict()
        for key, value in data.items():
            result[key] = Configuration.load(value)
        return result

    @staticmethod
    def read_json(path: str):
        with open(path, "r") as f:
            result = Configuration.load(json.loads(f.read()))
        return result
