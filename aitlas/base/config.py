from abc import ABC
import inspect
import json

from marshmallow import Schema, fields


class Config:
    """ Generic configuration object. This will store configuration names and values.
    """
    def __init__(self, dictionary):
        def _traverse(key, element):
            if isinstance(element, dict):
                return key, Config(element)
            else:
                return key, element

        self.__dict__.update(dict(_traverse(k, v) for k, v in dictionary.items()))
