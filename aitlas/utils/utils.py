import importlib
from time import time


def get_class(class_name):
    """returns the class type for a given class name. Expects a string of type `module.submodule.Class`"""
    module = class_name[: class_name.rindex(".")]
    cls = class_name[class_name.rindex(".") + 1 :]
    return getattr(importlib.import_module(module), cls)


def current_ms():
    """returns current timestamp in ms"""
    return int(time() * 1000)
