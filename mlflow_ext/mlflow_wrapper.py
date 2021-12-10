"""mlflow_wrapper.py
Decorators for class/functions
"""
from .tracker import TrackMetric, TrackParam


__all__ = [
    "metric",
    "param",
]


def metric(*args, **kwargs):
    def function_wrapper(function):
        return TrackMetric(function, *args, **kwargs)
    return function_wrapper


def param(*args, **kwargs):
    def function_wrapper(function):
        return TrackParam(function, *args, **kwargs)
    return function_wrapper
