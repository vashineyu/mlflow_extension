import inspect
from abc import ABCMeta
from functools import partial, update_wrapper
from typing import Callable

from .utils import set_experiment, get_mlflow_env


class MlFlowBaseLogger(metaclass=ABCMeta):
    """MLFlow base logger

    Args:
        function (callable): function to wrap
        metrice_names (list): give names to return values (used in log-metrics)
    """
    def __init__(self, function: Callable):
        update_wrapper(self, function)
        self.function = function

    @property
    def is_member_function(self):
        return self._get_function_type(self.function)

    def execute(self, obj=None, *args, **kwargs):
        if self.is_member_function:
            outputs = self.function(obj, *args, **kwargs)
        else:
            outputs = self.function(*args, **kwargs)
        return outputs

    def _get_function_type(self, function):
        """To make it works as good as deco in normal functions and class functions
        """
        signature = inspect.signature(function)
        return "self" in signature.parameters

    def _call_impl(self, obj=None, teleport=None, *args, **kwargs):
        return self.execute(*args, **kwargs)

    def __call__(self, obj=None, teleport=None, *args, **kwargs):
        """Used for acting as decorator to deco main calling function
        Must call self._set_experiment at first
        """
        teleport = get_mlflow_env()
        set_experiment(teleport)
        return self._call_impl(obj=None, teleport=None, *args, **kwargs)

    def __get__(self, obj, objtype):
        return partial(self._call_impl, obj)
