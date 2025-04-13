from abc import abstractmethod

import numpy as np


class BaseLayer:
    """A base layer class."""

    def __init__(self, parameters: list[str] = ()):
        # List of training parameters
        self.parameters = parameters

        # Cache for backward propagation
        self.inputs_cache = None

        # Indicates mode of the layer
        self.trainable = True

    def train(self):
        """Sets training mode."""
        self.trainable = True

    def eval(self):
        """Sets evaluation mode."""
        self.trainable = False

    @abstractmethod
    def __call__(self, *args, **kwargs):
        raise NotImplementedError

    def load_params(self, params: dict):
        """Loads layer parameters.

        Args:
            params: dictionary with parameters names (as dict keys) and their values (as dict values)
        """
        for param_name, param_value in params.items():
            setattr(self, param_name, param_value)

    def get_params(self):
        """Returns layer parameters."""
        return {param: getattr(self, param) for param in self.parameters}

    @abstractmethod
    def backward(self, grad: np.ndarray):
        raise NotImplementedError

    def zero_grad(self):
        for param_name in self.parameters:
            setattr(self, f'grad_{param_name}', np.zeros_like(getattr(self, param_name)))
