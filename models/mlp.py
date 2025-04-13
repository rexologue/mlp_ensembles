import sys

import numpy as np

from modules.layers.activations import ReLU, Sigmoid, LeakyReLU, Tanh
from modules.layers.dropout import Dropout
from modules.layers.linear import Linear
from modules.utils.parameter_initialization import ParametersInit


class MLP:
    """A class for implementing Multilayer perceptron model."""

    def __init__(self, config):
        self.config = config
        self.params_init = ParametersInit(config.params)
        self.layers = self._init_layers()


    def _init_layers(self) -> list:
        """MLP layers initialization."""
        layers = []

        for layer_config in self.config.layers:
            layer = getattr(sys.modules[__name__], layer_config['type'].name)(**layer_config['params'])

            if len(layer.parameters) != 0:
                self.params_init(layer)
            
            layers.append(layer)

        return layers


    def train(self):
        """Sets the training mode for each layer."""
        for layer in self.layers:
            layer.train()


    def eval(self):
        """Sets the evaluation mode for each layer."""
        for layer in self.layers:
            layer.eval()


    def __call__(self, inputs: np.ndarray) -> np.ndarray:
        """Forward propagation implementation.

        This method propagates inputs through all the layers from self.layers.

        Returns:
            np.ndarray: The updated inputs.
        """
        inputs = inputs.reshape(inputs.shape[0], -1)

        for layer in self.layers:
            inputs = layer(inputs)

        return inputs


    def load_params(self, parameters: list):
        """Loads model parameters."""
        assert len(parameters) == len(self.layers)

        for i, layer in enumerate(self.layers):
            layer.load_params(parameters[i])


    def get_params(self) -> list:
        """Returns model parameters."""
        parameters = []
        
        for layer in self.layers:
            parameters.append(layer.get_params())

        return parameters
