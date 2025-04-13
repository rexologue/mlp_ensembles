import numpy as np

from modules.layers.base import BaseLayer


class Linear(BaseLayer):
    """Fully-connected layer."""

    def __init__(self, in_features: int, out_features: int):
        """Linear layer initialization.

        Args:
            in_features: The number of input features (M_{l-1}).
            out_features: The number of output features (M_l).
        """
        # Training parameters initialization
        super().__init__(['weights', 'bias'])

        self.in_features = in_features
        self.out_features = out_features

        self.weights = np.zeros(shape=(out_features, in_features))
        self.grad_weights = None

        self.bias = np.zeros(shape=(out_features,))
        self.grad_bias = None


    def __call__(self, z: np.ndarray) -> np.ndarray:
        """Forward pass for fully-connected layer.

        For minibatch, FC layer forward pass can be defined as follows:
            z * W^T + b,

            where:
                - z (batch_size x in_features matrix) represents the output of the previous layer,
                - W (out_features x in_features matrix) a matrix represents the weight matrix,
                - b (vector of length out_features) represents the bias vector.

        During the training phase, inputs (or its transformation) are stored in self.inputs_cache for back propagation.

        Args:
            z: A matrix of shape (batch_size, in_features).

        Returns:
            np.ndarray: A matrix of shape (batch_size, out_features).
        """
        if self.trainable:
            self.inputs_cache = z
        else:
            self.inputs_cache = None

        return z @ self.weights.T + self.bias


    def backward(self, grad: np.ndarray) -> np.ndarray:
        """Backward pass for fully-connected layer.

        For mini-batch, FC layer backward pass can be defined as follows:
            ∇_{b^l} E = Σ(i=0 to N-1) u_i
            ∇_{W^l} E = (∇_{A^l} E)^T * Z^{l-1}
            ∇_{Z^{l-1}} E = ∇_{A^l} E * W^l

            where:
                - u_i:  i-th row of matrix ∇_{A^l} E,
                - W^l (out_features x in_features matrix): weights of current layer,
                - Z^{l-1} (batch_size x in_features matrix): inputs, that are passed during the forward propagation.

        Stores gradients of weights and bias in grad_weights and grad_bias.

        Args:
            grad: A matrix of shape (batch_size, out_features) - ∇_{A^l} E.

        Returns:
            np.ndarray: ∇_{Z^{l-1}} E - a matrix of shape (batch_size, in_features).
        """
        self.grad_bias = np.sum(grad, axis=0)
        self.grad_weights = grad.T @ self.inputs_cache

        return grad @ self.weights
