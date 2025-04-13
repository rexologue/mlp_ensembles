import numpy as np

from modules.layers.base import BaseLayer


class ReLU(BaseLayer):
    """ReLU (rectified linear unit) activation function."""

    def __init__(self):
        super().__init__()

    def __call__(self, a: np.ndarray) -> np.ndarray:
        """Forward pass.

        For mini-batch, ReLU forward pass can be defined as follows:
            z = max(0, a),

            where:
                - a (batch_size x M_l matrix) represents the output of fully-connected layer,
                - z (batch_size x M_l matrix) represents activations.

        During the training phase, inputs (or its transformation) are stored in self.inputs_cache for back propagation.

        Args:
            a: A matrix of shape (batch_size, M_l).

        Returns:
            np.ndarray: A matrix of shape (batch_size, M_l).
        """
        if self.trainable:
            self.inputs_cache = a
        else:
            self.inputs_cache = None
        
        return np.maximum(0, a)


    def backward(self, grad: np.ndarray) -> np.ndarray:
        """Backward pass for ReLU.

        For mini-batch, activation function backward pass can be defined as follows:
            ∇_{A^l} E = f'(A^l) * ∇_{Z^l} E,

            where:
                - f'(A^l) (batch_size x M_l matrix): derivative of activation function,
                - A^l (batch_size x M_l matrix): inputs, that are passed during the forward propagation.

        The derivative of ReLU activation function can be defined as follows:
            f'(x) = {0 if x < 0 and 1 otherwise}

        Args:
            grad: A matrix of shape (batch_size, M_l) - ∇_{Z^l} E.

        Returns:
            np.ndarray: ∇_{A^l} E - a matrix of shape (batch_size, M_l)
        """
        derivative = (self.inputs_cache > 0).astype(float)
        return derivative * grad


class LeakyReLU(BaseLayer):
    """Leaky ReLU (rectified linear unit) activation function."""

    def __init__(self, alpha=0.01):
        super().__init__()
        self.alpha = alpha


    def __call__(self, a: np.ndarray) -> np.ndarray:
        """Forward pass for LeakyReLU.

        For mini-batch, LeakyReLU forward pass can be defined as follows:
            z = max(alpha * a, a),

           where:
                - a (batch_size x M_l matrix) represents the output of fully-connected layer,
                - z (batch_size x M_l matrix) represents activations.

        During the training phase, inputs (or its transformation) are stored in self.inputs_cache for back propagation.

        Args:
            a: A matrix of shape (batch_size, M_l).

        Returns:
            np.ndarray: A matrix of shape (batch_size, M_l).
        """
        if self.trainable:
            self.inputs_cache = a
        else:
            self.inputs_cache = None
        
        return np.maximum(self.alpha * a, a)


    def backward(self, grad: np.ndarray) -> np.ndarray:
        """Backward pass for LeakyReLU.

        For mini-batch, activation function backward pass can be defined as follows:
            ∇_{A^l} E = f'(A^l) * ∇_{Z^l} E,

            where:
                - f'(A^l) (batch_size x M_l matrix): derivative of activation function,
                - A^l (batch_size x M_l matrix): inputs, that are passed during the forward propagation.


        The derivative of LeakyReLU activation function can be defined as follows:
            f'(x) = {1 if x > 0 and alpha otherwise}

        Args:
            grad: A matrix of shape (batch_size, M_l) - ∇_{Z^l} E.

        Returns:
            np.ndarray: ∇_{A^l} E - a matrix of shape (batch_size, M_l)
        """
        derivative = np.ones_like(self.inputs_cache)
        derivative[self.inputs_cache < 0] = self.alpha
        
        return derivative * grad


class Sigmoid(BaseLayer):
    """Sigmoid activation function."""

    def __init__(self):
        super().__init__()

    def __call__(self, a: np.ndarray) -> np.ndarray:
        """Forward pass for sigmoid function.

        For mini-batch, sigmoid function forward pass can be defined as follows:
            z = 1 / (1 + e^(-a)),

            where:
                - a (batch_size x M_l matrix) represents the output of fully-connected layer,
                - z (batch_size x M_l matrix) represents activations.

        For numerical stability, for negative values compute sigmoid using this function:
            z = e^(a) / (1 + e^(a))

        During the training phase, inputs (or its transformation) are stored in self.inputs_cache for back propagation.

        Args:
            a: A matrix of shape (batch_size, M_l).

        Returns:
            np.ndarray: A matrix of shape (batch_size, M_l).
        """
        positive_mask = a >= 0
        result = np.empty_like(a)

        result[positive_mask] = 1 / (1 + np.exp(-a[positive_mask]))

        exp_a = np.exp(a[~positive_mask])
        result[~positive_mask] = exp_a / (1 + exp_a)

        if self.trainable:
            self.inputs_cache = result
        else:
            self.inputs_cache = None

        return result

    def backward(self, grad: np.ndarray) -> np.ndarray:
        """Backward pass for Sigmoid.

        For mini-batch, activation function backward pass can be defined as follows:
            ∇_{A^l} E = f'(A^l) * ∇_{Z^l} E,

            where:
                - f'(A^l) (batch_size x M_l matrix): derivative of activation function,
                - A^l (batch_size x M_l matrix): inputs, that are passed during the forward propagation.


        The derivative of Sigmoid activation function can be defined as follows:
            f'(x) = f(x) * (1 - f(x)),

        where:
            - f(x) = 1 / (1 + e^(-x))

        Args:
            grad: A matrix of shape (batch_size, M_l) - ∇_{Z^l} E.

        Returns:
            np.ndarray: ∇_{A^l} E - a matrix of shape (batch_size, M_l)
        """
        derivative = self.inputs_cache * (1 - self.inputs_cache)

        return derivative * grad


class Tanh(BaseLayer):
    """Tanh activation function."""

    def __init__(self):
        super().__init__()

    def __call__(self, a: np.ndarray) -> np.ndarray:
        """Forward pass for Tanh.

        For mini-batch, Tanh function forward pass can be defined as follows:
            z = (e^a - e^(-a)) / (e^a + e^(-a)),

            where:
                - a (batch_size x M_l matrix) represents the output of fully-connected layer,
                - z (batch_size x M_l matrix) represents activations.

        During the training phase, inputs (or its transformation) are stored in self.inputs_cache for back propagation.

        Args:
            a: A matrix of shape (batch_size, M_l).

        Returns:
            np.ndarray: A matrix of shape (batch_size, M_l).
        """
        z = np.tanh(a)

        if self.trainable:
            self.inputs_cache = z
        else:
            self.inputs_cache = None

        return z

    def backward(self, grad: np.ndarray) -> np.ndarray:
        """Backward pass for Tanh.

        For mini-batch, activation function backward pass can be defined as follows:
            ∇_{A^l} E = f'(A^l) * ∇_{Z^l} E,

            where:
                - f'(A^l) (batch_size x M_l matrix): derivative of activation function,
                - A^l (batch_size x M_l matrix): inputs, that are passed during the forward propagation.


        The derivative of Tanh activation function can be defined as follows:
            f'(x) = 1 - f(x)^2,

        where:
            - f(x) = tanh(x)

        Args:
            grad: A matrix of shape (batch_size, M_l) - ∇_{Z^l} E.

        Returns:
            np.ndarray: ∇_{A^l} E - a matrix of shape (batch_size, M_l)
        """
        derivative = 1 - (self.inputs_cache**2)

        return derivative * grad
