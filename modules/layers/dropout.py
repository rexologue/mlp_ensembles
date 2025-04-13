import numpy as np

from modules.layers.base import BaseLayer


class Dropout(BaseLayer):
    """Dropout layer."""

    def __init__(self, p: float):
        """
        Args:
            p: probability of an element to be zeroed
        """
        super().__init__()
        self.keep_prob = 1. - p


    def __call__(self, z: np.ndarray) -> np.ndarray:
        """Forward pass.

        Dropout forward propagation for the training phase can be defined as follows:
            Z̃^(l) = (r^l ⊙ Z^l) / (1 - p),

            where:
                - r^l ~ Bernoulli(1 - p): a mask of ones and zeros of the same shape as Z^l, generated using
                            the Bernoulli distribution with probability of 1 equal to 1 - p (probability to keep)

        During the training phase, inputs (or its transformation) are stored in self.inputs_cache for back propagation.

        Args:
            z: A matrix of shape (batch_size, M_l) - Z^l.
        """
        if self.trainable:
            self.inputs_cache = np.random.binomial(n=1, p=self.keep_prob, size=z.shape).astype(float) / self.keep_prob
            return self.inputs_cache * z
        else:
            return z


    def backward(self, grad: np.ndarray) -> np.ndarray:
        """Backward pass for Dropout layer.

        Dropout backward propagation can be defined as follows:
            ∇_{Z^l} E = (r^l ⊙ ∇_{Z̃^l} E) / (1 - p),

            where:
                - r^l ~ Bernoulli(1 - p): a mask of ones and zeros of the same shape as ∇_{Z̃^l} E, generated
                            (during the forward pass!) using the Bernoulli distribution with probability of 1 equal
                            to 1 - p (probability to keep)

        Args:
            grad: A matrix of shape (batch_size, M_l) - ∇_{Z̃^l} E.
        """
        return self.inputs_cache * grad
