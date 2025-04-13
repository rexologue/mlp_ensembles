import numpy as np


class SGD:
    """A class for implementing Stochastic gradient descent."""

    def __init__(self, model, learning_rate=1e-4, weight_decay: float = 0.0):
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.model = model

    def backward(self, grad: np.ndarray):
        """Backward pass for the model.

        This method propagates gradient across all layers from model.layers in reverse order.

        Computes gradients sequentially by passing through each layer in model.layers (using layer's backward() method).

        Args:
            grad: The gradient of the loss function w.r.t. the model output - ∇_{Z^L} E.
        """
        for layer in reversed(self.model.layers):
            grad = layer.backward(grad)

    def step(self):
        """Updates the parameters of the model layers."""
        for layer in self.model.layers:
            for param_name in layer.parameters:
                param = getattr(layer, param_name)
                grad = getattr(layer, f"grad_{param_name}")
                setattr(layer, param_name, self.update_param(param, grad))

    def update_param(self, param: np.ndarray, grad: np.ndarray) -> np.ndarray:
        """Update layer parameters with weight decay (L2 regularization).

        Layers parameters should be updated according to the following rule:
            w_new = w_old - γ * (∇_{w_old} E + λ * w_old),

            where:
                γ - self.learning_rate,
                w_old - layer's current parameter value,
                ∇_{w_old} E - gradient w.r.t. the layer's parameter,
                λ - self.weight_decay (L2 regularization coefficient),
                w_new - new parameter value to set.

        Args:
            param: A parameter matrix (w_old).
            grad: A gradient matrix (∇_{w_old} E).

        Returns:
            np.ndarray: w_new - updated parameter matrix.
        """
        # Add weight decay term to the gradient
        grad_with_decay = grad + self.weight_decay * param
        return param - self.learning_rate * grad_with_decay

    def zero_grad(self):
        """Reset gradient parameters for the model layers."""
        for layer in self.model.layers:
            layer.zero_grad()
