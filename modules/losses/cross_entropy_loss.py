import numpy as np


class CrossEntropyLoss:
    """Cross-Entropy loss with Softmax"""

    def __init__(self):
        pass

    def __call__(self, targets: np.ndarray, logits: np.ndarray) -> float:
        """Forward pass for Cross-Entropy Loss.

        For an one-hot encoded targets t and model output y:
            E = - (1 / N) Σ(i=0 to N-1) Σ(k=0 to K-1) t_ik * ln(y_k (x_i)),

            where:
                - N is the number of data points,
                - K is the number of classes,
                - t_{ik} is the value from OHE target matrix for data point i and class k,
                - y_k (x_i) is model output after softmax for data point i and class k.

        Numerically stable formula:
            E = (1 / N) Σ(i=0 to N-1) Σ(k=0 to K-1) t_ik * ( ln( Σ(l=0 to K-1) e^(z_il - c_i) ) - (z_ik - c_i) ),

            where:
                - N is the number of data points,
                - K is the number of classes,
                - t_{ik} is the value from OHE target matrix for data point i and class k,
                - z_{il} is the model output before softmax for data point i and class l,
                - z is the model output before softmax (logits),
                - c_i is maximum value for each data point i in vector z_i.

        Parameters:
            targets: The one-hot encoded target data.
            logits: The model output before softmax.

        Returns:
            float: The value of the loss function.
        """
        stable_logits = logits - np.max(logits, axis=1, keepdims=True)
        exp_z = np.exp(stable_logits)
        sum_exp_z = np.sum(exp_z, axis=1, keepdims=True)
        y = np.log(sum_exp_z) - stable_logits

        # To speed up on backward
        self.softmax_cache = exp_z / sum_exp_z
 
        return np.sum((targets * y), axis=1).mean()


    def backward(self, targets: np.ndarray) -> np.ndarray:
        """Backward pass for Cross-Entropy Loss.

        For mini-batch, backward pass can be defined as follows:
            ∇_{Z^L} E = 1 / N (y - t)
            y = Softmax(Z^L)

            where:
                - Z^L - the model output before softmax,
                - t (N x K matrix): One-Hot encoded targets representation.

        Args:
            targets: The one-hot encoded target data.

        Returns:
            np.ndarray: ∇_{Z^L} E - a matrix of shape (batch_size, K)
        """
        grad = (1 / targets.shape[0]) * (self.softmax_cache - targets)

        return grad
