import numpy as np


def softmax(logits: np.ndarray) -> np.ndarray:
    """Softmax function.

    The formula for numerically stable softmax function is:
        y_j = e^(z_j - c) / Σ(i=0 to K-1) e^(z_i - c),

        where:
            - y_j is the softmax probability of class j,
            - z_j is the model output (logits) for class j before softmax,
            - K is the total number of classes,
            - c is maximum(z),
            - Σ denotes summation.
    Returns:
        np.ndarray: The softmax probabilities.
    """
    exp_z = np.exp(logits - np.max(logits, axis=1, keepdims=True))

    return exp_z / np.sum(exp_z, axis=1, keepdims=True)