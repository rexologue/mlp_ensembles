import numpy as np


class OneHotEncoding:
    """Creates matrix of one-hot encoding vectors for input targets"""

    def __init__(self, classes_num: int):
        self.classes_num = classes_num

    def __call__(self, targets: np.ndarray) -> np.ndarray:
        """Makes OHE representation.

        One-hot encoding vector representation:
            t_i^(k) = 1 if k = t_i otherwise  0,

            where:
                - k in [0, self.k-1],
                - t_i - target class of i-sample.
        Args:
            targets: The targets matrix.
        """
        
        ohe = np.zeros((len(targets), self.classes_num))
        ohe[np.arange(len(targets)), targets] = 1.0

        return ohe
