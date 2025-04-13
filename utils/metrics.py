import numpy as np
from typing import Union


def accuracy_score(targets: np.ndarray, predictions: np.ndarray) -> float:
    """Accuracy score.

    The formula is as follows:
        accuracy = (1 / N) Σ(i=0 to N-1) I(y_i == t_i),

        where:
            - N - number of samples,
            - y_i - predicted class of i-sample,
            - t_i - correct class of i-sample,
            - I(y_i == t_i) - indicator function.
    Args:
        targets: The true labels.
        predictions: The predicted classes.
    """
    return np.mean(targets == predictions)


def accuracy_score_per_class(targets: np.ndarray, predictions: np.ndarray) -> np.ndarray:
    """Accuracy score for each class.

    The formula is as follows:
        accuracy_k = (1 / N_k) Σ(i=0 to N) I(y_i == t_i) * I(t_i == k)

        where:
            - N_k -  number of k-class elements,
            - y_i - predicted class of i-sample,
            - t_i - correct class of i-sample,
            - I(y_i == t_i), I(t_i == k) - indicator function.

    Args:
        targets: The true labels.
        predictions: The predicted classes.

    Returns:
        list[float]: Accuracy for each class.
    """
    accuracy_per_class = []

    for cls in np.unique(targets):
        ind = targets == cls
        accuracy_per_class.append(
            np.mean(predictions[ind] == cls)
        )
        
    return accuracy_per_class


def balanced_accuracy_score(targets: np.ndarray, predictions: np.ndarray) -> float:
    """Balanced accuracy score.

    The formula is as follows:
        balanced_accuracy = (1 / K) Σ(k=0 to K-1) accuracy_k,
        accuracy_k = (1 / N_k) Σ(i=0 to N) I(y_i == t_i) * I(t_i == k)

        where:
            - K - number of classes,
            - N_k - number of k-class elements,
            - accuracy_k - accuracy for k-class,
            - y_i - predicted class of i-sample,
            - t_i - correct class of i-sample,
            - I(y_i == t_i), I(t_i == k) - indicator function.

    Args:
        targets: The true labels.
        predictions: The predicted classes.
    """
    return np.mean(accuracy_score_per_class(targets, predictions))


def confusion_matrix(targets: np.ndarray, predictions: np.ndarray, classes_num: Union[int, None] = None, normalize=False) -> np.ndarray:
    """Confusion matrix.

    Confusion matrix C with shape KxK:
        c[i, j] - number of observations known to be in class i and predicted to be in class j,

        where:
            - K is the number of classes.

    Args:
        targets: The true labels.
        predictions: The predicted classes.
        classes_num: The number of unique classes.
    """
    # Determine the number of classes
    if classes_num is None:
        max_class = max(np.max(targets), np.max(predictions)) if targets.size > 0 and predictions.size > 0 else 0
        classes_num = max_class + 1
    
    # Initialize confusion matrix
    cm = np.zeros((classes_num, classes_num), dtype=int)
    # Use numpy's advanced indexing to accumulate counts
    np.add.at(cm, (targets, predictions), 1)
    
    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        # Avoid division by zero by replacing zeros with 1
        row_sums[row_sums == 0] = 1
        cm = cm.astype(np.float32) / row_sums
    else:
        cm = cm.astype(np.float32)
    
    return cm

