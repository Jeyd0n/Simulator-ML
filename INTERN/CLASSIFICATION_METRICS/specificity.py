from typing import Tuple

import numpy as np


def confusion_matrix(
    y_true: np.ndarray, y_pred: np.ndarray, threshold: float
) -> Tuple[int, int, int, int]:
    """Calculate confusion matrix."""
    TP, TN, FP, FN = 0, 0, 0, 0

    i = 0
    while i < len(y_true):
        predicted = int(y_pred[i] >= threshold)
        true = y_true[i]

        if predicted == true == 1:
            TP +=1
        elif predicted == true == 0:
            TN += 1
        elif predicted == 1 and true == 0:
            FP += 1
        elif predicted == 0 and true == 1:
            FN += 1

        i += 1

    return TP, TN, FP, FN


def specificity(TN: int, FP: int) -> float:
    """Calculate specificity."""
    metric = TN / (TN + FP)

    return metric


def test():
    """Test function."""
    y_true = np.array([1, 0, 1, 1, 0, 0, 1, 0, 0, 1])
    y_pred = np.array([0.9, 0.1, 0.8, 0.7, 0.2, 0.3, 0.6, 0.4, 0.5, 0.7])
    threshold = 0.5
    TP, TN, FP, FN = confusion_matrix(y_true, y_pred, threshold)

    assert TP == 5
    assert TN == 4
    assert FP == 1
    assert FN == 0

    assert np.allclose(specificity(TN, FP), 0.8)
    print("All tests passed.")


if __name__ == "__main__":
    test()