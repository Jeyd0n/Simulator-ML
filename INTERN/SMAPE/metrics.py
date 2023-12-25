import numpy as np


def smape(y_true: np.array, y_pred: np.array) -> float:
    """
    Compute SMAPE metric for regression task

    Parameters
    ----------
    True : np.array
        Array of true labels

    Predicted : np.array
        Array of predicted labels

    Returns
    -------
    float
        Metric value


    """
    if (np.abs(y_true) + np.abs(y_pred)) == 0:
        return 0

    return np.mean(2 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred)))
