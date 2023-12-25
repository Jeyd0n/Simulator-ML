import numpy as np

def ltv_error(y_true: np.array, y_pred: np.array) -> float:
    """
    Calculate asymmetric metrics for LTV 

    Parameters
    ----------
    y_true : np.array
        Array of true values

    y_pred : np.array
        Array of predicted values

    Returns
    -------
    float
        Error value


    """
    if np.sum(y_true - y_pred) < 0:
        error = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    else:
        error = np.mean(y_true - y_pred) ** 2

    return error