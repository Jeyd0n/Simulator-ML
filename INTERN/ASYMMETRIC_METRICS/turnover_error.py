import numpy as np

def turnover_error(y_true: np.array, y_pred: np.array) -> float:
    """
    Calculate asymmetric metrics for amount of items to be delivered

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
    if np.sum(y_true - y_pred) > 0:
        error = np.sum(np.abs(y_true - y_pred))
    else:
        error = np.sum((y_true - y_pred) ** 2)

    return error
