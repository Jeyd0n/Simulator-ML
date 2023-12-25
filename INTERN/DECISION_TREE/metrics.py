import numpy as np

def mse(y: np.ndarray) -> float:
    """
    Compute the mean squared error of a vector

    Parameters
    ----------
    y : np.ndarray
        array of true labels 

    Returns
    -------
    float
        mse metric value


    """
    mean = np.mean(y)
    print(mean)

    metric = np.mean((y - mean) ** 2)
    print(metric)

    return metric 


def weighted_mse(y_left: np.ndarray, y_right: np.ndarray) -> float:
    """
    Compute the weighted mean squared error of two vectors

    Parameters
    ----------
    y_left : np.ndarray
        array of left part values after splitting

    y_right : np.ndarray
        array of right part values after splitting

    Returns
    -------
    float
        weighted mse metric value for splitting 


    """
    n_left = len(y_left)
    n_right = len(y_right)
    mse_left = mse(y_left)
    mse_right = mse(y_right)

    weighted_metric = (mse_left * n_left + mse_right * n_right) / (n_left + n_right)
    
    return weighted_metric
