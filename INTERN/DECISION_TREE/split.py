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


def split(X: np.ndarray, y: np.ndarray, feature: int) -> float:
    """
    Find the best split for a node (one feature)
    
    Parameters
    ----------
    X : np.ndarray
        matrix of features 

    y : np.ndarray
        vector of target values 

    feature : int
        index of feature column in dataset

    Returns 
    -------
    float
        best split value for input feature


    """
    dataset = np.concatenate((X, y), axis=1)

    best_mse = 1e3
    best_threshold = 0

    for threshold in np.unique(dataset[feature]):
        dataset_left = np.where(dataset[feature] >= threshold, dataset, dataset[-1])
        dataset_right = np.where(dataset[feature] < threshold, dataset, dataset[-1])

        weighted_metric = weighted_mse(
            y_left=dataset_left[:, -1],
            y_right=dataset_right[:, -1]
        )

        if weighted_metric < best_mse:
            best_mse = weighted_metric
            best_threshold = threshold
    
    return best_threshold
