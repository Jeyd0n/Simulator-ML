"""Baseline for Kaggle AB."""

from typing import Callable, Dict, List, Tuple, Union

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, RepeatedKFold
from tqdm import tqdm


def prepare_dataset(DATA_PATH: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare dataset.
    Load data, split into X and y, one-hot encode categorical

    Parameters
    ----------
    DATA_PATH: str :
        path to the dataset

    Returns
    -------
    Tuple[np.ndarray, np.ndarray] :
        X and y
    """
    df = pd.read_csv(DATA_PATH)
    df = df.drop(["ID"], axis=1)
    y = df.pop("y").values

    # select only numeric columns
    X_num = df.select_dtypes(include="number")

    # select only categorical columns and one-hot encode them
    X_cat = df.select_dtypes(exclude="number")
    X_cat = pd.get_dummies(X_cat)

    # combine numeric and categorical
    X = pd.concat([X_num, X_cat], axis=1)
    X = X.fillna(0).values

    return X, y


def cross_val_score(
    model: Callable,
    X: np.ndarray,
    y: np.ndarray,
    cv: Union[int, Tuple[int, int]],
    params_list: List[Dict],
    scoring: Callable,
    random_state: int = 42,
    show_progress: bool = False,
) -> np.ndarray:
    """
    Cross-validation score.

    Parameters
    ----------
    model: Callable :
        model to train (e.g. RandomForestRegressor)

    X: np.ndarray :

    y: np.ndarray :

    cv Union[int, Tuple[int, int]]:
        (Default value = 5)
        number of folds or (n_folds, n_repeats)
        if int, then KFold is used
        if tuple, then RepeatedKFold is used

    params_list: List[Dict] :
        list of model parameters

    scoring: Callable :
        scoring function (e.g. r2_score)

    random_state: int :
        (Default value = 0)
        random state for cross-validation

    show_progress: bool :
        (Default value = False)

    Returns
    -------
    np.ndarray :
        cross-validation scores [n_models x n_folds]

    """
    # YOUR CODE HERE
    metrics = []

    if isinstance(cv, int):
        kf = KFold(n_splits=cv, shuffle=True, random_state=random_state)
    elif isinstance(cv, tuple):
        kf = RepeatedKFold(n_splits=cv[0], n_repeats=cv[1], random_state=random_state)
    for params in params_list:
        metric = []
        for train_index, test_index in kf.split(X):
            X_train, y_train = X[train_index], y[train_index]
            X_test, y_test = X[test_index], y[test_index]
            
            model.set_params(**params)
            model.fit(X_train, np.log1p(y_train))

            metric.append(
                scoring(y_test, np.expm1(model.predict(X_test)))
            )
        metrics.append(metric)

    return np.array(metrics)


def compare_models(
    cv: Union[int, Tuple[int, int]],
    model: Callable,
    params_list: List[Dict],
    X: np.ndarray,
    y: np.ndarray,
    random_state: int = 42,
    alpha: float = 0.05,
    show_progress: bool = False,
) -> List[Dict]:
    """Compare models with Cross val.

    Parameters
    ----------
    cv: Union[int, Tuple[int, int]] :
        (Default value = 5)
        number of folds or (n_folds, n_repeats)
        if int, then KFold is used
        if tuple, then RepeatedKFold is used

    model: Callable :
        model to train (e.g. RandomForestRegressor)

    params_list: List[Dict] :
        list of model parameters

    X: np.ndarray :

    y: np.ndarray :

    random_state: int :
        (Default value = 0)
        random state for cross-validation

    alpha: float :
        (Default value = 0.05)
        significance level for t-test

    show_progress: bool :
        (Default value = False)

    Returns
    -------
    List[Dict] :
        list of dicts with model comparison results
        {
            model_index,
            avg_score,
            p_value,
            effect_sign
        }
    """
    results = []
    metrics = cross_val_score(
        model=model,
        X=X,
        y=y,
        cv=cv,
        params_list=params_list,
        scoring=r2_score,
        random_state=random_state,
        show_progress=show_progress,
    )
    baseline_metric = metrics[0]
    print(np.mean(baseline_metric))

    counter = 1
    for metric in metrics[1:]:
        result_dict = {}
        result_dict["model_index"] = counter
        score, p_value = ttest_rel(
            a=baseline_metric,
            b=metric
        )
        result_dict['avg_score'] = np.mean(metric)
        result_dict['p_value'] = p_value
        if p_value < alpha and np.mean(metric) > np.mean(baseline_metric):
            result_dict['effect_sign'] = 1
        elif p_value < alpha and np.mean(metric) < np.mean(baseline_metric):
            result_dict['effect_sign'] = -1 
        elif p_value > alpha:
            result_dict['effect_sign'] = 0
        results.append(result_dict)

        counter += 1

    results = pd.DataFrame(results).sort_values('avg_score', ascending=False)
    return results.to_dict('records')


def run() -> None:
    """Run."""

    data_path = "JUNIOR/KAGGLE_AB_TEST/train.csv.zip"
    random_state = 42
    cv = 5
    params_list = [
        {"max_depth": 10},  # baseline
        {"max_depth": 2},
        {"max_depth": 3},
        {"max_depth": 4},
        {"max_depth": 5},
        {"max_depth": 9},
        {"max_depth": 11},
        {"max_depth": 12},
        {"max_depth": 15},
    ]

    X, y = prepare_dataset(data_path)
    model = RandomForestRegressor(n_estimators=200, n_jobs=-1, random_state=random_state)

    result = compare_models(
        cv=(5, 3),
        model=model,
        params_list=params_list,
        X=X,
        y=y,
        random_state=random_state,
        show_progress=True,
    )
    print("KFold")
    print(pd.DataFrame(result))


if __name__ == "__main__":
    run()
