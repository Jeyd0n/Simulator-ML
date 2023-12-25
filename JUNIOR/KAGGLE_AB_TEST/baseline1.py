"""Baseline for Kaggle AB."""

from typing import Callable, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
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
    cv: int,
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

    cv :
        number of folds fo cross-validation

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
    # YOR CODE HERE
    metrics = []

    kf = KFold(n_splits=cv, shuffle=True, random_state=random_state)
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
    cv: int,
    model: Callable,
    params_list: List[Dict],
    X: np.ndarray,
    y: np.ndarray,
    random_state: int = 42,
    show_progress: bool = False,
) -> List[Dict]:
    """Compare models with Cross val.

    Parameters
    ----------
    cv: int :
        number of folds fo cross-validation

    model: Callable :
        model to train (e.g. RandomForestRegressor)

    params_list: List[Dict] :
        list of model parameters

    X: np.ndarray :

    y: np.ndarray :

    random_state: int :
        (Default value = 0)
        random state for cross-validation

    show_progress: bool :
        (Default value = False)

    Returns
    -------
    List[Dict] :
        list of dicts with model comparison results
        {
            model_index,
            avg_score,
            effect_sign
        }
    """
    # YOR CODE HERE
    results = []
    scores = cross_val_score(
        model=model,
        X=X,
        y=y,
        cv=cv,
        params_list=params_list,
        scoring=r2_score, 
        random_state=random_state, 
        show_progress=show_progress
    )
    baseline_score = np.mean(scores[0])\

    counter = 1
    for score in scores[1:]:
        result_dict = {}
        result_dict["model_index"] = counter
        result_dict["avg_score"] = np.mean(score)
        if np.mean(score) < baseline_score:
            result_dict["effect_sign"] = -1
        elif np.mean(score) > baseline_score:
            result_dict["effect_sign"] = 1
        else:
            result_dict["effect_sign"] = 0
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
        cv=cv,
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
