import os
from typing import Any
from typing import Tuple

import fire
import mlflow
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import PrecisionRecallDisplay
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import RocCurveDisplay

IDENTIFIER = f'antifraud-{os.environ.get("KCHECKER_USER_USERNAME", "default")}'
TRACKING_URI = os.environ.get("TRACKING_URI")


def recall_at_precision(
    true_labels: np.ndarray,
    pred_scores: np.ndarray,
    min_precision: float = 0.95,
) -> float:
    """Compute recall at precision

    Args:
        true_labels (np.ndarray): True labels
        pred_scores (np.ndarray): Target scores
        min_precision (float, optional): Min precision for recall. Defaults to 0.95.

    Returns:
        float: Metric value
    """

    precision, recall, _ = precision_recall_curve(true_labels, pred_scores)
    metric = max(recall[precision >= min_precision])
    
    return metric


def recall_at_specificity(
    true_labels: np.ndarray,
    pred_scores: np.ndarray,
    min_specificity: float = 0.95,
) -> float:
    """Compute recall at specificity

    Args:
        true_labels (np.ndarray): True labels
        pred_scores (np.ndarray): Target scores
        min_specificity (float, optional): Min specificity for recall. Defaults to 0.95.

    Returns:
        float: Metric value
    """
    fpr, tpr, _ = roc_curve(
        y_true=true_labels,
        y_score=pred_scores
    )
    specificity = 1 - fpr 
    metric = max(tpr[specificity >= min_specificity])

    return metric


def curves(true_labels: np.ndarray, pred_scores: np.ndarray) -> Tuple[np.ndarray]:
    """Return ROC and FPR curves

    Args:
        true_labels (np.ndarray): True labels
        pred_scores (np.ndarray): Target scores

    Returns:
        Tuple[np.ndarray]: ROC and FPR curves
    """

    def fig2numpy(fig: Any) -> np.ndarray:
        fig.canvas.draw()
        img = fig.canvas.buffer_rgba()
        img = np.asarray(img)
        return img

    pr_curve = PrecisionRecallDisplay.from_predictions(true_labels, pred_scores)
    pr_curve = fig2numpy(pr_curve.figure_)

    roc_curve = RocCurveDisplay.from_predictions(true_labels, pred_scores)
    roc_curve = fig2numpy(roc_curve.figure_)

    return pr_curve, roc_curve


def job(
    train_path: str = "",
    test_path: str = "",
    target: str = "target",
):
    """Model training job

    Args:
        train_path (str): Train dataset path
        test_path (str): Test dataset path
        target (str): Target column name
    """
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(IDENTIFIER)
    mlflow.start_run()
    mlflow.set_tags({
        'task_type': 'anti-fraud',
        'framework': 'sklearn'
    })

    train_dataset = pd.read_csv(train_path)
    test_dataset = pd.read_csv(test_path)
    FEATURES = list(train_dataset.columns[:-1])
    TARGET = test_dataset.columns[-1]
    mlflow.log_params(params={
        'features': FEATURES,
        'target': TARGET
    })

    model = IsolationForest(n_estimators=100)
    model.fit(
        X=train_dataset.drop(target, axis=1)
    )
    mlflow.log_params({
        'model_type': model.__class__.__name__
    })
    mlflow.sklearn.log_model(
        model,
        artifact_path=IDENTIFIER,
        registered_model_name=IDENTIFIER
    )

    test_targets = test_dataset[target]
    pred_scores = -model.score_samples(test_dataset.drop(target, axis=1))

    recall_precision_95 = recall_at_precision(
        true_labels=test_targets,
        pred_scores=pred_scores
    )
    recall_specificity_95 = recall_at_specificity(
        true_labels=test_targets,
        pred_scores=pred_scores
    )
    roc_auc = roc_auc_score(
        y_true=test_targets,
        y_score=pred_scores
    )
    pr_curve, roc_curve = curves(
        true_labels=test_targets,
        pred_scores=pred_scores
    )
    mlflow.log_metrics({
        'roc_auc': roc_auc,
        'recall_precision_95': recall_precision_95,
        'recall_specificity_95': recall_specificity_95
    })

    mlflow.log_artifact(train_path, 'data/')
    mlflow.log_artifact(test_path, 'data/')
    mlflow.log_image(pr_curve, 'metrics/pr.png')
    mlflow.log_image(roc_curve, 'metrics/roc.png')

    mlflow.end_run()


if __name__ == "__main__":
    fire.Fire(job)
