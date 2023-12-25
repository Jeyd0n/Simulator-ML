from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import log_loss, make_scorer
from typing import List

def evaluate(model, embeddings, labels, cv=5) -> List[float]:
    kf = KFold(n_splits=cv)
    losses = cross_val_score(
        estimator=model, 
        X=embeddings,
        y=labels,
        cv=kf,
        scoring=make_scorer(log_loss, needs_proba=True),
        error_score='raise'
    )

    return losses