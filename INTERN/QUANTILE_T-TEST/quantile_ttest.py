from typing import List
from typing import Tuple

import numpy as np
from scipy.stats import ttest_ind


def quantile_ttest(
    control: List[float],
    experiment: List[float],
    alpha: float = 0.05,
    quantile: float = 0.95,
    n_bootstraps: int = 1000,
) -> Tuple[float, bool]:
    """
    Bootstrapped t-test for quantiles of two samples.
    """
    control_quantiles = []
    experiment_quantiles = []
    for _ in range(n_bootstraps):
        bootstraped_control = np.random.choice(control, size=1000, replace=True)
        bootstraped_experiment = np.random.choice(experiment, size=1000, replace=True)
        control_quantiles.append(sorted(bootstraped_control)[int(1000 * quantile)])
        experiment_quantiles.append(sorted(bootstraped_experiment)[int(1000 * quantile)])

    statistic = ttest_ind(
        a=control_quantiles,
        b=experiment_quantiles
    )
    p_value = statistic[1]
    result = p_value < alpha

    return p_value, bool(result)
