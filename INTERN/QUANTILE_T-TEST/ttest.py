from typing import List, Tuple
import numpy as np
from scipy import stats


def ttest(
    control: List[float],
    experiment: List[float],
    alpha: float = 0.05,
) -> Tuple[float, bool]:
    """
    Two-sample t-test for the means of two independent samples
    """
    statistic = stats.ttest_ind(
        a=control,
        b=experiment
    )
    p_value = statistic[1]
    result = p_value < alpha

    return p_value, bool(result)


if __name__ == '__main__':
    control = stats.norm.rvs(
        loc=5,
        scale=10,
        size=500
    )
    experiment = stats.norm.rvs(
        loc=5,
        scale=10,
        size=500
    )

    print(
        ttest(
            control=control,
            experiment=experiment
        )
    )