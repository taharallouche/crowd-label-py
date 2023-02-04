# import dependencies
import numpy as np
import scipy.stats


# Computes the 0.95 confidence interval
def confidence_margin_mean(
    data: np.ndarray, confidence: float = 0.95
) -> "tuple[np.ndarray, np.ndarray, np.ndarray]":
    """
    Given sampled data and desired confidence level, return the mean and the bounds of the 95% confidence interval
    :param data: sampled data
    :param confidence: desired level of confidence
    :return: mean, lower bound of the CI, upper bound of the CI
    """
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2.0, n - 1)
    return m, m - h, m + h
