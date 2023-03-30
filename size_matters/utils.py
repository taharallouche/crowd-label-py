import numpy as np
import scipy.stats


def confidence_margin_mean(
    data: np.ndarray, confidence: float = 0.95
) -> "tuple[np.ndarray, np.ndarray, np.ndarray]":
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2.0, n - 1)
    return m, m - h, m + h
