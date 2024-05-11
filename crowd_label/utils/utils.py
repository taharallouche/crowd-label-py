import numpy as np
import scipy.stats
from numpy.typing import NDArray


def confidence_margin_mean(
	data: NDArray, confidence: float = 0.95
) -> tuple[NDArray, NDArray, NDArray]:
	a = 1.0 * np.array(data)
	n = len(a)
	m, se = np.mean(a), scipy.stats.sem(a)
	h: NDArray = se * scipy.stats.t.ppf((1 + confidence) / 2.0, n - 1)
	return m, m - h, m + h
