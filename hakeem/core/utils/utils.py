import numpy as np
import scipy.stats
from numpy.typing import NDArray


def get_mean_confidence_interval(
	data: NDArray, confidence: float = 0.95
) -> tuple[NDArray, NDArray, NDArray]:
	mean, standard_error = np.mean(data), scipy.stats.sem(data)
	margin: NDArray = standard_error * scipy.stats.t.ppf(
		(1 + confidence) / 2.0, len(data) - 1
	)
	return mean, mean - margin, mean + margin
