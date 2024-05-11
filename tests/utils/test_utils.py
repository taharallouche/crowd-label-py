import numpy as np
import pytest

from crowd_label.utils.utils import get_mean_confidence_interval


@pytest.mark.ut
@pytest.mark.parametrize(
	"data, confidence, expected",
	[
		(
			np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
			0.95,
			(5.5, 3.3341494103866087, 7.665850589613392),
		),
		(
			np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
			0.99,
			(5.5, 2.388519354409619, 8.611480645590381),
		),
		(
			np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
			0.90,
			(5.5, 3.7449279866986926, 7.255072013301307),
		),
	],
)
def test_get_mean_confidence_interval(data, confidence, expected):
	result = get_mean_confidence_interval(data, confidence)
	assert result == pytest.approx(expected, abs=1e-6)
