import pandas as pd
import pytest

from crowd_label.utils.inventory import COLUMNS


@pytest.mark.ut
def test_VoterMixin_handles_empty_input():
	# Given
	from crowd_label.aggregation.aggregators import VoterMixin

	annotations = pd.DataFrame(
		columns=["a", "b"],
		index=pd.MultiIndex.from_tuples((), names=[COLUMNS.question, COLUMNS.voter]),
	)

	expected_result = pd.DataFrame(
		columns=pd.CategoricalIndex(
			["a", "b"], categories=["a", "b"], ordered=False, dtype="category"
		),
		index=pd.Index([], name=COLUMNS.question),
		dtype="uint8",
	)

	# Then
	result = VoterMixin._get_aggregated_labels(annotations)

	# Then
	pd.testing.assert_frame_equal(expected_result, result)
