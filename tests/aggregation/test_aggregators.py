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

	# Then
	result = VoterMixin._get_aggregated_labels(annotations)

	# Then
	assert result.empty
	pd.testing.assert_index_equal(
		result.columns, annotations.columns, check_categorical=False, exact=False
	)
