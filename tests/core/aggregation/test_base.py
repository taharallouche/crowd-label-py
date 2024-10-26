import pandas as pd
import pytest

from hakeem.core.utils.inventory import COLUMNS


@pytest.mark.ut
def test_VoterMixin_handles_empty_input():
	# Given
	from hakeem.core.aggregation.base import VoterMixin

	annotations = pd.DataFrame(
		columns=["a", "b"],
		index=pd.MultiIndex.from_tuples((), names=[COLUMNS.question, COLUMNS.voter]),
	)

	# Then
	result = VoterMixin._get_aggregated_labels(annotations)

	# Then
	assert result.empty


@pytest.mark.ut
@pytest.mark.parametrize(
	["weighted_votes", "expected_result"],
	[
		(
			pd.DataFrame(
				{
					COLUMNS.question: ["q1", "q1"],
					COLUMNS.voter: ["v1", "v2"],
					"a": [1, 0],
					"b": [0, 1],
					"c": [1, 1],
				}
			).set_index([COLUMNS.question, COLUMNS.voter]),
			pd.DataFrame(
				[[False, False, True]],
				columns=pd.CategoricalIndex(
					["a", "b", "c"],
					categories=["a", "b", "c"],
					ordered=False,
					dtype="category",
				),
				index=pd.Index(["q1"], name=COLUMNS.question),
			),
		),
	],
)
def test_Voter_Mixin_get_aggregated_labels_handles_one_question(
	weighted_votes: pd.DataFrame, expected_result: pd.DataFrame
) -> None:
	# Given
	from hakeem.core.aggregation.base import VoterMixin

	# When
	result = VoterMixin._get_aggregated_labels(weighted_votes)

	# Then
	pd.testing.assert_frame_equal(expected_result, result)
