from unittest.mock import patch

import pandas as pd
import pytest

from hakeem.core.aggregation.base import Aggregator
from hakeem.core.utils.inventory import COLUMNS


@pytest.mark.ut
def test_WeightedApprovalMixin_handles_empty_input():
	# Given
	from hakeem.core.aggregation.base import WeightedApprovalMixin

	annotations = pd.DataFrame(
		columns=["a", "b"],
		index=pd.MultiIndex.from_tuples((), names=[COLUMNS.question, COLUMNS.voter]),
	)

	# Then
	result = WeightedApprovalMixin._get_aggregated_labels(
		annotations, task_column=COLUMNS.question
	)

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
	from hakeem.core.aggregation.base import WeightedApprovalMixin

	# When
	result = WeightedApprovalMixin._get_aggregated_labels(
		weighted_votes, task_column=COLUMNS.question
	)

	# Then
	pd.testing.assert_frame_equal(expected_result, result)


@pytest.mark.ut
@pytest.mark.parametrize(
	["annotations", "task_column", "worker_column", "expected_result"],
	[
		(
			pd.DataFrame(
				{
					"task": ["q1", "q1"],
					"worker": ["v1", "v2"],
					"a": [1, 0],
					"b": [0, 1],
				}
			).set_index(["task", "worker"]),
			"task",
			"worker",
			pd.DataFrame(
				{
					"task": ["q1", "q1"],
					"worker": ["v1", "v2"],
					"a": [1, 0],
					"b": [0, 1],
				}
			).set_index(["task", "worker"]),
		),
		(
			pd.DataFrame(
				{
					"question": ["q1", "q1"],
					"voter": ["v1", "v2"],
					"a": [1, 0],
					"b": [0, 1],
				}
			),
			"question",
			"voter",
			pd.DataFrame(
				{
					"question": ["q1", "q1"],
					"voter": ["v1", "v2"],
					"a": [1, 0],
					"b": [0, 1],
				}
			).set_index(["question", "voter"]),
		),
		(
			pd.DataFrame(
				{
					"question": ["q1", "q1"],
					"voter": ["v1", "v2"],
					"a": [1, 0],
					"b": [0, 1],
				}
			).set_index("question"),
			"question",
			"voter",
			pd.DataFrame(
				{
					"question": ["q1", "q1"],
					"voter": ["v1", "v2"],
					"a": [1, 0],
					"b": [0, 1],
				}
			).set_index(["question", "voter"]),
		),
		(
			pd.DataFrame(
				{
					"extra_index_level": ["l1", "l2"],
					"question": ["q1", "q1"],
					"voter": ["v1", "v2"],
					"a": [1, 0],
					"b": [0, 1],
				}
			).set_index(["extra_index_level", "question"]),
			"question",
			"voter",
			pd.DataFrame(
				{
					"question": ["q1", "q1"],
					"voter": ["v1", "v2"],
					"a": [1, 0],
					"b": [0, 1],
				}
			).set_index(["question", "voter"]),
		),
	],
)
@patch.multiple(Aggregator, __abstractmethods__=set())
def test_Aggregator__coerce_annotations(
	annotations: pd.DataFrame,
	task_column: str,
	worker_column: str,
	expected_result: pd.DataFrame,
):
	# When
	result = Aggregator(
		task_column=task_column, worker_column=worker_column
	)._coerce_annotations(annotations)

	# Then
	pd.testing.assert_frame_equal(expected_result, result)
