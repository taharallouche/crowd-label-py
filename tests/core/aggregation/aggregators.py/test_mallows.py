from contextlib import AbstractContextManager
from contextlib import nullcontext as does_not_raise

import pandas as pd
import pytest


@pytest.mark.ut
@pytest.mark.parametrize(
    ["annotations", "expected_result"],
    [
        (
            pd.DataFrame(
                {"task": ["q1", "q1"], "worker": ["v1", "v2"], "a": [1, 0], "b": [0, 1]}
            ).set_index(["task", "worker"]),
            pd.Series(
                [1, 1],
                index=pd.MultiIndex.from_tuples(
                    [("q1", "v1"), ("q1", "v2")], names=["task", "worker"]
                ),
            ),
        ),
    ],
)
def test_StandardApprovalAggregator_compute_weights(
    annotations: pd.DataFrame, expected_result: pd.Series
) -> None:
    # Given
    from hakeem.core.aggregation.aggregators.mallows import StandardApprovalAggregator

    # When
    result = StandardApprovalAggregator().compute_weights(annotations)

    # Then
    pd.testing.assert_series_equal(expected_result, result)


@pytest.mark.ut
@pytest.mark.parametrize(
    ["annotations", "expected_result"],
    [
        (
            pd.DataFrame(
                {
                    "task": ["q1", "q1", "q2"],
                    "worker": ["v1", "v2", "v1"],
                    "a": [1, 1, 1],
                    "b": [0, 1, 1],
                }
            ).set_index(["task", "worker"]),
            pd.Series(
                [1, 0.5, 0.5],
                index=pd.MultiIndex.from_tuples(
                    [("q1", "v1"), ("q1", "v2"), ("q2", "v1")], names=["task", "worker"]
                ),
            ),
        ),
    ],
)
def test_JaccardAggregator_compute_weights(
    annotations: pd.DataFrame, expected_result: pd.Series
) -> None:
    # Given
    from hakeem.core.aggregation.aggregators.mallows import JaccardAggregator

    # When
    result = JaccardAggregator().compute_weights(annotations)

    # Then
    pd.testing.assert_series_equal(expected_result, result)


@pytest.mark.ut
@pytest.mark.parametrize(
    ["annotations", "expected_result"],
    [
        (
            pd.DataFrame(
                {
                    "task": ["q1", "q1", "q2"],
                    "worker": ["v1", "v2", "v1"],
                    "a": [0, 1, 1],
                    "b": [0, 1, 1],
                }
            ).set_index(["task", "worker"]),
            pd.DataFrame(
                [[True, False], [True, False]],
                columns=pd.CategoricalIndex(
                    ["a", "b"],
                    categories=["a", "b"],
                    ordered=False,
                    dtype="category",
                ),
                index=pd.Index(["q1", "q2"], name="task"),
            ),
        ),
    ],
)
def test_JaccardAggregator_handles_empty_votes(
    annotations: pd.DataFrame, expected_result: pd.Series
) -> None:
    # Given
    from hakeem.core.aggregation.aggregators.mallows import JaccardAggregator

    aggregator = JaccardAggregator(task_column="task", worker_column="worker")

    # When
    result = aggregator.fit_predict(annotations)

    # Then
    pd.testing.assert_frame_equal(expected_result, result)


@pytest.mark.ut
@pytest.mark.parametrize(
    ["annotations", "expectation"],
    [
        (
            pd.DataFrame(
                {
                    "task": ["q1", "q1", "q2"],
                    "worker": ["v1", "v2", "v1"],
                    "a": [0, 1, 1],
                    "b": [0, 1, 1],
                }
            ).set_index(["task", "worker"]),
            pytest.raises(
                AssertionError, match="Jaccard weights are not defined for empty votes"
            ),
        ),
        (
            pd.DataFrame(
                {
                    "task": ["q1", "q2"],
                    "worker": ["v2", "v1"],
                    "a": [1, 1],
                    "b": [1, 1],
                }
            ).set_index(["task", "worker"]),
            does_not_raise(),
        ),
    ],
)
def test_JaccardAggregator_raises_on_empty_vote(
    annotations: pd.DataFrame, expectation: AbstractContextManager
) -> None:
    # Given
    from hakeem.core.aggregation.aggregators.mallows import JaccardAggregator

    aggregator = JaccardAggregator(task_column="task", worker_column="worker")

    # When / Then
    with expectation:
        aggregator.compute_weights(annotations)
