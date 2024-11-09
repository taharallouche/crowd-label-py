import pandas as pd
import pytest


@pytest.mark.ut
@pytest.mark.parametrize(
    [
        "lower_reliability_bound",
        "upper_reliability_bound",
        "task_column",
        "worker_column",
    ],
    [(0.1, 0.9, "task", "worker"), (0.2, 0.8, "question", "voter")],
)
def test_CondorcetAggregator_init(
    lower_reliability_bound: float,
    upper_reliability_bound: float,
    task_column: str,
    worker_column: str,
) -> None:
    # Given
    from hakeem.aggregation.aggregators.condorcet import CondorcetAggregator

    # When
    result = CondorcetAggregator(
        lower_reliability_bound=lower_reliability_bound,
        upper_reliability_bound=upper_reliability_bound,
        task_column=task_column,
        worker_column=worker_column,
    )

    # Then
    assert result.lower_reliability_bound == lower_reliability_bound
    assert result.upper_reliability_bound == upper_reliability_bound
    assert result.task_column == task_column
    assert result.worker_column == worker_column


@pytest.mark.ut
@pytest.mark.parametrize(
    [
        "annotations",
        "lower_reliability_bound",
        "upper_reliability_bound",
        "expected_result",
    ],
    [
        (
            pd.DataFrame(
                {
                    "task": ["q1", "q1", "q2"],
                    "worker": ["v1", "v2", "v1"],
                    "a": [1, 1, 0],
                    "b": [0, 1, 1],
                    "c": [0, 0, 1],
                    "d": [0, 0, 1],
                }
            ).set_index(["task", "worker"]),
            0.1,
            0.9,
            pd.Series(
                [0.9, 0.5, 0.1],
                index=pd.MultiIndex.from_tuples(
                    [("q1", "v1"), ("q1", "v2"), ("q2", "v1")], names=["task", "worker"]
                ),
            ),
        ),
        (
            pd.DataFrame(
                {
                    "task": ["q1", "q1", "q2", "q3"],
                    "worker": ["v1", "v2", "v1", "v3"],
                    "a": [1, 1, 0, 1],
                    "b": [0, 1, 1, 1],
                    "c": [0, 0, 1, 1],
                    "d": [0, 0, 1, 1],
                    "e": [0, 0, 0, 0],
                }
            ).set_index(["task", "worker"]),
            0.01,
            0.99,
            pd.Series(
                [0.99, 2 / 3, 1 / 3, 0.01],
                index=pd.MultiIndex.from_tuples(
                    [("q1", "v1"), ("q1", "v2"), ("q2", "v1"), ("q3", "v3")],
                    names=["task", "worker"],
                ),
            ),
        ),
    ],
)
def test_CondorcetAggregator__compute_reliabilities(
    annotations: pd.DataFrame,
    lower_reliability_bound: float,
    upper_reliability_bound: float,
    expected_result: pd.Series,
) -> None:
    # Given
    from hakeem.aggregation.aggregators.condorcet import CondorcetAggregator

    # When
    result = CondorcetAggregator(
        lower_reliability_bound=lower_reliability_bound,
        upper_reliability_bound=upper_reliability_bound,
    )._compute_reliabilities(annotations)

    # Then
    pd.testing.assert_series_equal(expected_result, result)


@pytest.mark.ut
@pytest.mark.parametrize(
    [
        "annotations",
        "lower_reliability_bound",
        "upper_reliability_bound",
        "expected_result",
    ],
    [
        (
            pd.DataFrame(
                {
                    "task": ["q1", "q1", "q2"],
                    "worker": ["v1", "v2", "v1"],
                    "a": [1, 1, 0],
                    "b": [0, 1, 1],
                    "c": [0, 0, 1],
                    "d": [0, 0, 1],
                }
            ).set_index(["task", "worker"]),
            0.1,
            0.9,
            pd.Series(
                [2.1972245773362196, 0.0, -2.197224577336219],
                index=pd.MultiIndex.from_tuples(
                    [("q1", "v1"), ("q1", "v2"), ("q2", "v1")], names=["task", "worker"]
                ),
            ),
        ),
        (
            pd.DataFrame(
                {
                    "task": ["q1", "q1", "q2", "q3"],
                    "worker": ["v1", "v2", "v1", "v3"],
                    "a": [1, 1, 0, 1],
                    "b": [0, 1, 1, 1],
                    "c": [0, 0, 1, 1],
                    "d": [0, 0, 1, 1],
                    "e": [0, 0, 0, 0],
                }
            ).set_index(["task", "worker"]),
            0.01,
            0.99,
            pd.Series(
                [
                    4.595119850134589,
                    0.6931471805599452,
                    -0.6931471805599454,
                    -4.59511985013459,
                ],
                index=pd.MultiIndex.from_tuples(
                    [("q1", "v1"), ("q1", "v2"), ("q2", "v1"), ("q3", "v3")],
                    names=["task", "worker"],
                ),
            ),
        ),
    ],
)
def test_CondorcetAggregator_compute_weights(
    annotations: pd.DataFrame,
    lower_reliability_bound: float,
    upper_reliability_bound: float,
    expected_result: pd.Series,
) -> None:
    # Given
    from hakeem.aggregation.aggregators.condorcet import CondorcetAggregator

    # When
    result = CondorcetAggregator(
        lower_reliability_bound=lower_reliability_bound,
        upper_reliability_bound=upper_reliability_bound,
    ).compute_weights(annotations)

    # Then
    pd.testing.assert_series_equal(expected_result, result, rtol=1e-5)


@pytest.mark.ut
def test_CondorcetAggregator__compute_reliabilities_handles_two_alternatives() -> None:
    # Given
    from hakeem.aggregation.aggregators.condorcet import CondorcetAggregator

    annotations = pd.DataFrame(
        {
            "task": ["q1", "q1", "q2"],
            "worker": ["v1", "v2", "v1"],
            "a": [1, 1, 0],
            "b": [0, 1, 1],
        }
    ).set_index(["task", "worker"])

    expected_reliabilities = pd.Series(
        [0.5, 0.5, 0.5],
        index=pd.MultiIndex.from_tuples(
            [("q1", "v1"), ("q1", "v2"), ("q2", "v1")], names=["task", "worker"]
        ),
    )

    # When
    result = CondorcetAggregator()._compute_reliabilities(annotations)

    # Then
    pd.testing.assert_series_equal(expected_reliabilities, result)
