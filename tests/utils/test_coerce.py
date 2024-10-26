import pandas as pd
import pytest


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
def test_coerce_schema(
    annotations: pd.DataFrame,
    task_column: str,
    worker_column: str,
    expected_result: pd.DataFrame,
):
    # Given
    from hakeem.utils.coerce import coerce_schema

    # When
    result = coerce_schema(annotations, task_column, worker_column)

    # Then
    pd.testing.assert_frame_equal(expected_result, result)
