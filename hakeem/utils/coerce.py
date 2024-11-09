import pandas as pd


def coerce_schema(
    annotations: pd.DataFrame, task_column: str, worker_column: str
) -> pd.DataFrame:
    """
    Coerce the schema of the annotations DataFrame to ensure it contains the required
    columns and reindex it based on the specified task and worker columns.

    Parameters:
    annotations (pd.DataFrame): The DataFrame containing annotation data.
    task_column (str): The name of the column representing tasks.
    worker_column (str): The name of the column representing workers.

    Returns:
    pd.DataFrame: A DataFrame reindexed by the task and worker columns,
    containing only the label columns.

    Raises:
    ValueError: If the required columns are missing or if there are no label columns.
    """
    all_columns = annotations.reset_index().columns
    required = [task_column, worker_column]

    if missing := set(required) - set(all_columns):
        raise ValueError(
            f"Annotations should have {task_column} and"
            f" {worker_column} as columns or index levels, missing {missing}."
        )

    if set(all_columns) == set(required):
        raise ValueError("Annotations should have at least one label column")

    annotations = annotations.reset_index().set_index(required)[
        [column for column in annotations.columns if column not in required]
    ]

    return annotations
