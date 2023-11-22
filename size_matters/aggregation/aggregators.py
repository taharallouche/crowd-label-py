import numpy as np
import pandas as pd

from size_matters.utils.inventory import COLUMNS, RELIABILITY_BOUNDS, RULES


def apply_standard_approval_aggregator(annotations: pd.DataFrame) -> pd.DataFrame:
    aggregated_labels = _get_aggregated_labels(annotations)

    return aggregated_labels


def apply_condorcet_aggregator(annotations: pd.DataFrame) -> pd.DataFrame:
    vote_size = annotations.sum(axis=1)
    reliabilities = (len(annotations.columns) - vote_size - 1) / (
        len(annotations.columns) - 2
    )
    reliabilities = reliabilities.clip(
        RELIABILITY_BOUNDS.lower, RELIABILITY_BOUNDS.upper
    )
    weights = np.log(reliabilities / (1 - reliabilities))

    aggregated_labels = _get_aggregated_labels(annotations, weights)

    return aggregated_labels


def apply_mallow_aggregator(
    annotations: pd.DataFrame,
    distance: str = RULES.jaccard,
) -> pd.DataFrame:
    vote_size = annotations.sum(axis=1)

    if distance == RULES.euclid:
        weights = np.sqrt(vote_size + 1) - np.sqrt(vote_size - 1)
    elif distance == RULES.jaccard:
        weights = 1 / vote_size
    elif distance == RULES.dice:
        weights = 2 / (vote_size + 1)

    aggregated_labels = _get_aggregated_labels(annotations, weights)

    return aggregated_labels


def _get_aggregated_labels(
    annotations: pd.DataFrame, weights: pd.Series | None = None
) -> pd.DataFrame:
    weighted_answers = annotations

    if weights is not None:
        weighted_answers = annotations.multiply(weights, axis="index")

    weighted_scores = weighted_answers.groupby(COLUMNS.question, sort=False)[
        annotations.columns
    ].sum()

    weighted_scores = weighted_scores.reindex(
        annotations.index.get_level_values(COLUMNS.question).unique()
    )

    winning_alternatives = pd.Categorical(
        weighted_scores.idxmax(axis=1),
        categories=annotations.columns,
        ordered=True,
    )

    aggregated_labels = pd.get_dummies(
        winning_alternatives, columns=annotations.columns
    )

    return aggregated_labels
