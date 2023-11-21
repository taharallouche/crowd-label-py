import numpy as np
import pandas as pd
import ray

from size_matters.utils.inventory import COLUMNS, RELIABILITY_BOUNDS, RULES


@ray.remote
def apply_standard_approval_aggregator(annotations: pd.DataFrame) -> pd.DataFrame:
    alternatives = _get_alternatives(annotations)

    approvals_per_question = annotations.groupby(COLUMNS.question, sort=False)[
        alternatives
    ].sum()

    winning_alternatives = pd.Categorical(
        approvals_per_question.idxmax(axis=1),
        categories=alternatives,
        ordered=True,
    )

    standard_approval_output = pd.get_dummies(
        winning_alternatives, columns=alternatives
    )

    return standard_approval_output


@ray.remote
def apply_condorcet_aggregator(annotations: pd.DataFrame) -> pd.DataFrame:
    alternatives = _get_alternatives(annotations)
    questions = list(annotations[COLUMNS.question].unique())
    number_of_alternatives = len(alternatives)

    aggregated_labels = pd.DataFrame(columns=[COLUMNS.question] + alternatives)

    for i, question in enumerate(questions):
        question_annotations = annotations[annotations[COLUMNS.question] == question][
            alternatives
        ].to_numpy()

        vote_size = np.sum(question_annotations, axis=1)

        reliabilities = (number_of_alternatives - 1 - vote_size) / (
            number_of_alternatives - 2
        )
        reliabilities = np.clip(
            reliabilities, RELIABILITY_BOUNDS.lower, RELIABILITY_BOUNDS.upper
        )

        weights = np.log(reliabilities / (1 - reliabilities))

        weighted_scores = np.matmul(weights.T, question_annotations)
        most_likely_label = np.argmax(weighted_scores)

        aggregated_labels.loc[i] = [question] + [
            label == most_likely_label for label in range(number_of_alternatives)
        ]

    return aggregated_labels


@ray.remote
def apply_mallow_aggregator(
    annotations: pd.DataFrame,
    distance: str = RULES.jaccard,
) -> pd.DataFrame:
    alternatives = _get_alternatives(annotations)

    questions = list(annotations[COLUMNS.question].unique())
    aggregated_labels = pd.DataFrame(columns=[COLUMNS.question] + alternatives)

    for i, question in enumerate(questions):
        question_annotations = annotations[annotations[COLUMNS.question] == question][
            alternatives
        ].to_numpy()

        vote_size = np.sum(question_annotations, axis=1)

        if distance == RULES.euclid:
            weights = np.sqrt(vote_size + 1) - np.sqrt(vote_size - 1)
        elif distance == RULES.jaccard:
            weights = 1 / vote_size
        elif distance == RULES.dice:
            weights = 2 / (vote_size + 1)

        weighted_scores = np.matmul(weights.T, question_annotations)
        most_likely_label = np.argmax(weighted_scores)

        aggregated_labels.loc[i] = [question] + [
            label == most_likely_label for label in range(len(alternatives))
        ]

    return aggregated_labels


def _get_alternatives(annotations: pd.DataFrame) -> list[str]:
    return [
        column
        for column in annotations.columns
        if column != COLUMNS.question and column != COLUMNS.voter
    ]
