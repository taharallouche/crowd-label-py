import numpy as np
import pandas as pd
import ray

from size_matters.utils.inventory import COLUMNS, RELIABILITY_BOUNDS, RULES, Dataset


@ray.remote
def apply_standard_approval_aggregator(annotations: pd.DataFrame) -> pd.DataFrame:
    alternatives = [
        column
        for column in annotations.columns
        if column != COLUMNS.question and column != COLUMNS.voter
    ]

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
    alternatives = [
        column
        for column in annotations.columns
        if column != COLUMNS.question and column != COLUMNS.voter
    ]
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
    Annotations: pd.DataFrame,
    dataset: Dataset,
    distance: str = RULES.jaccard,
) -> pd.DataFrame:
    """
    Takes Annotations as input and applies weighted approval rule.
    The weights are determined according to the ballot's size.
    These weights are the optimal Mallows noise with the input distance.
    :param Annotations: dataframe of answers as binary vectors
    :param data: name of the dataset
    :param distance: The distance of the noise model
    :return: agg_weighted: dataframe of the aggregated answers
    """

    Alternatives = dataset.alternatives

    # Initialize the aggregation dataframe
    Questions = list(Annotations.Question.unique())
    agg_weighted = pd.DataFrame(columns=[COLUMNS.question] + Alternatives)

    # weight of each voter and aggregate the answers for each question
    weights = pd.DataFrame(columns=[COLUMNS.voter, COLUMNS.weight])
    weights[COLUMNS.voter] = Annotations.Voter.unique()
    n = len(list(weights[COLUMNS.voter]))
    D = Annotations.loc[:, Alternatives].to_numpy()
    for i in range(len(Questions)):
        # Compute the weight of each voter according to the chosen distance
        if distance == RULES.euclid:
            weights[COLUMNS.weight] = np.sqrt(
                np.sum(D[n * i : n * (i + 1), :], axis=1) + 1  # noqa: E203
            ) - np.sqrt(
                np.sum(D[n * i : n * (i + 1), :], axis=1) - 1  # noqa: E203
            )
        elif distance == RULES.jaccard:
            weights[COLUMNS.weight] = 1 / np.sum(
                D[n * i : n * (i + 1), :], axis=1  # noqa: E203
            )
        elif distance == RULES.dice:
            weights[COLUMNS.weight] = 2 / (
                np.sum(D[n * i : n * (i + 1), :], axis=1) + 1  # noqa: E203
            )

        L = np.matmul(
            weights[COLUMNS.weight].T, D[n * i : n * (i + 1), :]  # noqa: E203
        )  # noqa: E203
        k = np.argmax(L)
        agg_weighted.loc[i] = [Questions[i]] + [
            t == k for t in range(0, len(Alternatives))
        ]

    return agg_weighted
