import numpy as np
import pandas as pd
import ray

from size_matters.inventory import COLUMNS, RULES, Dataset


@ray.remote
def standard_approval_voting(
    Annotations: pd.DataFrame, dataset: Dataset
) -> pd.DataFrame:
    """
    Standard approval voting returns the label that is approved by
    biggest number of voters for each question
    """

    Alternatives = dataset.alternatives

    approvals_per_question = Annotations.groupby(COLUMNS.question, sort=False)[
        Alternatives
    ].sum()

    winning_alternatives = pd.Categorical(
        approvals_per_question.idxmax(axis=1),
        categories=Alternatives,
        ordered=True,
    )

    standard_approval_output = pd.get_dummies(
        winning_alternatives, columns=Alternatives
    )

    return standard_approval_output


@ray.remote
# Estimate the voter's weight question-wise
def weighted_approval_qw(
    Annotations: pd.DataFrame, dataset: Dataset
) -> pd.DataFrame:
    """
    Takes Annotations as input and applies weighted approval rule .
    The weights are determined question-wise by estimating the reliabilities.
    This reliability is estimated from the number of alternatives that the
    voter selects in each of the questions.
    :param Annotations: dataframe
    containing the answers of voters as binary vectors
    :param data: name of the dataset
    :return: agg_weighted: dataframe of the aggregated answers
    """

    Alternatives = dataset.alternatives

    # initialize the aggregation dataframe
    m = len(Alternatives)
    Questions = list(Annotations.Question.unique())
    agg_weighted = pd.DataFrame(columns=[COLUMNS.question] + Alternatives)

    # weight of each voter and aggregate the answers in each question
    weights = pd.DataFrame(columns=[COLUMNS.voter, COLUMNS.weight])
    weights[COLUMNS.voter] = Annotations.Voter.unique()
    n = len(list(weights[COLUMNS.voter]))
    D = Annotations.loc[:, Alternatives].to_numpy()

    for i in range(len(Questions)):
        # The number of alternatives selected by each voter in this question
        s = np.sum(D[n * i : n * (i + 1), :], axis=1)  # noqa: E203

        # The estimated reliability of each voter in this question
        p = (m - 1 - s) / (m - 2)
        p = np.clip(p, 0.001, 0.999)
        p = p.astype(float)

        # The weight of each voter in this question
        weights[COLUMNS.weight] = np.log(p / (1 - p))

        L = np.matmul(
            weights[COLUMNS.weight].T, D[n * i : n * (i + 1), :]  # noqa: E203
        )
        k = np.argmax(L)
        agg_weighted.loc[i] = [Questions[i]] + [
            t == k for t in range(0, len(Alternatives))
        ]

    return agg_weighted


@ray.remote
# Compute the weight of a voter according to a specified mallows noise model
def mallows_weight(
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
