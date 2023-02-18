import numpy as np
import pandas as pd
import ray

from inventory import DataInfos


@ray.remote
# Select the label with greatest number of approvals
def simple_approval(Annotations: pd.DataFrame, dataset_info: DataInfos) -> pd.DataFrame:
    """
    Takes the Annotation dataframe as input and applies the majority rule to all the instances.
    :param Annotations: dataframe containing the answers of voters as binary vectors
    :param data: name of the dataset
    :return: agg_majority: dataframe structured like the GroundTruth dataframe containing the aggregated answers
    """

    Alternatives = dataset_info.alternatives

    # Initializing the aggregation dataframe
    Questions = Annotations.Question.unique()
    agg_majority = pd.DataFrame(columns=["Question"] + Alternatives)

    # Applying majority rule for each question
    for i in range(len(Questions)):
        # get the alternative with maximum approvals
        k = (
            Annotations.loc[Annotations["Question"] == Questions[i], Alternatives]
            .sum()
            .idxmax()
        )

        # add the result to the aggregation dataframe
        agg_majority.loc[i] = [Questions[i]] + [
            alternative == k for alternative in Alternatives
        ]

    return agg_majority


@ray.remote
# Estimate the voter's weight question-wise
def weighted_approval_qw(
    Annotations: pd.DataFrame, dataset_info: DataInfos
) -> pd.DataFrame:
    """
    Takes the Annotation dataframe as input and applies weighted approval rule to all the instances. The weights are
    determined question-wise according to the estimated reliability of the voter. This reliability is estimated from
    the number of alternatives that the voter selects in each of the questions.
    :param Annotations: dataframe
    containing the answers of voters as binary vectors
    :param data: name of the dataset
    :return: agg_weighted: dataframe structured like the GroundTruth dataframe containing the aggregated answers
    """

    Alternatives = dataset_info.alternatives

    # initialize the aggregation dataframe
    m = len(Alternatives)
    Questions = list(Annotations.Question.unique())
    agg_weighted = pd.DataFrame(columns=["Question"] + Alternatives)

    # Comute the weight of each voter and aggregate the answers in each question
    weights = pd.DataFrame(columns=["Voter", "Weight"])
    weights["Voter"] = Annotations.Voter.unique()
    n = len(list(weights["Voter"]))
    D = Annotations.loc[:, Alternatives].to_numpy()
    # vectorized version of the rest of the function
    for i in range(len(Questions)):
        # The number of alternatives selected by each voter in this question
        s = np.sum(D[n * i : n * (i + 1), :], axis=1)

        # The estimated reliability of each voter in this question
        p = (m - 1 - s) / (m - 2)
        p = np.clip(p, 0.001, 0.999)
        p = p.astype(float)

        # The weight of each voter in this question
        weights["Weight"] = np.log(p / (1 - p))

        L = np.matmul(weights["Weight"].T, D[n * i : n * (i + 1), :])
        k = np.argmax(L)
        agg_weighted.loc[i] = [Questions[i]] + [
            t == k for t in range(0, len(Alternatives))
        ]

    return agg_weighted


@ray.remote
# Compute the weight of a voter according to a specified mallows noise model
def mallows_weight(
    Annotations: pd.DataFrame, dataset_info: DataInfos, distance: str = "Jaccard"
) -> pd.DataFrame:
    """
    Takes the Annotation dataframe as input and applies weighted approval rule to all the instances. The weights are
    determined according to the number of alternatives that a ballot contains. These weights are the optimal weight
    when the noise model is supposed to be a Mallows noise with the correspondant distance.
    :param Annotations: dataframe containing the answers of voters as binary vectors
    :param data: name of the dataset
    :param distance: The distance of the noise model
    :return: agg_weighted: dataframe structured like the GroundTruth dataframe containing the aggregated answers
    """

    Alternatives = dataset_info.alternatives

    # Initialize the aggregation dataframe
    m = len(Alternatives)
    Questions = list(Annotations.Question.unique())
    agg_weighted = pd.DataFrame(columns=["Question"] + Alternatives)

    # Compute the weight of each voter and aggregate the answers for each question
    weights = pd.DataFrame(columns=["Voter", "Weight"])
    weights["Voter"] = Annotations.Voter.unique()
    n = len(list(weights["Voter"]))
    D = Annotations.loc[:, Alternatives].to_numpy()
    for i in range(len(Questions)):
        # Compute the weight of each voter according to the chosen distance
        if distance == "Euclid":
            weights["Weight"] = np.sqrt(
                np.sum(D[n * i : n * (i + 1), :], axis=1) + 1
            ) - np.sqrt(np.sum(D[n * i : n * (i + 1), :], axis=1) - 1)
        elif distance == "Jaccard":
            weights["Weight"] = 1 / np.sum(D[n * i : n * (i + 1), :], axis=1)
        elif distance == "Dice":
            weights["Weight"] = 2 / (np.sum(D[n * i : n * (i + 1), :], axis=1) + 1)

        L = np.matmul(weights["Weight"].T, D[n * i : n * (i + 1), :])
        k = np.argmax(L)
        agg_weighted.loc[i] = [Questions[i]] + [
            t == k for t in range(0, len(Alternatives))
        ]

    return agg_weighted
