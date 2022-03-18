# import dependencies
import pandas as pd
import numpy as np
import ray


@ray.remote
# Select the label with greatest number of approvals
def simple_approval(Annotations, data="animals"):
    """
    Takes the Annotation dataframe as input and applies the majority rule to all the instances.
    :param Annotations: dataframe containing the answers of voters as binary vectors
    :param data: name of the dataset
    :return: agg_majority: dataframe structured like the GroundTruth dataframe containing the aggregated answers
    """
    # Setting the dataset
    if data == "animals":
        Alternatives = ["Leopard", "Tiger", "Puma", "Jaguar", "Lion(ess)", "Cheetah"]
    elif data == "textures":
        Alternatives = ["Gravel", "Grass", "Brick", "Wood", "Sand", "Cloth"]
    else:
        Alternatives = ["Hebrew", "Russian", "Japanese", "Thai", "Chinese", "Tamil", "Latin", "Hindi"]

    # Initializing the aggregation dataframe
    Questions = Annotations.Question.unique()
    agg_majority = pd.DataFrame(columns=["Question"] + Alternatives)

    # Applying majority rule for each question
    for i in range(len(Questions)):
        # List that will contains the number of approval each alternative gets
        L = []
        for alternative in Alternatives:
            # Compute the number of approvals for each alternative
            L += [int(sum(Annotations.loc[Annotations["Question"] == Questions[i], alternative]))]

        # Search for the alternative with maximum approvals and add it to the aggregation dataframe
        k = L.index(max(L))
        agg_majority.loc[i] = [Questions[i]] + [t == k for t in range(0, len(Alternatives))]
    return agg_majority


@ray.remote
# Estimate weights from all instances
def weighted_approval(Annotations, data="animals"):
    """
    Takes the Annotation dataframe as input and applies weighted approval rule to all the instances. The weights are
    determined according to the estimated reliability of the voter. This reliability is estimated from the number of
    alternatives that the voter selects in all the questions.
    :param Annotations: dataframe containing the answers of voters as binary vectors
    :param data: name of the dataset
    :return: agg_weighted: dataframe structured like the GroundTruth dataframe containing the aggregated answers
    """
    if data == "animals":
        Alternatives = ["Leopard", "Tiger", "Puma", "Jaguar", "Lion(ess)", "Cheetah"]
    elif data == "textures":
        Alternatives = ["Gravel", "Grass", "Brick", "Wood", "Sand", "Cloth"]
    else:
        Alternatives = ["Hebrew", "Russian", "Japanese", "Thai", "Chinese", "Tamil", "Latin", "Hindi"]

    # Initialize the aggregation dataframe
    m = len(Alternatives)
    Questions = list(Annotations.Question.unique())
    agg_weighted = pd.DataFrame(columns=["Question"] + Alternatives)

    # Computing the weight of each voter
    weights = pd.DataFrame(columns=["Voter", "Weight"])
    weights["Voter"] = Annotations.Voter.unique()
    for voter in list(weights["Voter"]):
        # Compute the average number of selected alternatives per question
        s = 1 / len(Questions) * np.sum(Annotations[Annotations.Voter == voter][Alternatives].to_numpy())
        # Estimate the reliability of the voter
        p = (m - 1 - s) / (m - 2)
        # Update the voter's weight
        weights.loc[weights.Voter == voter, "Weight"] = np.log(max([p, 0.0001]) / (1 - min([p, 0.9999])))

    # Aggregation
    for i in range(len(Questions)):
        # List to contain the weighted approval score of each alternative
        L = []
        for alternative in Alternatives:
            # Compute the weighted approval score of the alternative
            L += [sum([weights.loc[weights.Voter == voter, "Weight"].values[0] for voter in weights["Voter"] if
                       ((Annotations.loc[
                             (Annotations.Voter == voter) & (Annotations.Question == Questions[i]), alternative].values[
                             0] == 1))])]

        # Search for the alternative with maximum score and add it to the aggregation dataframe
        k = L.index(max(L))
        agg_weighted.loc[i] = [Questions[i]] + [t == k for t in range(0, len(Alternatives))]
    return agg_weighted


@ray.remote
# Estimate the voter's weight question-wise
def weighted_approval_qw(Annotations, data="animals"):
    """
    Takes the Annotation dataframe as input and applies weighted approval rule to all the instances. The weights are
    determined question-wise according to the estimated reliability of the voter. This reliability is estimated from
    the number of alternatives that the voter selects in each of the questions.
    :param Annotations: dataframe
    containing the answers of voters as binary vectors
    :param data: name of the dataset
    :return: agg_weighted: dataframe structured like the GroundTruth dataframe containing the aggregated answers
    """

    # Setting the dataset
    if data == "animals":
        Alternatives = ["Leopard", "Tiger", "Puma", "Jaguar", "Lion(ess)", "Cheetah"]
    elif data == "textures":
        Alternatives = ["Gravel", "Grass", "Brick", "Wood", "Sand", "Cloth"]
    else:
        Alternatives = ["Hebrew", "Russian", "Japanese", "Thai", "Chinese", "Tamil", "Latin", "Hindi"]

    # initialize the aggregation dataframe
    m = len(Alternatives)
    Questions = list(Annotations.Question.unique())
    agg_weighted = pd.DataFrame(columns=["Question"] + Alternatives)

    # Comute the weight of each voter and aggregate the answers in each question
    weights = pd.DataFrame(columns=["Voter", "Weight"])
    weights["Voter"] = Annotations.Voter.unique()
    n = len(list(weights["Voter"]))
    D = Annotations.loc[:, Alternatives].to_numpy()
    for i in range(len(Questions)):
        j = 0
        for voter in list(weights["Voter"]):

            # The number of alternatives selected by the voter in this question
            s = np.sum(D[n * i + j, :])

            # The estimated reliability of the voter in this question
            p = (m - 1 - s) / (m - 2)

            # The weight of this voter in this question
            weights.loc[weights.Voter == voter, "Weight"] = np.log(max([p, 0.0001]) / (1 - min([p, 0.9999])))
            j += 1

        # List to contain the weighted approval scores of each alternative
        L = []
        for alternative in Alternatives:

            # Compute the weighted approval score of each alternative
            L += [sum([weights.loc[weights.Voter == voter, "Weight"].values[0] for voter in weights["Voter"] if
                       ((Annotations.loc[
                             (Annotations.Voter == voter) & (Annotations.Question == Questions[i]), alternative].values[
                             0] == 1))])]

        # Search for the alternative with maximum score and add it to the aggregation dataframe
        k = L.index(max(L))
        agg_weighted.loc[i] = [Questions[i]] + [t == k for t in range(0, len(Alternatives))]
    return agg_weighted


@ray.remote
# Compute the weight of a voter according to a specified mallows noise model
def mallows_weight(Annotations, data="animals", distance="Jaccard"):
    """
    Takes the Annotation dataframe as input and applies weighted approval rule to all the instances. The weights are
    determined according to the number of alternatives that a ballot contains. These weights are the optimal weight
    when the noise model is supposed to be a Mallows noise with the correspondant distance.
    :param Annotations: dataframe containing the answers of voters as binary vectors
    :param data: name of the dataset
    :param distance: The distance of the noise model
    :return: agg_weighted: dataframe structured like the GroundTruth dataframe containing the aggregated answers
    """

    # Setting the dataset
    if data == "animals":
        Alternatives = ["Leopard", "Tiger", "Puma", "Jaguar", "Lion(ess)", "Cheetah"]
    elif data == "textures":
        Alternatives = ["Gravel", "Grass", "Brick", "Wood", "Sand", "Cloth"]
    else:
        Alternatives = ["Hebrew", "Russian", "Japanese", "Thai", "Chinese", "Tamil", "Latin", "Hindi"]

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
        j = 0
        for voter in list(weights["Voter"]):

            # Compute the number of selected alternatives by the voter for the question
            s = np.sum(D[n * i + j, :])

            # Compute the weight for each voter according to the chosen distance
            if distance == "Euclid":
                weights.loc[weights.Voter == voter, "Weight"] = np.sqrt(s + 1) - np.sqrt(s - 1)
            elif distance == "Jaccard":
                weights.loc[weights.Voter == voter, "Weight"] = 1 / s
            elif distance == "Dice":
                weights.loc[weights.Voter == voter, "Weight"] = 2 / (s + 1)
            j += 1

        # List to contain the weighted approval score of each alternative
        L = []
        for alternative in Alternatives:

            # Compute the weighted approval score of each alternative
            L += [sum([weights.loc[weights.Voter == voter, "Weight"].values[0] for voter in weights["Voter"] if
                       ((Annotations.loc[
                             (Annotations.Voter == voter) & (Annotations.Question == Questions[i]), alternative].values[
                             0] == 1))])]

        # Search for the alternative with the highest score and add it to the aggregation dataframe
        k = L.index(max(L))
        agg_weighted.loc[i] = [Questions[i]] + [t == k for t in range(0, len(Alternatives))]
    return agg_weighted
