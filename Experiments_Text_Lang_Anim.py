import pandas as pd
import numpy as np
from sklearn.metrics import zero_one_loss, hamming_loss
import scipy.stats
import random
import matplotlib.pyplot as plt
import ray

# Initialize ray for parallel computing
ray.init()


def prepare_data(data="animals"):
    """
    This function prepares two dataframes: one containing the ground truths of the instances and one containing the
    annotations. Each row contains the question, the voter, and a binary vector whose coordinates equal one if and
    only if the associated alternative is selected by the voter.
     :param data: name of the dataset: "animals","textures" or "languages".
     :return: Annotations and GroundTruths dataframes
    """

    # Setting the path to the dataset and some of its properties
    if data == "animals":
        path = "Data/Data_Shah/data_animals.csv"
        nbr_questions = 16
        Alternatives = ["Leopard", "Tiger", "Puma", "Jaguar", "Lion(ess)", "Cheetah"]
    elif data == "textures":
        path = "Data/Data_Shah/data_textures.csv"
        nbr_questions = 16
        Alternatives = ["Gravel", "Grass", "Brick", "Wood", "Sand", "Cloth"]
    else:
        path = "Data/Data_Shah/data_languages.csv"
        nbr_questions = 25
        Alternatives = ["Hebrew", "Russian", "Japanese", "Thai", "Chinese", "Tamil", "Latin", "Hindi"]

    # Reading Dataset
    Data_brut = pd.read_csv(path, delimiter=',', index_col=False, header=0,
                            names=["Interface", "Mechanism"] + ["Question" + str(i) for i in
                                                                range(0, nbr_questions)] + [
                                      "TrueAnswer" + str(i) for i in range(0, nbr_questions)] + ["Answer" + str(i) for i
                                                                                                 in
                                                                                                 range(0,
                                                                                                       nbr_questions)] + [
                                      "Comments"],
                            usecols=["Interface"] + ["Question" + str(i) for i in range(0, nbr_questions)] + [
                                "TrueAnswer" + str(i) for i in range(0, nbr_questions)] + ["Answer" + str(i) for i in
                                                                                           range(0, nbr_questions)])
    Data_brut = Data_brut.loc[Data_brut.Interface == "subset"]
    del Data_brut["Interface"]

    # Preparing GroundTruth Dataframe
    Questions = Data_brut.iloc[0, 0:nbr_questions].to_numpy()
    GroundTruth = pd.DataFrame(columns=["Question"] + Alternatives)
    for i in range(len(Questions)):
        L = Data_brut.iloc[0, i + nbr_questions]
        row = {"Question": Questions[i]}
        for alternative in Alternatives:
            row[alternative] = int(alternative == L)
        GroundTruth = GroundTruth.append(row, ignore_index=True)

    # Preparing Annotations Dataframe
    Annotations = pd.DataFrame(
        columns=["Voter", "Question"] + Alternatives)
    for i in range(len(Questions)):
        for j in range(Data_brut.shape[0]):
            col = 0
            for c in range(0, nbr_questions):
                if Data_brut.iloc[j, c] == Questions[i]:
                    break
                else:
                    col += 1
            L = Data_brut.iloc[j, col + 2 * nbr_questions].split("|")
            row = {"Voter": j, "Question": Questions[i]}
            for alternative in Alternatives:
                row[alternative] = int(alternative in L)
            Annotations = Annotations.append(row, ignore_index=True)

    return Annotations, GroundTruth


@ray.remote
def majority(Annotations, data="animals"):
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
        L = []  # List that will contains the number of approval each alternative gets
        for alternative in Alternatives:
            # Compute the number of approvals for each alternative
            L += [int(sum(Annotations.loc[Annotations["Question"] == Questions[i], alternative]))]
        # Search for the alternative with maximum approvals and add it to the aggregation dataframe
        k = L.index(max(L))
        agg_majority.loc[i] = [Questions[i]] + [t == k for t in range(0, len(Alternatives))]
    return agg_majority


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
        L = []  # List to contain the weighted approval score of each alternative
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

        L = []  # List to contain the weighted approval scores of each alternative
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
        L = []  # List to contain the weighted approval score of each alternative
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


def compare_methods_qw(n_batch=25, data="animals"):
    """
    Plots the averaged accuracy of different aggregation methods over number of batches for different number of voters.
    :param n_batch: the number of batches of voters for each number of voter.
    :param data: name of the dataset
    :return: None
    """
    if data == "animals":
        Alternatives = ["Leopard", "Tiger", "Puma", "Jaguar", "Lion(ess)", "Cheetah"]
    elif data == "textures":
        Alternatives = ["Gravel", "Grass", "Brick", "Wood", "Sand", "Cloth"]
    else:
        Alternatives = ["Hebrew", "Russian", "Japanese", "Thai", "Chinese", "Tamil", "Latin", "Hindi"]

    # Initialize the Annotation and GroundTruth dataframes
    Anno, GroundTruth = prepare_data(data)
    # Set the maximum number of voters
    n = 100
    # initialize the accuracy array
    Acc = np.zeros([5, n_batch, n - 1])
    for num in range(1, n):
        print("Number of Voters :", num)
        for batch in range(n_batch):
            print("###### Batch ", batch, " ######")
            # Randomly sample num voters
            voters = random.sample(list(Anno.Voter.unique()), num)
            Annotations = Anno[Anno["Voter"].isin(voters)]

            # Apply the majority and different weighted rules to aggregate the answers
            maj, weight_sqrt_ham, weight_jaccard, weight_dice, weight_qw = ray.get(
                [majority.remote(Annotations, data), mallows_weight.remote(Annotations, data, "Euclid"),
                 mallows_weight.remote(Annotations, data, "Jaccard"), mallows_weight.remote(Annotations, data, "Dice"),
                 weighted_approval_qw.remote(Annotations, data)])
            G = GroundTruth[Alternatives].to_numpy().astype(int)
            Weight_sqrt_ham = weight_sqrt_ham[Alternatives].to_numpy().astype(int)
            Weight_jaccard = weight_jaccard[Alternatives].to_numpy().astype(int)
            Weight_dice = weight_dice[Alternatives].to_numpy().astype(int)
            Weight_qw = weight_qw[Alternatives].to_numpy().astype(int)
            Maj = maj[Alternatives].to_numpy().astype(int)

            # Compute the accuracy of each method
            Acc[0, batch, num - 1] = 1 - zero_one_loss(G, Maj)
            Acc[1, batch, num - 1] = 1 - zero_one_loss(G, Weight_sqrt_ham)
            Acc[2, batch, num - 1] = 1 - zero_one_loss(G, Weight_jaccard)
            Acc[3, batch, num - 1] = 1 - zero_one_loss(G, Weight_dice)
            Acc[4, batch, num - 1] = 1 - zero_one_loss(G, Weight_qw)

    # Plot the evolution of the accuracies of the methods when the number of voters grows
    fig = plt.figure()
    Zero_one_margin = np.zeros([5, n - 1, 3])
    for num in range(1, n):
        Zero_one_margin[0, num - 1, :] = confidence_margin_mean(Acc[0, :, num - 1])
        Zero_one_margin[1, num - 1, :] = confidence_margin_mean(Acc[1, :, num - 1])
        Zero_one_margin[2, num - 1, :] = confidence_margin_mean(Acc[2, :, num - 1])
        Zero_one_margin[3, num - 1, :] = confidence_margin_mean(Acc[3, :, num - 1])
        Zero_one_margin[4, num - 1, :] = confidence_margin_mean(Acc[4, :, num - 1])

    plt.errorbar(range(1, n), Zero_one_margin[0, :, 0], label='Simple', linestyle="solid")
    plt.fill_between(range(1, n), Zero_one_margin[0, :, 1], Zero_one_margin[0, :, 2], alpha=0.2)

    plt.errorbar(range(1, n), Zero_one_margin[4, :, 0], label='Condorcet', linestyle="dotted")
    plt.fill_between(range(1, n), Zero_one_margin[4, :, 1], Zero_one_margin[4, :, 2], alpha=0.2)

    plt.errorbar(range(1, n), Zero_one_margin[1, :, 0], label="Euclid", linestyle="dashdot")
    plt.fill_between(range(1, n), Zero_one_margin[1, :, 1], Zero_one_margin[1, :, 2], alpha=0.2)

    plt.errorbar(range(1, n), Zero_one_margin[2, :, 0], label="Jaccard", linestyle="dashed")
    plt.fill_between(range(1, n), Zero_one_margin[2, :, 1], Zero_one_margin[2, :, 2], alpha=0.2)

    plt.errorbar(range(1, n), Zero_one_margin[3, :, 0], label='Dice', linestyle=(0, (3, 5, 1, 5)))
    plt.fill_between(range(1, n), Zero_one_margin[3, :, 1], Zero_one_margin[3, :, 2], alpha=0.2)

    plt.legend()
    plt.xlabel("Number of voters")
    plt.ylabel("Accuracy")
    plt.title(data)


def confidence_margin_mean(data, confidence=0.95):
    """
    Given sampled data and desired confidence level, return the mean and the bounds of the 95% confidence interval
    :param data: sampled data
    :param confidence: desired level of confidence
    :return: mean, lower bound of the CI, upper bound of the CI
    """
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n - 1)
    return m, m - h, m + h
