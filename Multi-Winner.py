import pandas as pd
import numpy as np
import random
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("TkAgg")
#import ray
from sklearn.metrics import zero_one_loss, hamming_loss

from utils import confidence_margin_mean

# Initialize ray for parallel computing
#ray.init()


def prepare_foot_data(path="Data/Multi_Winner/data_quiz_foot.csv"):
    """
        Read csv file containing ground truth and answers and returns two dataframes containing ground truth and answers
        :param path: path to csv file
        :return: GroundTruth dataframe (each row contains name of instance and a binary
        vector of belonging or not of the alternative to the ground truth), Annotations Dataframe
        """

    # Image labels
    Images = ["Question_1", "Question_2", "Question_3", "Question_4", "Question_5", "Question_6", "Question_7",
              "Question_8", "Question_9",
              "Question_10", "Question_11", "Question_12", "Question_13", "Question_14", "Question_15"]

    # Reading data from csv
    print("Getting values")
    Answers = pd.read_csv(path)
    print("Values saved")

    # Cleaning the dataframe
    Answers.columns = ["Date", "Score"] + Images + ["Voter"]
    Answers.drop(Answers[(Answers.Date == "")].index, inplace=True)
    del Answers["Score"]

    # Delete columns where groudn truth is made of single winner
    Answers.drop(columns=["Question_10", "Question_11"], inplace=True)
    Images.remove("Question_10")
    Images.remove("Question_11")

    # Create GroundTruth DataFrame
    GroundTruth = pd.DataFrame(columns=["Question", "RealMadrid", "Barcelone", "BayernMunich", "InterMilan", "PSG"])
    for i in range(len(Images)):
        C = list(Answers[Answers.Voter == "Correction987"][Images[i]])[0].split(",")
        for j in range(len(C)):
            C[j] = C[j].replace(" ", "")
        GroundTruth.loc[i] = [Images[i]] + [int("RealMadrid" in C)] + [int("Barcelone" in C)] + [
            int("BayernMunich" in C)] + [int("InterMilan" in C)] + [int("PSG" in C)]

    # Create Annotations Dataframe
    Answers.drop(Answers[(Answers.Voter == "Correction987")].index,
                 inplace=True)  # Remove the first row, contains the ground truth
    Annotations = pd.DataFrame(
        columns=["Voter", "Question", "RealMadrid", "Barcelone", "BayernMunich", "InterMilan", "PSG"])
    for voter in Answers["Voter"]:
        for image in Images:
            L = list(Answers[Answers.Voter == voter][image])[0].split(",")
            for i in range(len(L)):
                L[i] = L[i].replace(" ", "")
            tmp_row = {"Voter": voter, "Question": image, "RealMadrid": int("RealMadrid" in L),
                       "Barcelone": int("Barcelone" in L), "BayernMunich": int("BayernMunich" in L),
                       "InterMilan": int("InterMilan" in L), "PSG": int("PSG" in L)}
            Annotations = Annotations.append(tmp_row, ignore_index=True)

    # Sort rows by question
    Annotations.sort_values(by="Question", ascending=True, inplace=True)

    return GroundTruth, Annotations


################################################################################################
########################## Aggregation rules ##############################
###############################################################################################

#@ray.remote
def top_2(Annotations):
    """
    Takes the annotations dataframe as input and computes the label-wise majority rule ouput for each instance.
    :param Annotations: Annotations DataFrame
    :return: agg_majority:DataFrame containing the output of the majority rule
    """
    n = Annotations.Voter.unique().shape[0]  # Number of Voters
    Alternatives = ["RealMadrid", "Barcelone", "BayernMunich", "InterMilan", "PSG"]
    # Initialize the aggregated dataframe
    agg_majority = pd.DataFrame(columns=["Question", "RealMadrid", "Barcelone", "BayernMunich", "InterMilan", "PSG"])
    for i in range(1, 16):
        if i in [10, 11]:
            continue
        image = "Question_" + str(i)
        L = [image]
        App = []
        for team in Alternatives:
            # Select top_2 alternatives
            App += [sum(Annotations.loc[Annotations["Question"] == image, team])]
            #print(App)

        App_np = np.array(App)
        top_2_indices = (-App_np).argsort()[:2]
        L += [int(Alternatives.index(team) in top_2_indices) for team in Alternatives]
        agg_majority.loc[i] = L
        #print(L)
    return agg_majority


# Search the most frequent rule in a numpy matrix
#@ray.remote
def most_frequent_row(matrix):
    """
    Compute the rule rule outcome given a set of approval ballots.
    :param votes: an array (n x m) of n approval ballots (m sized binary line)
    :return: an array of m binary labels
    """
    a = np.ascontiguousarray(matrix)
    void_dt = np.dtype((np.void, a.dtype.itemsize * np.prod(a.shape[1:])))
    _, ids, count = np.unique(a.view(void_dt).ravel(), return_index=1, return_counts=1)
    largest_count_id = ids[count.argmax()]
    mostfrequent_row = a[largest_count_id]
    return mostfrequent_row


# Modal aggregation rule
#@ray.remote
def mode(Annotations):
    """
    Takes annotations DataFrame and outputs the outcome of the modal rule (plurality over all the approval ballots)
    :param Annotations: Annotations dataframe
    :return: agg_mode: dataframe containg the output of the modal rule for all instances
    """

    Alternatives = ["RealMadrid", "Barcelone", "BayernMunich", "InterMilan", "PSG"]

    # initialize the aggregated dataframe
    agg_mode = pd.DataFrame(columns=["Question", "RealMadrid", "Barcelone", "BayernMunich", "InterMilan", "PSG"])
    for i in range(1, 16):
        if i in [10, 11]:
            continue
        image = "Question_" + str(i)
        arr = Annotations[Annotations.Question == image][
            Alternatives].to_numpy().astype(int)

        # Most frequent ballot
        agg_mode.loc[i] = [image] + list(most_frequent_row(arr))
    return agg_mode


# Estimate the voter's weight question-wise
def weighted_approval(Annotations):
    """
    Takes the Annotation dataframe as input and applies weighted approval rule to all the instances. The weights are
    determined question-wise according to the estimated reliability of the voter. This reliability is estimated from
    the number of alternatives that the voter selects in each of the questions.
    :param Annotations: dataframe
    containing the answers of voters as binary vectors
    :param data: name of the dataset
    :return: agg_weighted: dataframe structured like the GroundTruth dataframe containing the aggregated answers
    """

    Alternatives = ["RealMadrid", "Barcelone", "BayernMunich", "InterMilan", "PSG"]

    # initialize the aggregation dataframe
    m = len(Alternatives)
    #Questions = list(Annotations.Question.unique())
    agg_weighted = pd.DataFrame(columns=["Question"] + Alternatives)

    # Comute the weight of each voter and aggregate the answers in each question
    weights = pd.DataFrame(columns=["Voter", "Weight"])
    weights["Voter"] = Annotations.Voter.unique()
    #n = len(list(weights["Voter"]))
    #D = Annotations.loc[:, Alternatives].to_numpy()
    for i in range(1,16):
        if i in [10,11]:
            continue
        Question = "Question_"+str(i)
        for voter in list(weights["Voter"]):
            row = Annotations[(Annotations["Question"] == Question) & (Annotations["Voter"] == voter)][Alternatives]
            s = row.sum(axis = 1)


            # The number of alternatives selected by the voter in this question
            #s = np.sum(D[n * i + j, :])

            # The estimated reliability of the voter in this question
            #p = (m - 1 - s) / (m - 2)
            p = 1- abs(s-2)/m
            p = p.values[0]            #p = 1- abs(s-2)/m

            # The weight of this voter in this question
            weights.loc[weights.Voter == voter, "Weight"] = np.log(max([p, 0.0001]) / (1 - min([p, 0.9999])))
            #j += 1
            #weights.loc[weights.Voter == voter, "Weight"] = 1/(2*m+abs(s-2))

        # List to contain the weighted approval scores of each alternative
        L = [Question]
        App = []
        for alternative in Alternatives:

            # Compute the weighted approval score of each alternative
            App += [sum([weights.loc[weights.Voter == voter, "Weight"].values[0] for voter in weights["Voter"] if
                       ((Annotations.loc[
                             (Annotations.Voter == voter) & (Annotations.Question == Question), alternative].values[
                             0] == 1))])]

        # Search for the alternative with maximum score and add it to the aggregation dataframe
        App_np = np.array(App)
        top_2_indices = (-App_np).argsort()[:2]
        L += [int(Alternatives.index(team) in top_2_indices) for team in Alternatives]
        agg_weighted.loc[i] = L
    return agg_weighted


###########################################################################################
############################ Compare Methods ##########################################
############################################################################################

def compare_methods_qw(n_batch=25, path="Data/Multi_Winner/data_quiz_foot.csv"):
    """
    Plots the averaged accuracy of different aggregation methods over number of batches for different number of voters.
    :param n_batch: the number of batches of voters for each number of voter.
    :param data: name of the dataset
    :return: None
    """

    Alternatives = ["RealMadrid", "Barcelone", "BayernMunich", "InterMilan", "PSG"]

    # Initialize the Annotation and GroundTruth dataframes
    GroundTruth, Anno = prepare_foot_data(path)
    # Set the maximum number of voters
    n = 65
    # initialize the accuracy array
    Acc = np.zeros([5, n_batch, n - 1])
    for num in range(1, n):
        print("Number of Voters :", num)
        for batch in range(n_batch):
            print("###### Batch ", batch, " ######")
            # Randomly sample num voters
            voters = random.sample(list(Anno.Voter.unique()), num)
            Annotations = Anno[Anno["Voter"].isin(voters)]

            # Apply the majority and different weighted rules to aggregate the answers in parallel
            #Top_2, Mode = ray.get([top_2.remote(Annotations), mode.remote(Annotations)])
            Top_2 = top_2(Annotations)
            Mode = mode(Annotations)
            Weighted_Top_2 = weighted_approval(Annotations)

            # Put results into numpy arrays
            G = GroundTruth[Alternatives].to_numpy().astype(int)
            Top_2 = Top_2[Alternatives].to_numpy().astype(int)
            Mode = Mode[Alternatives].to_numpy().astype(int)
            Weighted_Top_2 = Weighted_Top_2[Alternatives].to_numpy().astype(int)

            # Compute the accuracy of each method
            Acc[0, batch, num - 1] = 1 - zero_one_loss(G, Top_2)
            Acc[1, batch, num - 1] = 1 - zero_one_loss(G, Mode)
            Acc[2, batch, num - 1] = 1 - zero_one_loss(G, Weighted_Top_2)

    # Plot the evolution of the accuracies of the methods when the number of voters grows
    fig = plt.figure()
    Zero_one_margin = np.zeros([5, n - 1, 3])
    for num in range(1, n):
        Zero_one_margin[0, num - 1, :] = confidence_margin_mean(Acc[0, :, num - 1])
        Zero_one_margin[1, num - 1, :] = confidence_margin_mean(Acc[1, :, num - 1])
        Zero_one_margin[2, num - 1, :] = confidence_margin_mean(Acc[2, :, num - 1])

    plt.errorbar(range(1, n), Zero_one_margin[0, :, 0], label='Top_2', linestyle="solid")
    plt.fill_between(range(1, n), Zero_one_margin[0, :, 1], Zero_one_margin[0, :, 2], alpha=0.2)

    plt.errorbar(range(1, n), Zero_one_margin[1, :, 0], label='Mode', linestyle="dotted")
    plt.fill_between(range(1, n), Zero_one_margin[1, :, 1], Zero_one_margin[1, :, 2], alpha=0.2)

    plt.errorbar(range(1, n), Zero_one_margin[2, :, 0], label='Weighted', linestyle="dashed")
    plt.fill_between(range(1, n), Zero_one_margin[2, :, 1], Zero_one_margin[2, :, 2], alpha=0.2)

    plt.legend()
    plt.xlabel("Number of voters")
    plt.ylabel("Accuracy")
    plt.title("Football quiz Loss")
