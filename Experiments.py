# import dependencies
import numpy as np
from sklearn.metrics import zero_one_loss, hamming_loss
import random
import matplotlib.pyplot as plt
import ray

# Import functions from files
from Data_preparation import prepare_data
from aggregation_rules import simple_approval, mallows_weight, weighted_approval_qw
from utils import confidence_margin_mean

# Initialize ray for parallel computing
ray.init()


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

            # Apply the majority and different weighted rules to aggregate the answers in parallel
            maj, weight_sqrt_ham, weight_jaccard, weight_dice, weight_qw = ray.get(
                [simple_approval.remote(Annotations, data), mallows_weight.remote(Annotations, data, "Euclid"),
                 mallows_weight.remote(Annotations, data, "Jaccard"), mallows_weight.remote(Annotations, data, "Dice"),
                 weighted_approval_qw.remote(Annotations, data)])

            # Put results into numpy arrays
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



