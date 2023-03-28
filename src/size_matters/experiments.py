import random

import matplotlib.pyplot as plt
import numpy as np
import ray
from sklearn.metrics import zero_one_loss

from size_matters.aggregation_rules import (
    mallows_weight,
    simple_approval,
    weighted_approval_qw,
)
from size_matters.data_preparation import prepare_data
from size_matters.inventory import DataInfos, data_infos
from size_matters.utils import confidence_margin_mean

# Initialize ray for parallel computing
ray.init()


def compare_methods(n_batch: int, dataset_info: DataInfos) -> None:
    """
    Plots the averaged accuracy over number of batches.
    :param n_batch: the number of batches of voters for each number of voter.
    :param data: name of the dataset
    :return: None
    """

    Alternatives = dataset_info.alternatives

    # Initialize the Annotation and GroundTruth dataframes
    Anno, GroundTruth = prepare_data(dataset_info)

    # Set the maximum number of voters
    n = 5

    # initialize the accuracy array
    Acc = np.zeros([5, n_batch, n - 1])
    for num in range(1, n):
        print("Number of Voters :", num)
        for batch in range(n_batch):
            print("###### Batch ", batch, " ######")

            # Randomly sample num voters
            voters = random.sample(list(Anno.Voter.unique()), num)
            Annotations = Anno[Anno["Voter"].isin(voters)]

            # Apply rules to aggregate the answers in parallel
            (
                maj,
                weight_sqrt_ham,
                weight_jaccard,
                weight_dice,
                weight_qw,
            ) = ray.get(
                [
                    simple_approval.remote(Annotations, dataset_info),
                    mallows_weight.remote(Annotations, dataset_info, "Euclid"),
                    mallows_weight.remote(
                        Annotations, dataset_info, "Jaccard"
                    ),
                    mallows_weight.remote(Annotations, dataset_info, "Dice"),
                    weighted_approval_qw.remote(Annotations, dataset_info),
                ]
            )

            # Put results into numpy arrays
            G = GroundTruth[Alternatives].to_numpy().astype(int)
            Weight_sqrt_ham = (
                weight_sqrt_ham[Alternatives].to_numpy().astype(int)
            )
            Weight_jaccard = (
                weight_jaccard[Alternatives].to_numpy().astype(int)
            )
            Weight_dice = weight_dice[Alternatives].to_numpy().astype(int)
            Weight_qw = weight_qw[Alternatives].to_numpy().astype(int)
            Maj = maj[Alternatives].to_numpy().astype(int)

            # Compute the accuracy of each method
            methods = (
                Maj,
                Weight_sqrt_ham,
                Weight_jaccard,
                Weight_dice,
                Weight_qw,
            )
            for i, method in enumerate(methods):
                Acc[i, batch, num - 1] = 1 - zero_one_loss(G, method)

    # Plot the accuracies of the methods when the number of voters grows
    fig = plt.figure()  # noqa
    Zero_one_margin = np.zeros([5, n - 1, 3])
    for num in range(1, n):
        for i in range(len(methods)):
            Zero_one_margin[i, num - 1, :] = confidence_margin_mean(
                Acc[i, :, num - 1]
            )

    plot_otions = {
        "SAV": {"linestyle": "solid", "index": 0},
        "Euclid": {"linestyle": "dashdot", "index": 1},
        "Jaccard": {"linestyle": "dashed", "index": 2},
        "Dice": {"linestyle": (0, (3, 5, 1, 5)), "index": 3},
        "Condorcet": {"linestyle": "dotted", "index": 4},
    }

    for method, options in plot_otions.items():
        plt.errorbar(
            range(1, n),
            Zero_one_margin[options["index"], :, 0],
            label=method,
            linestyle=options["linestyle"],
        )
        plt.fill_between(
            range(1, n),
            Zero_one_margin[options["index"], :, 1],
            Zero_one_margin[options["index"], :, 2],
            alpha=0.2,
        )

    plt.legend()
    plt.xlabel("Number of voters")
    plt.ylabel("Accuracy")
    plt.title(dataset_info.name)
    plt.show()


if __name__ == "__main__":
    dataset_name = input("Select a dataset [animals|textures|languages]: ")
    dataset_info = data_infos[dataset_name]
    n_batch = int(input("Choose the number of batches: "))
    compare_methods(n_batch, dataset_info)
