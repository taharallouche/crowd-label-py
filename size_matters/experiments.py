from random import sample

import matplotlib.pyplot as plt
import numpy as np
import ray
from tqdm import tqdm
from sklearn.metrics import zero_one_loss

from size_matters.aggregation_rules import (
    mallows_weight,
    standard_approval_voting,
    weighted_approval_qw,
)
from size_matters.data_preparation import prepare_data
from size_matters.inventory import (
    COLUMNS,
    DATASETS,
    PLOT_OPTIONS,
    RULES,
    Dataset,
)
from size_matters.utils import confidence_margin_mean


def compare_methods(dataset: Dataset, max_voters: int, n_batch: int) -> None:
    """
    Plots the averaged accuracy over number of batches.
    :param n_batch: the number of batches of voters for each number of voter.
    :param data: name of the dataset
    :return: None
    """

    alternatives = dataset.alternatives
    annotations, groundtruth = prepare_data(dataset)

    # Set the maximum number of voters
    max_voters = max_voters

    # initialize the accuracy array
    accuracy = np.zeros([5, n_batch, max_voters - 1])
    for num in tqdm(
        range(1, max_voters), desc="Number of voters", position=0, leave=True
    ):
        for batch in tqdm(range(n_batch), desc="Batch", position=1, leave=False):
            # Randomly sample num voters

            voters = sample(list(annotations[COLUMNS.voter].unique()), num)
            annotations_batch = annotations[annotations[COLUMNS.voter].isin(voters)]

            # Apply rules to aggregate the answers in parallel
            (
                standard_approval,
                weight_sqrt_ham,
                weight_jaccard,
                weight_dice,
                weight_qw,
            ) = ray.get(
                [
                    standard_approval_voting.remote(annotations_batch, dataset),
                    mallows_weight.remote(annotations_batch, dataset, RULES.euclid),
                    mallows_weight.remote(annotations_batch, dataset, RULES.jaccard),
                    mallows_weight.remote(annotations_batch, dataset, RULES.dice),
                    weighted_approval_qw.remote(annotations_batch, dataset),
                ]
            )

            # Put results into numpy arrays
            G = groundtruth[alternatives].to_numpy().astype(int)
            Weight_sqrt_ham = weight_sqrt_ham[alternatives].to_numpy().astype(int)
            Weight_jaccard = weight_jaccard[alternatives].to_numpy().astype(int)
            Weight_dice = weight_dice[alternatives].to_numpy().astype(int)
            Weight_qw = weight_qw[alternatives].to_numpy().astype(int)
            standard_approval = standard_approval[alternatives].to_numpy().astype(int)

            # Compute the accuracy of each method
            rules = (
                standard_approval,
                Weight_sqrt_ham,
                Weight_jaccard,
                Weight_dice,
                Weight_qw,
            )
            for i, rule in enumerate(rules):
                accuracy[i, batch, num - 1] = 1 - zero_one_loss(G, rule)

    zero_one_margin = np.zeros([len(rules), max_voters - 1, 3])
    for num in range(1, max_voters):
        for i in range(len(rules)):
            zero_one_margin[i, num - 1, :] = confidence_margin_mean(
                accuracy[i, :, num - 1]
            )

    # Plot the accuracies of the methods when the number of voters grows
    fig = plt.figure()  # noqa

    for rule, options in PLOT_OPTIONS.items():
        plt.errorbar(
            range(1, max_voters),
            zero_one_margin[options["index"], :, 0],
            label=rule,
            linestyle=options["linestyle"],
        )
        plt.fill_between(
            range(1, max_voters),
            zero_one_margin[options["index"], :, 1],
            zero_one_margin[options["index"], :, 2],
            alpha=0.2,
        )

    plt.legend()
    plt.xlabel("Number of voters")
    plt.ylabel("Accuracy")
    plt.title(dataset.name)
    plt.show()

    return zero_one_margin


if __name__ == "__main__":  # pragma: no cover
    ray.init()
    dataset_name = input("Select a dataset [animals|textures|languages]: ")
    dataset = DATASETS[dataset_name]
    max_voters = int(
        input(f"Choose the maximum number of voters, max={dataset.nbr_voters}:")
    )
    assert max_voters <= dataset.nbr_voters, "Too many voters"
    n_batch = int(input("Choose the number of batches: "))
    compare_methods(dataset=dataset, max_voters=max_voters, n_batch=n_batch)
