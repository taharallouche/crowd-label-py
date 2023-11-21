from random import sample
import logging
import matplotlib.pyplot as plt
import numpy as np
import ray
from numpy.typing import NDArray
from sklearn.metrics import zero_one_loss
from tqdm import tqdm

from size_matters.aggregation.aggregators import (
    apply_condorcet_aggregator,
    apply_mallow_aggregator,
    apply_standard_approval_aggregator,
)
from size_matters.parsing.data_preparation import prepare_data
from size_matters.utils.inventory import COLUMNS, PLOT_OPTIONS, RULES, Dataset
from size_matters.utils.utils import confidence_margin_mean

logging.basicConfig(
    level=logging.INFO, format="'%(asctime)s - %(levelname)s - %(message)s'"
)


def compare_methods(dataset: Dataset, max_voters: int, n_batch: int) -> NDArray:
    """
    Plots the averaged accuracy over number of batches.
    :param n_batch: the number of batches of voters for each number of voter.
    :param data: name of the dataset
    :return: zero_one_margin: the accuracy of each method
    """

    alternatives = dataset.alternatives
    annotations, groundtruth = prepare_data(dataset)

    # Set the maximum number of voters
    max_voters = max_voters

    # initialize the accuracy array
    accuracy = np.zeros([5, n_batch, max_voters - 1])

    logging.info("Vote started : running the different methods ")
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
                    apply_standard_approval_aggregator.remote(
                        annotations_batch, dataset
                    ),
                    apply_mallow_aggregator.remote(
                        annotations_batch, dataset, RULES.euclid
                    ),
                    apply_mallow_aggregator.remote(
                        annotations_batch, dataset, RULES.jaccard
                    ),
                    apply_mallow_aggregator.remote(
                        annotations_batch, dataset, RULES.dice
                    ),
                    apply_condorcet_aggregator.remote(annotations_batch),
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
    logging.info("Vote completed")
    zero_one_margin = np.zeros([len(rules), max_voters - 1, 3])
    for num in range(1, max_voters):
        for i in range(len(rules)):
            zero_one_margin[i, num - 1, :] = confidence_margin_mean(
                accuracy[i, :, num - 1]
            )

    _plot_accuracies(dataset, max_voters, zero_one_margin)

    return zero_one_margin


def _plot_accuracies(
    dataset: Dataset, max_voters: int, zero_one_margin: NDArray
) -> None:
    fig = plt.figure()  # noqa: unused

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
