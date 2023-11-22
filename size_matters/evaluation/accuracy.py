from random import sample
import pandas as pd
import logging
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import zero_one_loss
from tqdm import tqdm

from size_matters.aggregation.aggregators import (
    apply_condorcet_aggregator,
    apply_mallow_aggregator,
    apply_standard_approval_aggregator,
)
from size_matters.utils.inventory import COLUMNS, PLOT_OPTIONS, RULES
from size_matters.utils.utils import confidence_margin_mean

logging.basicConfig(
    level=logging.INFO, format="'%(asctime)s - %(levelname)s - %(message)s'"
)


def compare_methods(
    annotations: pd.DataFrame, groundtruth: pd.DataFrame, max_voters: int, n_batch: int
) -> NDArray:
    accuracy = np.zeros([5, n_batch, max_voters - 1])

    logging.info("Experiment started : running the different aggregators ...")

    for num in tqdm(
        range(1, max_voters), desc="Number of voters", position=0, leave=True
    ):
        for batch in tqdm(range(n_batch), desc="Batch", position=1, leave=False):
            voters = sample(
                list(annotations.index.get_level_values(COLUMNS.voter).unique()), num
            )
            annotations_batch = annotations[
                annotations.index.get_level_values(COLUMNS.voter).isin(voters)
            ]

            (
                standard_approval_labels,
                euclid_labels,
                jaccard_labels,
                dice_labels,
                condorcet_labels,
            ) = (
                apply_standard_approval_aggregator(annotations_batch),
                apply_mallow_aggregator(annotations_batch, RULES.euclid),
                apply_mallow_aggregator(annotations_batch, RULES.jaccard),
                apply_mallow_aggregator(annotations_batch, RULES.dice),
                apply_condorcet_aggregator(annotations_batch),
            )

            rules = (
                standard_approval_labels,
                euclid_labels,
                jaccard_labels,
                dice_labels,
                condorcet_labels,
            )
            for i, rule in enumerate(rules):
                accuracy[i, batch, num - 1] = 1 - zero_one_loss(
                    groundtruth.to_numpy().astype(int), rule.to_numpy().astype(int)
                )

    logging.info("Experiment completed, gathering the results ..")

    zero_one_margin = np.zeros([len(rules), max_voters - 1, 3])
    for num in range(1, max_voters):
        for i in range(len(rules)):
            zero_one_margin[i, num - 1, :] = confidence_margin_mean(
                accuracy[i, :, num - 1]
            )

    _plot_accuracies(max_voters, zero_one_margin)

    return zero_one_margin


def _plot_accuracies(max_voters: int, zero_one_margin: NDArray) -> None:
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
    plt.show()
