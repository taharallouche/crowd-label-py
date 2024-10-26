import logging
from random import sample
from typing import Mapping

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.metrics import accuracy_score
from tqdm import tqdm

from hakeem.core.aggregation.aggregators.condorcet import (
    CondorcetAggregator,
)
from hakeem.core.aggregation.aggregators.mallows import (
    DiceAggregator,
    EuclidAggregator,
    JaccardAggregator,
    StandardApprovalAggregator,
)
from hakeem.core.aggregation.base import Aggregator
from hakeem.paper_results.evaluation.utils import get_mean_confidence_interval
from hakeem.paper_results.inventory import COLUMNS

logging.basicConfig(
    level=logging.INFO, format="'%(asctime)s - %(levelname)s - %(message)s'"
)


def compare_methods(
    annotations: pd.DataFrame,
    groundtruth: pd.DataFrame,
    max_voters: int,
    n_batch: int,
    aggregators: Mapping[str, Aggregator] = {
        "Standard Approval Aggregator": StandardApprovalAggregator(),
        "Euclidean Mallow Aggregator": EuclidAggregator(),
        "Jaccard Mallow Aggregator": JaccardAggregator(),
        "Dice Mallow Aggregator": DiceAggregator(),
        "Condorcet Aggregator": CondorcetAggregator(),
    },
) -> dict[str, NDArray]:
    accuracy = {
        aggregator: np.zeros([n_batch, max_voters - 1]) for aggregator in aggregators
    }
    confidence_intervals = {
        aggregator: np.zeros([max_voters - 1, 3]) for aggregator in aggregators
    }

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

            for name, aggregator in aggregators.items():
                aggregated_labels = aggregator.fit_predict(annotations_batch)
                accuracy[name][batch, num - 1] = accuracy_score(
                    groundtruth, aggregated_labels
                )

        for name in aggregators:
            confidence_intervals[name][num - 1, :] = get_mean_confidence_interval(
                accuracy[name][:, num - 1]
            )

    logging.info("Experiment completed, gathering the results ..")

    return confidence_intervals


def plot_accuracies(confidence_intervals: Mapping[str, NDArray]) -> None:
    fig = plt.figure()  # noqa: F841
    x_limit = (
        max(accuracies.shape[0] for accuracies in confidence_intervals.values()) + 1
    )

    for aggregator, confidence_interval in confidence_intervals.items():
        plt.errorbar(
            range(1, x_limit),
            confidence_interval[:, 0],
            label=aggregator,
        )
        plt.fill_between(
            range(1, x_limit),
            confidence_interval[:, 1],
            confidence_interval[:, 2],
            alpha=0.2,
        )

    plt.legend()
    plt.xlabel("Number of voters")
    plt.ylabel("Accuracy")
    plt.savefig("results.png")
