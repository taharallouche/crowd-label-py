import logging
from collections.abc import Iterable
from random import sample
from typing import Mapping

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.metrics import accuracy_score
from tqdm import tqdm

from crowd_label.aggregation.aggregators import (
	Aggregator,
	CondorcetAggregator,
	DiceAggregator,
	EuclidAggregator,
	JaccardAggregator,
	StandardApprovalAggregator,
)
from crowd_label.utils.inventory import COLUMNS
from crowd_label.utils.utils import get_mean_confidence_interval

logging.basicConfig(
	level=logging.INFO, format="'%(asctime)s - %(levelname)s - %(message)s'"
)


def compare_methods(
	annotations: pd.DataFrame,
	groundtruth: pd.DataFrame,
	max_voters: int,
	n_batch: int,
	aggregators: Iterable[Aggregator] = (
		StandardApprovalAggregator(),
		EuclidAggregator(),
		JaccardAggregator(),
		DiceAggregator(),
		CondorcetAggregator(),
	),
) -> dict[str, NDArray]:
	accuracy = {
		str(aggregator): np.zeros([n_batch, max_voters - 1])
		for aggregator in aggregators
	}
	confidence_intervals = {
		str(aggregator): np.zeros([max_voters - 1, 3]) for aggregator in aggregators
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

			for aggregator in aggregators:
				aggregated_labels = aggregator.aggregate(annotations_batch)
				accuracy[str(aggregator)][batch, num - 1] = accuracy_score(
					groundtruth, aggregated_labels
				)

		for aggregator in aggregators:
			confidence_intervals[str(aggregator)][num - 1, :] = (
				get_mean_confidence_interval(accuracy[str(aggregator)][:, num - 1])
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
	plt.show()
