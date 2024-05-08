import logging
from random import sample

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.metrics import accuracy_score
from tqdm import tqdm

from size_matters.aggregation.aggregators import (
	Aggregator,
	CondorcetAggregator,
	DiceAggregator,
	EuclidAggregator,
	JaccardAggregator,
	StandardApprovalAggregator,
)
from size_matters.utils.inventory import COLUMNS, PLOT_OPTIONS
from size_matters.utils.utils import confidence_margin_mean

logging.basicConfig(
	level=logging.INFO, format="'%(asctime)s - %(levelname)s - %(message)s'"
)


def compare_methods(
	annotations: pd.DataFrame, groundtruth: pd.DataFrame, max_voters: int, n_batch: int
) -> NDArray:
	accuracy = np.zeros([5, n_batch, max_voters - 1])
	aggregators: list[Aggregator] = [
		StandardApprovalAggregator(),
		EuclidAggregator(),
		JaccardAggregator(),
		DiceAggregator(),
		CondorcetAggregator(),
	]

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

			for i, aggregator in enumerate(aggregators):
				aggregated_labels = aggregator.aggregate(annotations_batch)
				accuracy[i, batch, num - 1] = accuracy_score(
					groundtruth, aggregated_labels
				)

	logging.info("Experiment completed, gathering the results ..")

	zero_one_margin = np.zeros([len(aggregators), max_voters - 1, 3])
	for num in range(1, max_voters):
		for i in range(len(aggregators)):
			zero_one_margin[i, num - 1, :] = confidence_margin_mean(
				accuracy[i, :, num - 1]
			)

	_plot_accuracies(max_voters, zero_one_margin)

	return zero_one_margin


def _plot_accuracies(max_voters: int, zero_one_margin: NDArray) -> None:
	fig = plt.figure()  # noqa: F841

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
