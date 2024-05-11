from pathlib import Path

import pandas as pd

from crowd_label._evaluation.accuracy import compare_methods
from crowd_label.utils.inventory import COLUMNS, DATASETS

if __name__ == "__main__":  # pragma: no cover
	dataset_name = input(f"Select a dataset [{'|'.join(DATASETS)}]: ")
	assert dataset_name in DATASETS, "Invalid dataset"
	dataset = DATASETS[dataset_name]

	max_voters = int(
		input(f"Choose the maximum number of voters, max={dataset.nbr_voters}:")
	)
	assert max_voters <= dataset.nbr_voters, "Too many voters"

	n_batch = int(input("Choose the number of batches: "))
	assert n_batch > 0, "Please choose a positive number of batches"

	annotations = pd.read_csv(
		Path(dataset.path).parent / "annotations.csv",
		index_col=[COLUMNS.question, COLUMNS.voter],
	)
	groundtruth = pd.read_csv(
		Path(dataset.path).parent / "ground_truth.csv", index_col=[COLUMNS.question]
	)

	compare_methods(
		annotations=annotations,
		groundtruth=groundtruth,
		max_voters=max_voters,
		n_batch=n_batch,
	)
