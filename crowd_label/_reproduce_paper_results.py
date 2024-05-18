from pathlib import Path

import pandas as pd

from crowd_label._evaluation.accuracy import compare_methods, plot_accuracies
from crowd_label.utils.inventory import COLUMNS, DATASETS


def _process_dataset() -> None:
	dataset, max_voters, n_batch = _read_input_parameters()

	annotations, groundtruth = _load_dataset(dataset)

	accuracies = compare_methods(
		annotations=annotations,
		groundtruth=groundtruth,
		max_voters=max_voters,
		n_batch=n_batch,
	)

	plot_accuracies(accuracies)


def _read_input_parameters():
	dataset_name = input(f"Select a dataset [{'|'.join(DATASETS)}]: ")
	if dataset_name not in DATASETS:
		raise ValueError(
			f"Invalid dataset {dataset_name} choose from  [{'|'.join(DATASETS)}]"
		)

	dataset = DATASETS[dataset_name]

	max_voters = int(
		input(f"Choose the maximum number of voters, max={dataset.nbr_voters}:")
	)
	if max_voters > dataset.nbr_voters:
		raise ValueError(f"Too many voters. Don't exceed {dataset.nbr_voters}")

	n_batch = int(input("Choose the number of batches: "))
	if n_batch <= 0:
		raise ValueError("Please provide a positive number of batches")
	return dataset, max_voters, n_batch


def _load_dataset(dataset):
	annotations = pd.read_csv(
		Path(dataset.path).parent / "annotations.csv",
		index_col=[COLUMNS.question, COLUMNS.voter],
	)
	groundtruth = pd.read_csv(
		Path(dataset.path).parent / "ground_truth.csv", index_col=[COLUMNS.question]
	)

	return annotations, groundtruth


if __name__ == "__main__":  # pragma: no cover
	_process_dataset()
