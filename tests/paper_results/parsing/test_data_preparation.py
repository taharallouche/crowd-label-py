from pathlib import Path

import pandas as pd
import pytest

from hakeem.core.utils.inventory import COLUMNS


@pytest.mark.e2e
@pytest.mark.parametrize("dataset", ["animals", "languages", "textures"])
def test_prepare_data(dataset) -> None:
	# With
	from hakeem.core.utils.inventory import DATASETS
	from hakeem.paper_results.parsing.data_preparation import prepare_data

	dataset = DATASETS[dataset]
	expected_ground_truth = pd.read_csv(
		Path(dataset.path).parent / "ground_truth.csv", index_col=[COLUMNS.question]
	)
	expected_annotations = pd.read_csv(
		Path(dataset.path).parent / "annotations.csv",
		index_col=[COLUMNS.question, COLUMNS.voter],
	)

	# When
	annotations, groundtruth = prepare_data(dataset)

	# Then
	pd.testing.assert_frame_equal(groundtruth, expected_ground_truth, check_dtype=False)
	pd.testing.assert_frame_equal(annotations, expected_annotations, check_dtype=False)
