from unittest.mock import MagicMock, patch

import numpy as np
import pytest


@pytest.mark.ut
@patch("size_matters.experiments.plt")
def test_compare_methods(mock_plt: MagicMock) -> None:
    # Given
    import random

    import ray

    from size_matters.experiments import compare_methods
    from size_matters.inventory import DATASETS

    ray.init()

    random.seed(42)
    dataset = DATASETS["animals"]
    max_voters = 5
    n_batch = 5
    expected_output = np.array(
        [
            [
                [0.5375, 0.35884218, 0.71615782],
                [0.6625, 0.52152533, 0.80347467],
                [0.8, 0.68489467, 0.91510533],
                [0.6375, 0.42215764, 0.85284236],
            ],
            [
                [0.5375, 0.35884218, 0.71615782],
                [0.6625, 0.52152533, 0.80347467],
                [0.8, 0.65072573, 0.94927427],
                [0.65, 0.41978933, 0.88021067],
            ],
            [
                [0.5375, 0.35884218, 0.71615782],
                [0.6625, 0.52152533, 0.80347467],
                [0.8, 0.65072573, 0.94927427],
                [0.65, 0.41978933, 0.88021067],
            ],
            [
                [0.5375, 0.35884218, 0.71615782],
                [0.6625, 0.52152533, 0.80347467],
                [0.8, 0.65072573, 0.94927427],
                [0.65, 0.41978933, 0.88021067],
            ],
            [
                [0.525, 0.33810482, 0.71189518],
                [0.5875, 0.42657692, 0.74842308],
                [0.7625, 0.54027596, 0.98472404],
                [0.575, 0.32115069, 0.82884931],
            ],
        ]
    )

    # When
    result = compare_methods(dataset, max_voters, n_batch)

    # Then
    np.testing.assert_allclose(result, expected_output, rtol=1e-6)
