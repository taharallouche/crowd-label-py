import numpy as np
import pandas as pd

from hakeem.aggregation.base import WeightedAggregator
from hakeem.utils.inventory import COLUMNS, DEFAULT_RELIABILITY_BOUNDS


class CondorcetAggregator(WeightedAggregator):
    def __init__(
        self,
        lower_reliability_bound: float = DEFAULT_RELIABILITY_BOUNDS.lower,
        upper_reliability_bound: float = DEFAULT_RELIABILITY_BOUNDS.upper,
        task_column: str = COLUMNS.question,
        worker_column: str = COLUMNS.voter,
    ):
        super().__init__(task_column, worker_column)
        self.lower_reliability_bound = lower_reliability_bound
        self.upper_reliability_bound = upper_reliability_bound

    def compute_weights(self, annotations: pd.DataFrame) -> pd.Series:
        reliabilities = self._compute_reliabilities(annotations)
        assert np.all(
            (reliabilities > 0) & (reliabilities < 1)
        ), "Reliabilities must be in (0, 1)."

        return np.log(reliabilities / (1 - reliabilities))

    def _compute_reliabilities(self, annotations: pd.DataFrame) -> pd.Series:
        vote_size = annotations.sum(axis=1)

        assert len(annotations.columns) > 2, "At least 3 labels are required currently."
        reliabilities = (len(annotations.columns) - vote_size - 1) / (
            len(annotations.columns) - 2
        )
        reliabilities = reliabilities.clip(
            self.lower_reliability_bound, self.upper_reliability_bound
        )
        return reliabilities
