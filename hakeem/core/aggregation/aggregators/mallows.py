import numpy as np
import pandas as pd

from hakeem.core.aggregation.base import WeightedAggregator


class StandardApprovalAggregator(WeightedAggregator):
    @staticmethod
    def compute_weights(annotations: pd.DataFrame) -> pd.Series:
        return pd.Series(1, index=annotations.index)


class EuclidAggregator(WeightedAggregator):
    @staticmethod
    def compute_weights(annotations: pd.DataFrame) -> pd.Series:
        vote_size = annotations.sum(axis=1)
        return np.sqrt(vote_size + 1) - np.sqrt(vote_size - 1)


class JaccardAggregator(WeightedAggregator):
    @staticmethod
    def compute_weights(annotations: pd.DataFrame) -> pd.Series:
        vote_size = annotations.sum(axis=1)
        assert np.all(vote_size > 0), "Jaccard weights are not defined for empty votes."
        return 1 / vote_size


class DiceAggregator(WeightedAggregator):
    @staticmethod
    def compute_weights(annotations: pd.DataFrame) -> pd.Series:
        vote_size = annotations.sum(axis=1)
        return 2 / (vote_size + 1)
