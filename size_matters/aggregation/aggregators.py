import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from size_matters.utils.inventory import COLUMNS, RELIABILITY_BOUNDS
from typing import Callable


class Aggregator(ABC):
    _type: str

    @abstractmethod
    def aggregate(self, annotations: pd.DataFrame, **kwargs) -> pd.DataFrame:
        pass

    @property
    def type(self) -> str:
        return self._type

    @staticmethod
    def _get_aggregated_labels(
        annotations: pd.DataFrame, weights: pd.Series | None = None
    ) -> pd.DataFrame:
        weighted_answers = annotations

        if weights is not None:
            weighted_answers = annotations.multiply(weights, axis="index")

        weighted_scores = weighted_answers.groupby(COLUMNS.question, sort=False)[
            annotations.columns
        ].sum()

        weighted_scores = weighted_scores.reindex(
            annotations.index.get_level_values(COLUMNS.question).unique()
        )

        winning_alternatives = pd.Categorical(
            weighted_scores.idxmax(axis=1),
            categories=annotations.columns,
            ordered=True,
        )

        aggregated_labels = pd.get_dummies(
            winning_alternatives, columns=annotations.columns
        )

        return aggregated_labels


class StandardApprovalAggregator(Aggregator):
    _type: str = "Standard Approval Voting"

    def aggregate(self, annotations: pd.DataFrame, **kwargs) -> pd.DataFrame:
        return self._get_aggregated_labels(annotations)


class CondorcetAggregator(Aggregator):
    _type: str = "Condorcet Voting"

    def aggregate(self, annotations: pd.DataFrame, **kwargs) -> pd.DataFrame:
        vote_size = annotations.sum(axis=1)
        reliabilities = (len(annotations.columns) - vote_size - 1) / (
            len(annotations.columns) - 2
        )
        reliabilities = reliabilities.clip(
            RELIABILITY_BOUNDS.lower, RELIABILITY_BOUNDS.upper
        )
        weights = np.log(reliabilities / (1 - reliabilities))

        return self._get_aggregated_labels(annotations, weights)


class MallowAggregator(Aggregator):
    @property
    @abstractmethod
    def _weight_calculator(self) -> Callable[[pd.Series], pd.Series]:
        pass

    def aggregate(self, annotations: pd.DataFrame, **kwargs) -> pd.DataFrame:
        vote_size = annotations.sum(axis=1)
        weights = type(self)._weight_calculator(vote_size)

        return self._get_aggregated_labels(annotations, weights)


class EuclidAggregator(MallowAggregator):
    _type: str = "Euclidean Mallow Voting"
    _weight_calculator = lambda vote_size: np.sqrt(  # noqa : E731
        vote_size + 1
    ) - np.sqrt(vote_size - 1)


class JaccardAggregator(MallowAggregator):
    _type: str = "Jaccard Mallow Voting"
    _weight_calculator = lambda vote_size: 1 / vote_size  # noqa : E731


class DiceAggregator(MallowAggregator):
    _type: str = "Dice Mallow Voting"
    _weight_calculator = lambda vote_size: 2 / (vote_size + 1)  # noqa : E731
