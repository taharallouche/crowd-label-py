from abc import ABC, abstractmethod

import pandas as pd

from hakeem.core.utils.coerce import coerce_schema
from hakeem.core.utils.inventory import COLUMNS


class Aggregator(ABC):
    def __init__(
        self, task_column: str = COLUMNS.question, worker_column: str = COLUMNS.voter
    ) -> None:
        self.task_column = task_column
        self.worker_column = worker_column

    def fit_predict(self, annotations: pd.DataFrame) -> pd.DataFrame:
        annotations = coerce_schema(annotations, self.task_column, self.worker_column)
        return self._aggregate(annotations)

    @abstractmethod
    def _aggregate(self, annotations: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError


class WeightedApprovalMixin:
    @staticmethod
    def _aggregate_weighted_answers(
        weighted_answers: pd.DataFrame, task_column: str
    ) -> pd.DataFrame:
        scores = weighted_answers.groupby(task_column, sort=False)[
            weighted_answers.columns
        ].sum()

        scores = scores.reindex(
            weighted_answers.index.get_level_values(task_column).unique()
        )

        winning_alternatives = scores.idxmax(axis=1).astype(
            pd.CategoricalDtype(categories=weighted_answers.columns)
        )

        aggregated_labels = pd.get_dummies(winning_alternatives)

        return aggregated_labels


class WeightedAggregator(Aggregator, WeightedApprovalMixin):
    @abstractmethod
    def compute_weights(self, annotations: pd.DataFrame) -> pd.Series:
        raise NotImplementedError

    def _aggregate(self, annotations: pd.DataFrame) -> pd.DataFrame:
        annotations = annotations.loc[annotations.sum(axis=1) > 0]

        weights = self.compute_weights(annotations)

        weighted_answers = annotations.multiply(weights, axis="index")

        return self._aggregate_weighted_answers(
            weighted_answers, task_column=self.task_column
        )
