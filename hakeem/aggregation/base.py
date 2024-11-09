from abc import ABC, abstractmethod

import pandas as pd

from hakeem.utils.coerce import coerce_schema
from hakeem.utils.inventory import COLUMNS


class Aggregator(ABC):
    """
    Abstract base class for aggregators.

    This class provides a template for creating custom aggregation methods. It requires
    subclasses to implement the `_aggregate` method, which performs the actual
    aggregation logic.

    Attributes:
        task_column (str): The name of the column containing task identifiers.
        worker_column (str): The name of the column containing worker identifiers.

    Methods:
        fit_predict(annotations: pd.DataFrame) -> pd.DataFrame:
            Coerces the schema of the annotations DataFrame and applies the aggregation
            method defined in the `_aggregate` method.

        _aggregate(annotations: pd.DataFrame) -> pd.DataFrame:
            Abstract method to be implemented by subclasses. This method should contain
            the logic for aggregating the annotations.
    """

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
    """
    A mixin class that provides functionality to aggregate weighted answers.

    Static Methods
    -------
    _aggregate_weighted_answers(
        weighted_answers: pd.DataFrame, task_column: str
    ) -> pd.DataFrame
        Aggregates weighted answers by summing the scores for each task and
        determining the winning alternatives.

    """

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
    """
    A base class for aggregators that use weighted annotations.

    This class extends the Aggregator and WeightedApprovalMixin classes and provides
    a framework for aggregating annotations with weights. Subclasses must implement
    the `compute_weights` method to define how weights are calculated.

    Methods
    -------
    compute_weights(annotations: pd.DataFrame) -> pd.Series
        Abstract method to compute weights for the given annotations. Must be
        implemented by subclasses.

    _aggregate(annotations: pd.DataFrame) -> pd.DataFrame
        Aggregates the given annotations using the computed weights.
    """

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
