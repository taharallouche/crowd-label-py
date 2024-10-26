from abc import ABC, abstractmethod

import pandas as pd

from hakeem.core.utils.inventory import COLUMNS


class Aggregator(ABC):
	def __init__(
		self, task_column: str = COLUMNS.question, worker_column: str = COLUMNS.voter
	) -> None:
		self.task_column = task_column
		self.worker_column = worker_column

	def fit_predict(self, annotations: pd.DataFrame) -> pd.DataFrame:
		annotations = self._coerce_annotations(annotations)
		return self._aggregate(annotations)

	@abstractmethod
	def _aggregate(self, annotations: pd.DataFrame) -> pd.DataFrame:
		raise NotImplementedError

	def _coerce_annotations(self, annotations: pd.DataFrame) -> pd.DataFrame:
		all_columns = annotations.reset_index().columns
		required = [self.task_column, self.worker_column]

		if missing := set(required) - set(all_columns):
			raise ValueError(
				f"Annotations should have {self.task_column} and"
				f" {self.worker_column} as columns or index levels, missing {missing}."
			)

		if set(all_columns) == set(required):
			raise ValueError("Annotations should have at least one label column")

		annotations = annotations.reset_index().set_index(required)[
			[column for column in annotations.columns if column not in required]
		]

		return annotations


class WeightedApprovalMixin:
	@staticmethod
	def _get_aggregated_labels(
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
		weights = self.compute_weights(annotations)

		weighted_answers = annotations.multiply(weights, axis="index")

		return self._get_aggregated_labels(weighted_answers, self.task_column)
