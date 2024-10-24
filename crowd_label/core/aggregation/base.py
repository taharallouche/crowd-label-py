from abc import ABC, abstractmethod

import pandas as pd

from crowd_label.core.utils.inventory import COLUMNS


class Aggregator(ABC):
	_name: str

	@abstractmethod
	def aggregate(self, annotations: pd.DataFrame) -> pd.DataFrame:
		raise NotImplementedError

	def __str__(self) -> str:
		return self._name


class VoterMixin:
	@staticmethod
	def _get_aggregated_labels(votes: pd.DataFrame) -> pd.DataFrame:
		scores = votes.groupby(COLUMNS.question, sort=False)[votes.columns].sum()

		scores = scores.reindex(votes.index.get_level_values(COLUMNS.question).unique())

		winning_alternatives = scores.idxmax(axis=1).astype(
			pd.CategoricalDtype(categories=votes.columns)
		)

		aggregated_labels = pd.get_dummies(winning_alternatives)

		return aggregated_labels


class WeightedAggregator(Aggregator, VoterMixin):
	@abstractmethod
	def _compute_weights(self, annotations: pd.DataFrame) -> pd.Series:
		raise NotImplementedError

	def aggregate(self, annotations: pd.DataFrame) -> pd.DataFrame:
		weights = self._compute_weights(annotations)

		weighted_answers = annotations.multiply(weights, axis="index")

		return self._get_aggregated_labels(weighted_answers)
