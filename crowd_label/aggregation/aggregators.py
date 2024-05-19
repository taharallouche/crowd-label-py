from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from crowd_label.utils.inventory import COLUMNS, DEFAULT_RELIABILITY_BOUNDS


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


class CondorcetAggregator(WeightedAggregator):
	_name: str = "Condorcet Aggregator"

	def __init__(
		self,
		lower_reliability_bound: float = DEFAULT_RELIABILITY_BOUNDS.lower,
		upper_reliability_bound: float = DEFAULT_RELIABILITY_BOUNDS.upper,
	):
		self.lower_reliability_bound = lower_reliability_bound
		self.upper_reliability_bound = upper_reliability_bound

	def _compute_weights(self, annotations: pd.DataFrame) -> pd.Series:
		vote_size = annotations.sum(axis=1)
		reliabilities = (len(annotations.columns) - vote_size - 1) / (
			len(annotations.columns) - 2
		)
		reliabilities = reliabilities.clip(
			self.lower_reliability_bound, self.upper_reliability_bound
		)
		weights = np.log(reliabilities / (1 - reliabilities))

		return weights


class StandardApprovalAggregator(WeightedAggregator):
	_name: str = "Standard Approval Aggregator"

	@staticmethod
	def _compute_weights(annotations: pd.DataFrame) -> pd.Series:
		return pd.Series(1, index=annotations.index)


class EuclidAggregator(WeightedAggregator):
	_name: str = "Euclidean Mallow Aggregator"

	@staticmethod
	def _compute_weights(annotations: pd.DataFrame) -> pd.Series:
		vote_size = annotations.sum(axis=1)
		return np.sqrt(vote_size + 1) - np.sqrt(vote_size - 1)


class JaccardAggregator(WeightedAggregator):
	_name: str = "Jaccard Mallow Aggregator"

	@staticmethod
	def _compute_weights(annotations: pd.DataFrame) -> pd.Series:
		vote_size = annotations.sum(axis=1)
		return 1 / vote_size


class DiceAggregator(WeightedAggregator):
	_name: str = "Dice Mallow Aggregator"

	@staticmethod
	def _compute_weights(annotations: pd.DataFrame) -> pd.Series:
		vote_size = annotations.sum(axis=1)
		return 2 / (vote_size + 1)
