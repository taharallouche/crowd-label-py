from abc import ABC, abstractmethod
from typing import Callable

import numpy as np
import pandas as pd

from crowd_label.utils.inventory import COLUMNS, DEFAULT_RELIABILITY_BOUNDS


class Aggregator(ABC):
	_name: str

	@abstractmethod
	def aggregate(self, annotations: pd.DataFrame, **kwargs) -> pd.DataFrame:
		pass

	def __str__(self) -> str:
		return self._name


class VoterMixin:
	@staticmethod
	def _get_aggregated_labels(votes: pd.DataFrame) -> pd.DataFrame:
		scores = votes.groupby(COLUMNS.question, sort=False)[votes.columns].sum()

		scores = scores.reindex(votes.index.get_level_values(COLUMNS.question).unique())

		winning_alternatives = pd.Categorical(
			scores.idxmax(axis=1),
			categories=votes.columns,
			ordered=True,
		)

		aggregated_labels = pd.get_dummies(winning_alternatives, columns=votes.columns)

		return aggregated_labels


class WeightedAggregator(Aggregator, VoterMixin):
	@property
	@abstractmethod
	def _weight_calculator(self) -> Callable[[pd.Series], pd.Series]:
		pass

	def aggregate(self, annotations: pd.DataFrame, **kwargs) -> pd.DataFrame:
		vote_size = annotations.sum(axis=1)
		weights = type(self)._weight_calculator(vote_size)

		weighted_answers = annotations.multiply(weights, axis="index")

		return self._get_aggregated_labels(weighted_answers)


class StandardApprovalAggregator(WeightedAggregator):
	_name: str = "Standard Approval Aggregator"

	_weight_calculator = lambda vote_size: pd.Series(  # noqa : E731
		1 * len(vote_size), index=vote_size.index
	)


class CondorcetAggregator(VoterMixin, Aggregator):
	_name: str = "Condorcet Aggregator"

	def __init__(
		self,
		lower_reliability_bound: float = DEFAULT_RELIABILITY_BOUNDS.lower,
		upper_reliability_bound: float = DEFAULT_RELIABILITY_BOUNDS.upper,
	):
		self.lower_reliability_bound = lower_reliability_bound
		self.upper_reliability_bound = upper_reliability_bound

	def aggregate(self, annotations: pd.DataFrame, **kwargs) -> pd.DataFrame:
		vote_size = annotations.sum(axis=1)
		reliabilities = (len(annotations.columns) - vote_size - 1) / (
			len(annotations.columns) - 2
		)
		reliabilities = reliabilities.clip(
			self.lower_reliability_bound, self.upper_reliability_bound
		)
		weights = np.log(reliabilities / (1 - reliabilities))
		weighted_answers = annotations.multiply(weights, axis="index")

		return self._get_aggregated_labels(weighted_answers)


class EuclidAggregator(WeightedAggregator):
	_name: str = "Euclidean Mallow Aggregator"
	_weight_calculator = lambda vote_size: np.sqrt(  # noqa : E731
		vote_size + 1
	) - np.sqrt(vote_size - 1)


class JaccardAggregator(WeightedAggregator):
	_name: str = "Jaccard Mallow Aggregator"
	_weight_calculator = lambda vote_size: 1 / vote_size  # noqa : E731


class DiceAggregator(WeightedAggregator):
	_name: str = "Dice Mallow Aggregator"
	_weight_calculator = lambda vote_size: 2 / (vote_size + 1)  # noqa : E731
