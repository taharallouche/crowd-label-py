import numpy as np
import pandas as pd

from crowd_label.core.utils.inventory import COLUMNS, Dataset


def _get_column_names(name: str, nbr_questions: int) -> list[str]:
	return [f"{name}{i}" for i in range(nbr_questions)]


def _get_columns(nbr_questions: int) -> list[str]:
	columns = (
		_get_column_names(COLUMNS.question, nbr_questions)
		+ _get_column_names(COLUMNS.true_answer, nbr_questions)
		+ _get_column_names(COLUMNS.answer, nbr_questions)
	)
	return columns


def _read_raw_data(dataset: Dataset) -> pd.DataFrame:
	path = dataset.path
	nbr_questions = dataset.nbr_questions
	columns = _get_columns(nbr_questions)
	raw_data = pd.read_csv(
		path,
		delimiter=",",
		index_col=False,
		header=0,
		names=[
			COLUMNS.interface,
			COLUMNS.mechanism,
			*columns,
			COLUMNS.comments,
		],
		usecols=[COLUMNS.interface, *columns],
	)
	raw_data = raw_data.loc[raw_data[COLUMNS.interface] == "subset", columns]
	return raw_data


def _get_ground_truth(
	raw_data: pd.DataFrame, nbr_questions: int, alternatives: list[str]
) -> pd.DataFrame:
	questions = raw_data.iloc[0][
		_get_column_names(COLUMNS.question, nbr_questions)
	].to_numpy()

	true_answers = raw_data.iloc[0][
		_get_column_names(COLUMNS.true_answer, nbr_questions)
	].to_numpy()

	groundtruth_matrix = np.equal.outer(alternatives, true_answers).astype(int)

	groundtruth_data = np.concatenate(
		[questions.reshape(-1, 1), groundtruth_matrix.T], axis=1
	)
	groundtruth = pd.DataFrame(
		groundtruth_data, columns=[COLUMNS.question] + alternatives
	).set_index([COLUMNS.question])
	return groundtruth


def _get_annotations(
	raw_data: pd.DataFrame, nbr_questions: int, alternatives: list[str]
) -> pd.DataFrame:
	annotations = pd.DataFrame(columns=[COLUMNS.voter, COLUMNS.question] + alternatives)
	questions = raw_data.iloc[0, 0:nbr_questions].to_numpy()
	for i in range(len(questions)):
		for j in range(raw_data.shape[0]):
			col = 0
			for c in range(0, nbr_questions):
				if raw_data.iloc[j, c] == questions[i]:
					break
				else:
					col += 1
			L = raw_data.iloc[j, col + 2 * nbr_questions].split("|")
			row = {COLUMNS.voter: j, COLUMNS.question: questions[i]}
			for alternative in alternatives:
				row[alternative] = int(alternative in L)
			annotations = annotations._append(row, ignore_index=True)
	annotations[alternatives] = annotations[alternatives].astype(int)
	return annotations.set_index([COLUMNS.question, COLUMNS.voter])


def prepare_data(dataset: Dataset) -> tuple[pd.DataFrame, pd.DataFrame]:
	nbr_questions = dataset.nbr_questions
	alternatives = dataset.alternatives

	raw_data = _read_raw_data(dataset)
	groundtruth = _get_ground_truth(raw_data, nbr_questions, alternatives)
	annotations = _get_annotations(raw_data, nbr_questions, alternatives)

	return annotations, groundtruth
