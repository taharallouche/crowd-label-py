import pandas as pd

from size_matters.inventory import Dataset


def _get_column_names(name: str, nbr_questions: int) -> "list[str]":
    return [f"{name}{i}" for i in range(nbr_questions)]


def _get_columns(nbr_questions: int) -> "list[str]":
    columns = (
        _get_column_names("Question", nbr_questions)
        + _get_column_names("TrueAnswer", nbr_questions)
        + _get_column_names("Answer", nbr_questions)
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
        names=["Interface", "Mechanism", *columns, "Comments"],
        usecols=["Interface", *columns],
    )
    raw_data = raw_data.loc[raw_data.Interface == "subset", columns]
    return raw_data


def _get_ground_truth(
    raw_data: pd.DataFrame, nbr_questions: int, alternatives: "list[str]"
) -> pd.DataFrame:
    questions = raw_data.iloc[0, 0:nbr_questions].to_numpy()
    groundtruth = pd.DataFrame(columns=["Question"] + alternatives)
    for i in range(len(questions)):
        L = raw_data.iloc[0, i + nbr_questions]
        row = {"Question": questions[i]}
        for alternative in alternatives:
            row[alternative] = int(alternative == L)
        groundtruth = groundtruth.append(row, ignore_index=True)
    return groundtruth


def _get_annotations(
    raw_data: pd.DataFrame, nbr_questions: int, alternatives: "list[str]"
) -> pd.DataFrame:
    annotations = pd.DataFrame(columns=["Voter", "Question"] + alternatives)
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
            row = {"Voter": j, "Question": questions[i]}
            for alternative in alternatives:
                row[alternative] = int(alternative in L)
            annotations = annotations.append(row, ignore_index=True)
    annotations[alternatives] = annotations[alternatives].astype(int)
    return annotations


def prepare_data(
    dataset: Dataset,
) -> "tuple[pd.DataFrame, pd.DataFrame]":
    nbr_questions = dataset.nbr_questions
    alternatives = dataset.alternatives

    raw_data = _read_raw_data(dataset)
    groundtruth = _get_ground_truth(raw_data, nbr_questions, alternatives)
    annotations = _get_annotations(raw_data, nbr_questions, alternatives)

    return annotations, groundtruth
