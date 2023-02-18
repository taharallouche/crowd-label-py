import pandas as pd
from inventory import DataInfos


def prepare_data(dataset_info: DataInfos) -> "tuple[pd.DataFrame, pd.DataFrame]":
    """
    This function prepares two dataframes: one containing the ground truths of the instances and one containing the
    annotations. Each row contains the question, the voter, and a binary vector whose coordinates equal one if and
    only if the associated alternative is selected by the voter.
     :param data: name of the dataset: "animals","textures" or "languages".
     :return: Annotations and GroundTruths dataframes
    """

    path = dataset_info.path
    nbr_questions = dataset_info.nbr_questions
    Alternatives = dataset_info.alternatives

    # Reading Dataset
    Data_brut = pd.read_csv(
        path,
        delimiter=",",
        index_col=False,
        header=0,
        names=["Interface", "Mechanism"]
        + ["Question" + str(i) for i in range(0, nbr_questions)]
        + ["TrueAnswer" + str(i) for i in range(0, nbr_questions)]
        + ["Answer" + str(i) for i in range(0, nbr_questions)]
        + ["Comments"],
        usecols=["Interface"]
        + ["Question" + str(i) for i in range(0, nbr_questions)]
        + ["TrueAnswer" + str(i) for i in range(0, nbr_questions)]
        + ["Answer" + str(i) for i in range(0, nbr_questions)],
    )

    # Cleaning the data
    Data_brut = Data_brut.loc[Data_brut.Interface == "subset"]
    del Data_brut["Interface"]

    # Preparing GroundTruth Dataframe
    Questions = Data_brut.iloc[0, 0:nbr_questions].to_numpy()
    GroundTruth = pd.DataFrame(columns=["Question"] + Alternatives)
    for i in range(len(Questions)):
        L = Data_brut.iloc[0, i + nbr_questions]
        row = {"Question": Questions[i]}
        for alternative in Alternatives:
            row[alternative] = int(alternative == L)
        GroundTruth = GroundTruth.append(row, ignore_index=True)

    # Preparing Annotations Dataframe
    Annotations = pd.DataFrame(columns=["Voter", "Question"] + Alternatives)
    for i in range(len(Questions)):
        for j in range(Data_brut.shape[0]):
            col = 0
            for c in range(0, nbr_questions):
                if Data_brut.iloc[j, c] == Questions[i]:
                    break
                else:
                    col += 1
            L = Data_brut.iloc[j, col + 2 * nbr_questions].split("|")
            row = {"Voter": j, "Question": Questions[i]}
            for alternative in Alternatives:
                row[alternative] = int(alternative in L)
            Annotations = Annotations.append(row, ignore_index=True)
    Annotations[Alternatives] = Annotations[Alternatives].astype(int)

    return Annotations, GroundTruth
