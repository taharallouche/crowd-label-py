# Import dependencies
import pandas as pd


# Read csv to pandas dataframe
def prepare_data(data="animals"):
    """
    This function prepares two dataframes: one containing the ground truths of the instances and one containing the
    annotations. Each row contains the question, the voter, and a binary vector whose coordinates equal one if and
    only if the associated alternative is selected by the voter.
     :param data: name of the dataset: "animals","textures" or "languages".
     :return: Annotations and GroundTruths dataframes
    """

    # Setting the path to the dataset and some of its properties
    if data == "animals":
        path = "Data/Data_Shah/data_animals.csv"
        nbr_questions = 16
        Alternatives = ["Leopard", "Tiger", "Puma", "Jaguar", "Lion(ess)", "Cheetah"]
    elif data == "textures":
        path = "Data/Data_Shah/data_textures.csv"
        nbr_questions = 16
        Alternatives = ["Gravel", "Grass", "Brick", "Wood", "Sand", "Cloth"]
    else:
        path = "Data/Data_Shah/data_languages.csv"
        nbr_questions = 25
        Alternatives = ["Hebrew", "Russian", "Japanese", "Thai", "Chinese", "Tamil", "Latin", "Hindi"]

    # Reading Dataset
    Data_brut = pd.read_csv(path, delimiter=',', index_col=False, header=0,
                            names=["Interface", "Mechanism"] + ["Question" + str(i) for i in
                                                                range(0, nbr_questions)] + [
                                      "TrueAnswer" + str(i) for i in range(0, nbr_questions)] + ["Answer" + str(i) for i
                                                                                                 in
                                                                                                 range(0,
                                                                                                       nbr_questions)] + [
                                      "Comments"],
                            usecols=["Interface"] + ["Question" + str(i) for i in range(0, nbr_questions)] + [
                                "TrueAnswer" + str(i) for i in range(0, nbr_questions)] + ["Answer" + str(i) for i in
                                                                                           range(0, nbr_questions)])

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
    Annotations = pd.DataFrame(
        columns=["Voter", "Question"] + Alternatives)
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
    return Annotations, GroundTruth
