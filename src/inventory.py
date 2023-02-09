from dataclasses import dataclass


@dataclass
class DataInfos:
    name: str
    path: str
    alternatives: list
    nbr_questions: int


ANIMALS_DATA_INFO = DataInfos(
    "animals",
    "data/data_animals.csv",
    ["Leopard", "Tiger", "Puma", "Jaguar", "Lion(ess)", "Cheetah"],
    16,
)

TEXTURES_DATA_INFO = DataInfos(
    "textures",
    "data/data_textures.csv",
    ["Gravel", "Grass", "Brick", "Wood", "Sand", "Cloth"],
    16,
)

LANGUAGE_DATA_INFO = DataInfos(
    "languages",
    "data/data_languages.csv",
    ["Hebrew", "Russian", "Japanese", "Thai", "Chinese", "Tamil", "Latin", "Hindi"],
    25,
)

data_infos = {
    "animals": ANIMALS_DATA_INFO,
    "textures": TEXTURES_DATA_INFO,
    "languages": LANGUAGE_DATA_INFO,
}
