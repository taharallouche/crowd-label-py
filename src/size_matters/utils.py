import os
from typing import Optional

import numpy as np
import scipy.stats


def get_data_folder_path(dataset_name: Optional[str]) -> Optional[str]:
    current_dir = os.path.abspath(os.getcwd())
    while current_dir != "/":  # Stop at the root directory
        if os.path.isdir(os.path.join(current_dir, "data")):
            current_dir = os.path.join(current_dir, "data")
            break
        data_folder = os.path.join(os.path.dirname(current_dir), "data")
        return f"{data_folder}/data_{dataset_name}.csv"
    else:
        raise Exception("Could not find the 'data' folder")


def confidence_margin_mean(
    data: np.ndarray, confidence: float = 0.95
) -> "tuple[np.ndarray, np.ndarray, np.ndarray]":
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2.0, n - 1)
    return m, m - h, m + h
