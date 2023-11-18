import ray

from size_matters.experiments import compare_methods
from size_matters.inventory import DATASETS

if __name__ == "__main__":  # pragma: no cover
    ray.init()

    dataset_name = input(f"Select a dataset [{'|'.join(DATASETS)}]: ")
    assert dataset_name in DATASETS, "Invalid dataset"
    dataset = DATASETS[dataset_name]

    max_voters = int(
        input(f"Choose the maximum number of voters, max={dataset.nbr_voters}:")
    )
    assert max_voters <= dataset.nbr_voters, "Too many voters"

    n_batch = int(input("Choose the number of batches: "))
    assert n_batch > 0, "Please choose a positive number of batches"

    compare_methods(dataset=dataset, max_voters=max_voters, n_batch=n_batch)
