import datasets


def cycle_dataset(dataset: datasets.IterableDataset) -> datasets.IterableDataset:
    while True:
        for example in dataset:
            yield example
        print(
            f"\033[32mCycling dataset {dataset.builder_name} {dataset.config_name} {dataset.split}\033[0m"
        )
