import numpy as np
from typing import Union


class BaseSampler:
    """A base class for sampling dataset indices."""

    def __init__(self, dataset, shuffle: bool):
        self.indices = np.arange(len(dataset))
        self.labels = dataset.labels
        self.shuffle = shuffle

    def _get_indices(self) -> Union[np.ndarray, list]:
        raise NotImplementedError

    def __iter__(self):
        """Iterates over indices."""
        indices = self._get_indices()
        if self.shuffle:
            indices = np.random.permutation(indices)
        return iter(indices)


class DefaultSampler(BaseSampler):
    """A default sampler iterating over samples without any changes."""

    def _get_indices(self):
        return self.indices


class UpsamplingSampler(BaseSampler):
    """A sampler upsampling minority class.

    Upsampling is a technique used to create additional data points of the minority class to balance class labels.
        This is usually done by duplicating existing samples or creating new ones.
    """

    def __init__(self, dataset, shuffle: bool):
        super().__init__(dataset, shuffle)

        unique_labels, counts = np.unique(self.labels, return_counts=True)
        self.max_count = max(counts)
        self.class_indices = {label: np.where(self.labels == label)[0] for label in unique_labels}

    def _get_indices(self):
        indices = []

        for label in self.class_indices.keys():
            indices.extend(self.class_indices[label])
            if len(self.class_indices[label]) < self.max_count:
                num_indices_to_add = self.max_count - len(self.class_indices[label])
                indices.extend(np.random.choice(self.class_indices[label], size=num_indices_to_add))

        return indices
