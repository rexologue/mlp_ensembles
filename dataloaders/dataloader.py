import numpy as np

from dataloaders import batch_samplers
from utils.enums import SamplerType


class Dataloader:
    """Provides an iterable over the given dataset."""

    def __init__(self, dataset, batch_size: int, sampler: SamplerType = SamplerType.Default,
                 shuffle: bool = False, drop_last: bool = False, **kwargs):
        
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last

        self.num_samples = len(self.dataset)
        self.sampler = getattr(batch_samplers, sampler.name + 'Sampler')(self.dataset, shuffle, **kwargs)

        if sampler == SamplerType.Upsampling:
            self.batch_amounts = int(np.ceil(len(self.sampler._get_indices()) / self.batch_size))
        else:
            self.batch_amounts = int(np.ceil(self.num_samples / self.batch_size))

    
    def __len__(self):
        return self.batch_amounts


    def __iter__(self):
        """Returns a batch at each iteration."""
        batch_data = []

        for idx in self.sampler:
            batch_data.append(self.dataset[idx])
            if len(batch_data) == self.batch_size:
                yield self._collate_fn(batch_data)
                batch_data = []

        if not self.drop_last and batch_data:
            yield self._collate_fn(batch_data)


    @staticmethod
    def _collate_fn(batch_data: list[dict]) -> dict:
        """Combines a list of samples into a dictionary.

        Example:
            batch_data = [
                {'image': <image_array_1>, 'target': <target_1>, <ohe_target>: <ohe_target_array_1>, 'path': <path_1>},
                {'image': <image_array_2>, 'target': <target_2>, <ohe_target>: <ohe_target_array_2>, 'path': <path_2>},
                {'image': <image_array_3>, 'target': <target_3>, <ohe_target>: <ohe_target_array_3>, 'path': <path_3>},
                {'image': <image_array_4>, 'target': <target_4>, <ohe_target>: <ohe_target_array_4>, 'path': <path_4>},
            ]
            →
            batch = {
                'image': np.array([<image_array_1>, <image_array_2>, <image_array_3>, <image_array_4>]),
                'target': np.array([<target_1>, <target_2>, <target_3>, <target_4>]),
                'ohe_target': np.array([<ohe_target_array_1>, <ohe_target_array_2>, <ohe_target_array_3>, <ohe_target_array_4>]),
                'path': np.array([<path_1>, <path_2>, <path_3>, <path_4>]),
            }

        Args:
            batch_data: A list of batch data dictionaries.

        Returns:
            batch: A batch constructed from batch data with its keys.
        """
        batch = {}

        for key in batch_data[0].keys():
            batch[key] = np.array([x[key] for x in batch_data]) 

        return batch
