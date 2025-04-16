import os

from utils.enums import SetType
from utils.common_functions import read_dataframe_file
from transforms.target_transforms import OneHotEncoding

class MetaDataset:
    """A class for the Meta dataset. This class defines how data is loaded."""

    def __init__(self, X, y, classes_num: int):
        self._data = X
        self._targets = y.tolist()

        self._ohe_targets = None

        if y is not None:
            self._ohe_targets = OneHotEncoding(classes_num)(self._targets)

    @property
    def labels(self):
        return self._targets

    def __len__(self):
        return len(self._targets)

    def __getitem__(self, idx: int) -> dict:
        """Loads and returns one sample from a dataset with the given idx index.

        Returns:
            A dict with the following data:
                {
                    'image': image (numpy.ndarray),
                    'target': target (int),
                    'ohe_target': ohe target (numpy.ndarray),
                }
        """
        ohe_target = None

        if self._ohe_targets is not None:
            ohe_target = self._ohe_targets[idx]

        return {'image': self._data[idx], 'target': self._targets[idx], 'ohe_target': ohe_target}