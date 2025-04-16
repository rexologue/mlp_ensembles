import os
import cv2

from transforms.target_transforms import OneHotEncoding
from utils.common_functions import read_dataframe_file
from utils.enums import SetType


class EmotionsDataset:
    """A class for the Emotions dataset. This class defines how data is loaded."""

    def __init__(self, config, set_type: SetType, cls: str = 'all', transforms=None):
        self.config = config
        self.set_type = set_type
        self.transforms = transforms

        # Read the annotation file that contains the image path, set_type, and target values for the entire dataset
        annotation = read_dataframe_file(os.path.join(config.path_to_data, config.annot_filename))

        # Filter the annotation file according to the set_type
        self.annotation = annotation[annotation['set'] == self.set_type.name]

        if cls != 'all':
            self.annotation.loc[:, 'target'] = self.annotation['target'].apply(
                lambda x: x if x == cls else 'unknown'
            )

        self._paths = self.annotation['path'].tolist()
        self._targets = self.annotation['target'].map(self.config.label_mapping).tolist()

        self._ohe_targets = None

        if set_type != SetType.test:
            self._ohe_targets = OneHotEncoding(self.config.classes_num)(self._targets)

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
                    'path': image path (str)
                }
        """
        image = cv2.imread(os.path.join(self.config.path_to_data, self._paths[idx]), cv2.IMREAD_GRAYSCALE)

        if self.transforms is not None:
            image = self.transforms(image)

        ohe_target = None
        if self._ohe_targets is not None:
            ohe_target = self._ohe_targets[idx]

        return {'image': image, 'target': self._targets[idx], 'ohe_target': ohe_target, 'path': self._paths[idx]}
    