import numpy as np
import pandas as pd

from dataloaders.dataloader import Dataloader
from transforms.image_transforms import Sequential
from datasets.emotions_dataset import EmotionsDataset

from config import exp_cfg
from trainer import Trainer
from utils.enums import SetType


def predict(model_path: str):
    trainer = Trainer(exp_cfg, init_logger=False)

    # Get data to make predictions on
    eval_transforms = Sequential(exp_cfg.data_cfg.eval_transforms)
    test_dataset = EmotionsDataset(exp_cfg.data_cfg, SetType.test, transforms=eval_transforms)
    test_dataloader = Dataloader(test_dataset, exp_cfg.train.batch_size, shuffle=False)

    # Get predictions
    predictions, image_paths = trainer.predict(model_path, test_dataloader)

    # Save results to submission file
    test_results_df = pd.DataFrame({'ID': image_paths, 'prediction': predictions})
    test_results_df.to_csv('test_predictions.csv', index=False)


if __name__ == '__main__':
    predict()
