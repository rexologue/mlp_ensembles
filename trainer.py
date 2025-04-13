import os

import numpy as np
from tqdm import tqdm

from dataloaders.dataloader import Dataloader
from datasets.emotions_dataset import EmotionsDataset

from models.mlp import MLP
from modules.optimizers.sgd import SGD
from modules.losses.cross_entropy_loss import CrossEntropyLoss

from transforms.image_transforms import Sequential
from utils.common_functions import set_seed

from utils.enums import SetType
from utils.logger import NeptuneLogger
from utils.checkpointer import Checkpointer
from utils.visualization import plot_confusion_matrix
from utils.metrics import balanced_accuracy_score, confusion_matrix


class Trainer:
    """A class for model training."""

    def __init__(self, config, init_logger: bool = True):
        self.config = config
        set_seed(self.config.seed)

        self._prepare_data()
        self._prepare_model()

        self._init_logger(init_logger)

        self.checkpointer = Checkpointer(config.checkpoints_dir, config.keep_last_n_checkpoints)


    def _init_logger(self, init_logger: bool):
        if init_logger:
            self.logger = NeptuneLogger(self.config.neptune)

            if self.config.continue_from_checkpoint is None:
                self.logger.log_hyperparameters(self.config)
        else:
            self.logger = None


    def _prepare_data(self):
        """Prepares training and validation data."""
        data_cfg = self.config.data_cfg
        batch_size = self.config.train.batch_size

        train_transforms = Sequential(data_cfg.train_transforms)
        validation_transforms = Sequential(data_cfg.eval_transforms)

        self.train_dataset = EmotionsDataset(data_cfg, SetType.train, self.config.expert_class, transforms=train_transforms)
        self.train_dataloader = Dataloader(self.train_dataset, batch_size, shuffle=True, sampler=data_cfg.sampler_type)

        self.eval_train_dataloader = Dataloader(self.train_dataset, batch_size, shuffle=False)

        self.validation_dataset = EmotionsDataset(data_cfg, SetType.validation, self.config.expert_class, transforms=validation_transforms)
        self.validation_dataloader = Dataloader(self.validation_dataset, batch_size=batch_size, shuffle=False)


    def _prepare_model(self):
        """Prepares model, optimizer and loss function."""
        self.model = MLP(self.config.model_cfg)
        self.optimizer = SGD(self.model, learning_rate=self.config.train.learning_rate, weight_decay=self.config.train.weight_decay)
        self.criterion = CrossEntropyLoss()


    def update_best_params(self, valid_metric: float, best_metric: float, epoch: int) -> float:
        """Updates best parameters: saves model if metrics exceeds the best values achieved."""
        if best_metric < valid_metric:
            best_metric = valid_metric
            self.checkpointer.save(self.model.get_params(), epoch, best_metric, best=True)

        return best_metric


    def make_step(self, batch: dict, update_model: bool = False, compute_loss: bool = True) -> tuple[float, np.ndarray]:
        """This method performs one step, including forward pass, calculation of the target function, backward
        pass and updating the model weights (if update_model is True).

        Args:
            batch: A batch data dictionary.
            update_model: If True it is necessary to perform a backward pass and update the model weights.
            compute_loss: Is it neccessary to compute loss?

        Returns:
            loss: The loss function value.
            output: The model output (batch_size x classes_num).
        """
        images = batch['image']
        ohes = batch['ohe_target'] if compute_loss else None

        logits = self.model(images)
        loss = self.criterion(ohes, logits) if compute_loss else None

        if update_model and compute_loss:
            self.optimizer.zero_grad()

            grad = self.criterion.backward(ohes)
            self.optimizer.backward(grad)

            self.optimizer.step()

        return loss, logits


    def train_epoch(self):
        """Trains the model on training data for one epoch.

        The method goes through all train_dataloader batches and calls the self.make_step() method at each step.
        """
        self.model.train()
        pbar = tqdm(total=len(self.train_dataloader))

        for i, batch in enumerate(self.train_dataloader):
            # Making step
            loss, logits = self.make_step(batch, True)

            # Get predictions, compute metrics
            preds = np.argmax(logits, axis=1)
            bal_acc = balanced_accuracy_score(batch['target'], preds)

            # Update pbar
            pbar.set_description(f"Loss: {loss:.4f}, Bal acc {bal_acc:.4f}")
            pbar.update(1)


    def fit(self):
        """The main model training loop."""
        if self.config.continue_from_checkpoint is not None:
            self.model, epoch, best_metric = self.checkpointer.load(self.model, self.config.continue_from_checkpoint)
            epoch += 1
        else:
            epoch = best_metric = 0

        # Параметры Early Stopping
        patience = 17            # Умеренный patience
        warmup_epochs = 5        # Период игнорирования остановки
        delta = 0.002            # Минимальный значимый прирост
        stop_counter = 0

        while epoch != self.config.num_epochs:
            # Train model 
            self.train_epoch()

            # Evaluate on train set
            self.evaluate(epoch, self.eval_train_dataloader, 'eval_train')

            # Evaluate on valid set
            current_metric = self.evaluate(epoch, self.validation_dataloader, 'eval')

            # Make checkpoint
            if ((epoch + 1) % self.config.checkpoint_save_frequency) == 0:
                self.checkpointer.save(self.model.get_params(), epoch, best_metric)

            # Update best metric if needed
            best_metric = self.update_best_params(current_metric, best_metric, epoch)

            if epoch > warmup_epochs:
                if current_metric > best_metric + delta:
                    stop_counter = 0
                else:
                    stop_counter += 1

                if stop_counter >= patience:
                    self.logger.add_tag('Early stopping')
                    break

            epoch += 1


    def evaluate(self, epoch: int, dataloader: Dataloader, set_type: str) -> float:
        """Evaluation.

        The method is used to make the model performance evaluation on training/validation/test data.

        Args:
            epoch: A current training epoch.
            dataloader: The dataloader for the chosen set type.
            set_type: What set is used for evaluating
        """
        self.model.eval()

        total_loss = []
        all_outputs, all_labels = [], []

        for batch in dataloader:
            loss, logits = self.make_step(batch, False)

            total_loss.append(loss)
            all_outputs.append(logits)
            all_labels.append(batch['target'])

        preds = np.argmax(np.vstack(all_outputs), axis=1)
        targets = np.hstack(all_labels)

        loss = np.mean(total_loss)
        bal_acc = balanced_accuracy_score(targets, preds)

        if self.logger is not None:
            self.logger.save_metrics(set_type, ['loss', 'bal_acc'], [loss, bal_acc], step=epoch)

            cm = confusion_matrix(targets, preds, classes_num=self.config.data_cfg.classes_num, normalize=True)
            fig = plot_confusion_matrix(cm, f'Epoch {epoch}', self.config.data_cfg.label_mapping.keys())
            self.logger.save_plot(set_type, f'CM Epoch {epoch}', fig)

        return bal_acc


    def predict(self, dataloader: Dataloader) -> tuple[np.ndarray, np.ndarray]:
        """Gets the model predictions for a given dataloader."""
        self.model = self.checkpointer.load_best_model(self.model)

        self.model.eval()
        all_outputs, all_image_paths = [], []

        for batch in dataloader:
            _, logits = self.make_step(batch, False, False)

            all_outputs.append(logits)
            all_image_paths.append(batch['path'])

        preds = np.argmax(np.vstack(all_outputs), axis=1)
        pathes = np.hstack(all_image_paths)

        return preds, pathes


    def batch_overfit(self):
        """One batch overfitting.

        This feature can be useful for debugging and evaluating your model's ability to learn and update its weights.
        """
        self.model.train()
        batch = next(iter(self.train_dataloader))

        pbar = tqdm(total=self.config.overfit.num_iterations)

        step = 0
        for i in range(self.config.overfit.num_iterations):
            # Making step
            loss, logits = self.make_step(batch, True)

            # Get predictions, compute metrics
            preds = np.argmax(logits, axis=1)
            bal_acc = balanced_accuracy_score(batch['target'], preds)

            # Log metrics
            if self.logger is not None:
                self.logger.save_metrics('batch_overfit', ['loss', 'bal_acc'], [loss, bal_acc], step=step)

            # Update pbar
            pbar.set_description(f"Loss: {loss:.4f}, Bal acc {bal_acc:.4f}")
            pbar.update(1)

            step += 1

