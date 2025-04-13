import os

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="threading")

from typing import Union
from abc import ABC, abstractmethod

import neptune
from neptune.utils import stringify_unsupported


class BaseLogger(ABC):
    """A base experiment logger class."""

    @abstractmethod
    def __init__(self, config):
        """Logs git commit id, dvc hash, environment."""
        pass

    @abstractmethod
    def log_hyperparameters(self, params: dict):
        pass

    @abstractmethod
    def save_metrics(self, *args, **kwargs):
        pass

    @abstractmethod
    def save_plot(self, *args, **kwargs):
        pass

    @abstractmethod
    def stop(self):
        pass


class NeptuneLogger(BaseLogger):
    """A neptune.ai experiment logger class."""

    def __init__(self, config):
        super().__init__(config)

        try:
            token = os.environ['NEPTUNE_API_TOKEN']
        except KeyError:
            token = config.token

        self.run = neptune.init_run(
            project=config.project,
            api_token=token,
            name=config.experiment_name,
            with_id=config.run_id,
            tags=[config.tags]
        )


    def log_hyperparameters(self, params: dict):
        """Model hyperparameters logging."""
        self.run['hyperparameters'] = stringify_unsupported(params)


    def save_metrics(self, 
                     type_set: str, 
                     metric_name: Union[list[str], str], 
                     metric_value: Union[list[float], float],
                     step=None):
        
        if isinstance(metric_name, list):
            for p_n, p_v in zip(metric_name, metric_value):
                self.run[f"{type_set}/{p_n}"].log(p_v, step=step)

        else:
            self.run[f"{type_set}/{metric_name}"].log(metric_value, step=step)


    def save_plot(self, 
                  type_set: str, 
                  plot_name: str, 
                  plt_fig):
        
        self.run[f"{type_set}/{plot_name}"].append(plt_fig)

    
    def add_tag(self, tag: str):
        self.run["sys/tags"].add([tag])


    def stop(self):
        self.run.stop()


