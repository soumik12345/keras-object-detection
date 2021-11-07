import os
from abc import ABC, abstractmethod

from keras_object_detection import utils


class BenchMark(ABC):
    def __init__(
        self,
        dataset_name: str = "shapes",
        dump_dir: str = "./dataset/",
        *args,
        **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.dump_dir = dump_dir
        self.data_dir = os.path.join(dump_dir, dataset_name)
        self.images_dir = os.path.join(self.data_dir, "images")
        utils.make_directory(dump_dir)
        utils.make_directory(self.data_dir)
        utils.make_directory(self.images_dir)

    @abstractmethod
    def make_dataset(self, *args, **kwargs):
        pass

    @abstractmethod
    def get_dataset(self, *args, **kwargs):
        pass

    @abstractmethod
    def plot_samples(self, *args, **kwargs):
        pass
