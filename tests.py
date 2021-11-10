import os
import unittest
from glob import glob

from keras_object_detection import benchmarks


class ShapesBenchMarkTester(unittest.TestCase):
    def __init__(self, methodName: str = ...) -> None:
        super().__init__(methodName=methodName)
        self.shapes_benchmark = benchmarks.ShapesBenchMark()
        self.n_data_samples = 64
        self.shapes_benchmark.make_dataset(n_data_samples=self.n_data_samples)
        self.shapes_benchmark.set_label_map()
        self.tfrecord_dir = self.shapes_benchmark.create_tfrecords(
            val_split=0.2, samples_per_shard=16
        )

    def test_shapes_benchmark(self):
        assert len(self.shapes_benchmark.dataset) == self.n_data_samples
        assert (
            len(glob(os.path.join(self.shapes_benchmark.images_dir, "*")))
            == self.n_data_samples
        )

    def test_shapes_tfrecords(self):
        assert len(glob(os.path.join(self.tfrecord_dir, "train/*"))) > 0
        assert len(glob(os.path.join(self.tfrecord_dir, "val/*"))) > 0
