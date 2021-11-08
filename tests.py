import os
import unittest
from glob import glob

from keras_object_detection import benchmarks


class ShapesBenchMarkTester(unittest.TestCase):
    def test_shapes_benchmark(self):
        shapes_benchmark = benchmarks.ShapesBenchMark()
        shapes_benchmark.make_dataset(n_data_samples=10)
        assert len(shapes_benchmark.dataset) == 10
        assert len(glob(os.path.join(shapes_benchmark.images_dir, "*"))) == 10
