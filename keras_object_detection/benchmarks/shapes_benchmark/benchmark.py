import os
from tqdm import tqdm
from absl import logging

import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imsave, imread

from ..base import BenchMark
from ..tfrecord import TFrecordWriter
from .utils import convert_box, draw_boxes

from keras_object_detection import utils


class ShapesBenchMark(BenchMark):
    def __init__(
        self,
        height: int = 640,
        width: int = 640,
        dataset_name: str = "shapes",
        dump_dir: str = "./dataset/",
        *args,
        **kwargs
    ) -> None:
        super().__init__(dataset_name=dataset_name, dump_dir=dump_dir, *args, **kwargs)
        self.height = height
        self.width = width
        self.dataset = {}

    def __len__(self):
        return len(self.dataset)

    def set_label_map(self):
        super().set_label_map()
        self.label_map = {"circle": 0, "rectangle": 1}

    def _draw_circle(self, rgb_canvas, trials=1):
        if trials > 100:
            return rgb_canvas, {}
        canvas = np.uint8(~np.all(rgb_canvas == [255, 255, 255], axis=-1))
        max_radius = 0.20 * int(np.sqrt(self.height * self.width))
        min_radius = 0.05 * int(np.sqrt(self.height * self.width))
        radius = int(np.random.randint(low=min_radius, high=max_radius))
        x = int(np.random.randint(low=radius + 1, high=(self.width - radius - 1)))
        y = int(np.random.randint(low=radius + 1, high=(self.height - radius - 1)))
        new_canvas = np.zeros_like(canvas)
        new_canvas = cv2.circle(
            new_canvas, center=(x, y), radius=radius, color=1, thickness=-1
        )
        if np.any(np.logical_and(new_canvas, canvas)):
            return self._draw_circle(rgb_canvas, trials=trials + 1)
        colors = [[255, 0, 0], [0, 255, 0], [0, 0, 255]]
        color = list(colors[np.random.randint(0, 3)])
        x1, y1, x2, y2 = map(
            int, convert_box([x, y, 2 * radius, 2 * radius], out_format="x1y1x2y2")
        )
        return cv2.circle(
            rgb_canvas, center=(x, y), radius=radius, color=color, thickness=-1
        ), {"box": [x1, y1, x2, y2], "category": "circle"}

    def _draw_rectangle(self, rgb_canvas, trials=0):
        if trials > 100:
            return rgb_canvas, {}
        canvas = np.uint8(~np.all(rgb_canvas == [255, 255, 255], axis=-1))
        smaller_side = min(self.height, self.width)
        min_dim = 0.15 * smaller_side
        max_dim = 0.6 * smaller_side
        h = int(np.random.randint(low=min_dim, high=max_dim)) / 2
        w = int(np.random.randint(low=min_dim, high=max_dim)) / 2
        x = int(np.random.randint(low=w + 1, high=(self.width - w - 1)))
        y = int(np.random.randint(low=h + 1, high=(self.height - h - 1)))
        x1, y1, x2, y2 = map(int, convert_box([x, y, w, h], out_format="x1y1x2y2"))
        new_canvas = np.zeros_like(canvas)
        new_canvas = cv2.rectangle(
            new_canvas, pt1=(x1, y1), pt2=(x2, y2), color=1, thickness=-1
        )
        if np.any(np.logical_and(new_canvas, canvas)):
            return self._draw_rectangle(rgb_canvas, trials=trials + 1)
        colors = [[255, 0, 0], [0, 255, 0], [0, 0, 255]]
        color = list(colors[np.random.randint(0, 3)])
        return cv2.rectangle(
            rgb_canvas, pt1=(x1, y1), pt2=(x2, y2), color=color, thickness=-1
        ), {"box": [x1, y1, x2, y2], "category": "rectangle"}

    def make_dataset(
        self,
        max_objects_per_image: int = 10,
        n_data_samples: int = 128,
        dump_dir: str = "./dataset/",
        *args,
        **kwargs
    ):
        super().make_dataset(dump_dir=dump_dir, *args, **kwargs)
        logging.info("Generating Dataset...")
        for i in tqdm(range(n_data_samples)):
            image_name = "{}.png".format(i)
            objects = []
            image_path = os.path.join(self.images_dir, image_name)
            rgb_canvas = (
                np.ones(shape=[self.height, self.width, 3], dtype=np.uint8) * 255
            )
            num_objects = np.random.randint(low=2, high=max_objects_per_image + 1)
            for i in range(num_objects):
                draw_function = np.random.choice(
                    [self._draw_circle, self._draw_rectangle]
                )
                _, annotation = draw_function(rgb_canvas)
                if annotation == {}:
                    continue
                objects.append(annotation)
            imsave(image_path, rgb_canvas, check_contrast=False)
            self.dataset[image_name] = objects
        utils.dump_dictionary_as_json(
            self.dataset, json_file_path=os.path.join(self.data_dir, "annotation.json")
        )

    def get_dataset(self, *args, **kwargs):
        return super().get_dataset(*args, **kwargs)

    def plot_samples(self, n_samples: int = 5, *args, **kwargs):
        super().plot_samples(*args, **kwargs)
        items = (
            list(
                utils.load_json_as_dict(
                    os.path.join(self.data_dir, "annotation.json")
                ).items()
            )[:n_samples]
            if self.dataset == {}
            else list(self.dataset.items())[:n_samples]
        )
        for image_path, annotation in items:
            boxes, categories = [], []
            for obj in annotation:
                boxes.append(obj["box"])
                categories.append(obj["category"])
            image = imread(os.path.join(self.images_dir, image_path))
            image = draw_boxes(image, boxes, categories)
            plt.figure(figsize=(8, 6))
            plt.imshow(image)
            plt.axis("off")

    def create_tfrecords(
        self, val_split: float = 0.2, samples_per_shard: int = 64, *args, **kwargs
    ):
        tfrecord_writer = TFrecordWriter(
            self.images_dir, self.dataset, self.label_map, prefix=self.dataset_name
        )
        tfrecord_writer.write_tfrecords(
            val_split,
            samples_per_shard,
            output_dir=os.path.join(self.data_dir, "tfrecords"),
        )
