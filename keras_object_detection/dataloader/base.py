import os
from typing import List
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod

import tensorflow as tf

from ..utils import box_utils


class DataLoader(ABC):
    def __init__(
        self,
        dataset_path,
        image_size: int = 448,
        grid_size: int = 7,
        n_boxes_per_grid: int = 2,
        predictions_per_cell: int = 2,
        n_classes: int = 2,
        run_sanity_checks: bool = False,
    ) -> None:
        super().__init__()
        self.dataset_path = dataset_path
        self.image_size = image_size
        self.grid_size = grid_size
        self.stride = image_size // grid_size,
        self.n_boxes_per_grid = n_boxes_per_grid
        self.predictions_per_cell = predictions_per_cell
        self.n_classes = n_classes
        self.augmentation_functions = []
        self.run_sanity_checks = run_sanity_checks
        self._get_tfrecord_files()
        self._set_feature_description()

    def add_augmentation(self, augmentation_fn):
        self.augmentation_functions.append(augmentation_fn)

    def set_augmentations(self, augmentation_fns: List):
        self.augmentation_functions = augmentation_fns

    def _get_tfrecord_files(self):
        train_tfrecord_pattern = os.path.join(self.dataset_path, "train", "*")
        val_tfrecord_pattern = os.path.join(self.dataset_path, "val", "*")
        self.train_tfrecords = tf.data.Dataset.list_files(train_tfrecord_pattern)
        self.val_tfrecords = tf.data.Dataset.list_files(val_tfrecord_pattern)
        if self.run_sanity_checks:
            for x_train, x_val in zip(self.train_tfrecords, self.val_tfrecords):
                print(x_train, x_val)

    def _set_feature_description(self):
        self.feature_description = {
            "image": tf.io.FixedLenFeature([], tf.string),
            "xmins": tf.io.VarLenFeature(tf.float32),
            "ymins": tf.io.VarLenFeature(tf.float32),
            "xmaxs": tf.io.VarLenFeature(tf.float32),
            "ymaxs": tf.io.VarLenFeature(tf.float32),
            "labels": tf.io.VarLenFeature(tf.int64),
        }

    def parse_example(self, example_proto):
        parsed_example = tf.io.parse_single_example(
            example_proto, self.feature_description
        )
        image = tf.io.decode_image(parsed_example["image"], channels=3)
        image = tf.cast(image, dtype=tf.float32)
        image.set_shape([None, None, 3])
        boxes = tf.stack(
            [
                tf.sparse.to_dense(parsed_example["xmins"]),
                tf.sparse.to_dense(parsed_example["ymins"]),
                tf.sparse.to_dense(parsed_example["xmaxs"]),
                tf.sparse.to_dense(parsed_example["ymaxs"]),
            ],
            axis=-1,
        )
        labels = tf.sparse.to_dense(parsed_example["labels"])
        return image, boxes, labels

    @abstractmethod
    def preprocess_outputs(self, boxes, labels):
        pass

    def _preprocess_image(self, example_proto):
        image, boxes, classes = self.parse_example(example_proto)
        image = (image - 127.5) / 127.5
        return image, boxes, classes

    def parse_fn(self, image, boxes, classes):
        label = self.preprocess_outputs(boxes, classes)
        return image, label

    @staticmethod
    def run_dataset_sanity_check(dataset, label_map):
        for image, boxes, class_ids in dataset.take(1):
            classes = [
                {v: k for k, v in label_map.items()}[int(x)] for x in class_ids.numpy()
            ]
            plot = box_utils.draw_boxes(image.numpy(), boxes.numpy(), classes)
            plt.figure(figsize=(8, 6))
            plt.imshow(plot)
            plt.axis("off")

    def build_dataset(
        self,
        cycle_length: int = 4,
        block_length: int = 16,
        buffer_size: int = 512,
        batch_size: int = 8,
        is_train: bool = False,
        label_map=None,
    ):
        tfrecords = self.train_tfrecords if is_train else self.val_tfrecords
        dataset = tfrecords.interleave(
            tf.data.TFRecordDataset,
            cycle_length=cycle_length,
            block_length=block_length,
            num_parallel_calls=tf.data.AUTOTUNE,
        )
        dataset = dataset.shuffle(buffer_size)
        dataset = dataset.map(
            self._preprocess_image, num_parallel_calls=tf.data.AUTOTUNE
        )
        for augmentation_fn in self.augmentation_functions:
            dataset = dataset.map(augmentation_fn, num_parallel_calls=tf.data.AUTOTUNE)
        if self.run_sanity_checks:
            if label_map is not None:
                DataLoader.run_dataset_sanity_check(dataset, label_map)
            else:
                print("Unable to run sanity check, please pass a valid label map")
        dataset = dataset.map(self.parse_fn, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.batch(batch_size)
        return dataset
