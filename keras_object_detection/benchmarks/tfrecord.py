"""TFRecord writer class.

The TFRecordWriter class can be used to generate TFRecord files from a dataset.

Typical usage example:

```python
tfrecord_writer = TFrecordWriter(
    images_dir, dataset, label_map, prefix=dataset_name
)
tfrecord_writer.write_tfrecords(
    val_split, samples_per_shard, output_dir
)
```
"""


import os
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from absl import logging
from typing import List, Dict

from keras_object_detection import utils


class TFrecordWriter:
    """TFRecord writer class.

    The TFRecordWriter class can be used to generate TFRecord files from a dataset.

    Attributes:
        images_dir: Directory path containing images.
        annotations: Annotation dictionary.
        label_map: Dictonary mapping labels to indices.
        prefix: Prefix for TFRecord file names.
    """

    def __init__(
        self,
        images_dir,
        annotations: Dict,
        label_map: Dict,
        prefix: str = "",
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.images_dir = images_dir
        self.annotations = annotations
        self.prefix = prefix
        self.label_map = label_map

    def _make_example(self, image, boxes, labels):
        feature = {
            "image": tf.train.Feature(bytes_list=tf.train.BytesList(value=[image])),
            "xmins": tf.train.Feature(float_list=tf.train.FloatList(value=boxes[:, 0])),
            "ymins": tf.train.Feature(float_list=tf.train.FloatList(value=boxes[:, 1])),
            "xmaxs": tf.train.Feature(float_list=tf.train.FloatList(value=boxes[:, 2])),
            "ymaxs": tf.train.Feature(float_list=tf.train.FloatList(value=boxes[:, 3])),
            "labels": tf.train.Feature(int64_list=tf.train.Int64List(value=labels)),
        }
        return tf.train.Example(features=tf.train.Features(feature=feature))

    def _read_image(self, image_path):
        with tf.io.gfile.GFile(image_path, "rb") as fp:
            image = fp.read()
        return image

    def _write_tfrecords_with_labels(
        self, images: List, samples_per_shard: int, output_dir, split: str
    ):
        dump_dir = os.path.join(output_dir, split)
        utils.make_directory(dump_dir)
        num_tfrecords = len(images) // samples_per_shard
        if len(images) % samples_per_shard:
            num_tfrecords += 1
        image_shards = utils.split_list(images, chunk_size=samples_per_shard)
        lower_limit, upper_limit = 0, samples_per_shard
        for index in range(num_tfrecords):
            image_shard = image_shards[index]
            file_name = "{}-{}-{:04d}-{:04d}.tfrec".format(
                self.prefix, split, lower_limit, upper_limit
            )
            lower_limit += samples_per_shard
            upper_limit += len(image_shard)
            logging.info(f"Shard Size: {len(image_shard)}")
            logging.info(f"Writing {file_name}...")
            with tf.io.TFRecordWriter(os.path.join(dump_dir, file_name)) as writer:
                for sample_index in tqdm(range(len(image_shard))):
                    image = self._read_image(
                        os.path.join(self.images_dir, image_shard[sample_index])
                    )
                    boxes, labels = [], []
                    for obj in self.annotations[image_shard[sample_index]]:
                        boxes.append(obj["box"])
                        labels.append(self.label_map[obj["category"]])
                    boxes = np.array(boxes, dtype=np.float32)
                    labels = np.array(labels, dtype=np.int32)
                    example = self._make_example(image, boxes, labels)
                    writer.write(example.SerializeToString())

    def write_tfrecords(
        self, val_split: float = 0.2, samples_per_shard: int = 64, output_dir: str = ""
    ) -> None:
        """Fetches rows from a Smalltable.

        Retrieves rows pertaining to the given keys from the Table instance
        represented by table_handle.  String keys will be UTF-8 encoded.

        Args:
            val_split: Validation split.
            samples_per_shard: Number of data samples per shard.
            output_dir: Directory that contains the tfrecord files.
        """
        utils.make_directory(output_dir)
        all_images = list(self.annotations.keys())
        split_index = int(len(all_images) * (1 - val_split))
        train_images = all_images[:split_index]
        print(f"Num train files: {len(train_images)}")
        logging.info("Writing Train TFRecords...")
        self._write_tfrecords_with_labels(
            train_images, samples_per_shard, output_dir, "train"
        )
        val_images = all_images[split_index:]
        logging.info("Writing Validation TFRecords...")
        self._write_tfrecords_with_labels(
            val_images, samples_per_shard, output_dir, "val"
        )
