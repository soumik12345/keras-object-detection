import tensorflow as tf

from .base import DataLoader
from ..utils import box_utils


class YOLOv1DataLoader(DataLoader):
    def __init__(
        self,
        dataset_path,
        image_size: int = 448,
        grid_size: int = 7,
        stride: int = 64,
        n_boxes_per_grid: int = 2,
        predictions_per_cell: int = 2,
        n_classes: int = 2,
        run_sanity_checks: bool = False,
    ) -> None:
        super().__init__(
            dataset_path,
            image_size=image_size,
            grid_size=grid_size,
            stride=stride,
            n_boxes_per_grid=n_boxes_per_grid,
            predictions_per_cell=predictions_per_cell,
            n_classes=n_classes,
            run_sanity_checks=run_sanity_checks,
        )
        self.output_dim = self.predictions_per_cell * 5 + self.n_classes

    def preprocess_outputs(self, boxes, labels):
        boxes = tf.cast(boxes, dtype=tf.float32)
        boxes_xywh = box_utils.convert_to_xywh(boxes)
        classes = tf.one_hot(labels, depth=self.n_classes, dtype=tf.float32)
        num_objects = tf.shape(classes)[0]
        pc = tf.ones(shape=[num_objects, 1], dtype=tf.float32)
        box_centers = boxes_xywh[:, :2]
        box_wh = boxes_xywh[:, 2:]
        grid_offset = tf.math.floordiv(box_centers, self.stride)
        normalized_box_centers = box_centers / self.stride - grid_offset
        normalized_wh = box_wh / tf.constant(
            [self.image_size, self.image_size], dtype=tf.float32
        )
        label_shape = [self.grid_size, self.grid_size, self.output_dim]
        label = tf.zeros(shape=label_shape, dtype=tf.float32)
        normalized_box = tf.concat([normalized_box_centers, normalized_wh], axis=-1)
        targets = tf.concat([pc, pc, normalized_box, normalized_box, classes], axis=-1)
        targets = tf.reshape(targets, shape=[1, num_objects, self.output_dim])
        grid_offset_reversed = tf.reverse(grid_offset, axis=[1])
        indices = tf.cast(grid_offset_reversed, dtype=tf.int32)
        indices = tf.reshape(indices, shape=[1, num_objects, 2])
        return tf.tensor_scatter_nd_update(label, indices, targets)
