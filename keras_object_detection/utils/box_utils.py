import cv2
import numpy as np
import tensorflow as tf


def convert_to_xywh(boxes):
    x = (1 + boxes[..., 0] + boxes[..., 2]) / 2.0
    y = (1 + boxes[..., 1] + boxes[..., 3]) / 2.0
    w = 1 + boxes[..., 2] - boxes[..., 0]
    h = 1 + boxes[..., 3] - boxes[..., 1]
    return tf.stack([x, y, w, h], axis=-1)


def convert_to_x1y1x2y2(boxes):
    x1 = boxes[..., 0] - boxes[..., 2] / 2.0
    y1 = boxes[..., 1] - boxes[..., 3] / 2.0
    x2 = (boxes[..., 0] + boxes[..., 2] / 2.0) - 1
    y2 = (boxes[..., 1] + boxes[..., 3] / 2.0) - 1
    return tf.stack([x1, y1, x2, y2], axis=-1)


def convert_box(box, out_format):
    assert out_format in ["xywh", "x1y1x2y2"], "Invalid box format"
    if out_format == "xywh":
        x1, y1, x2, y2 = box
        x = (x1 + x2) / 2
        y = (y1 + y2) / 2
        w = x2 - x1
        h = y2 - y1
        return [x, y, w, h]
    x, y, w, h = box
    x1 = x - (w / 2)
    y1 = y - (h / 2)
    x2 = x + (w / 2)
    y2 = y + (h / 2)
    return [x1, y1, x2, y2]


def draw_boxes(image, boxes, categories):
    boxes = np.array(boxes, dtype=np.int32)
    for _box, _cls in zip(boxes, categories):
        text = _cls
        char_len = len(text) * 9
        text_orig = (_box[0] + 5, _box[1] - 6)
        text_bg_xy1 = (_box[0], _box[1] - 20)
        text_bg_xy2 = (_box[0] + char_len, _box[1])
        image = cv2.rectangle(image, text_bg_xy1, text_bg_xy2, [255, 252, 150], -1)
        image = cv2.putText(
            image,
            text,
            text_orig,
            cv2.FONT_HERSHEY_COMPLEX_SMALL,
            0.6,
            [0, 0, 0],
            5,
            lineType=cv2.LINE_AA,
        )
        img = cv2.putText(
            image,
            text,
            text_orig,
            cv2.FONT_HERSHEY_COMPLEX_SMALL,
            0.6,
            [255, 255, 255],
            1,
            lineType=cv2.LINE_AA,
        )
        img = cv2.rectangle(
            image, (_box[0], _box[1]), (_box[2], _box[3]), [30, 15, 30], 1
        )
    return img
