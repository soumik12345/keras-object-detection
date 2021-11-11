import tensorflow as tf

from .base import DataLoader


class YOLOv1DataLoader(DataLoader):
    def __init__(
        self,
        dataset_path,
        image_size: int = 448,
        grid_size: int = 7,
        stride: int = 64,
        n_boxes_per_grid: int = 2,
        n_classes: int = 2,
        run_sanity_checks: bool = False,
    ) -> None:
        super().__init__(
            dataset_path,
            image_size=image_size,
            grid_size=grid_size,
            stride=stride,
            n_boxes_per_grid=n_boxes_per_grid,
            n_classes=n_classes,
            run_sanity_checks=run_sanity_checks,
        )
