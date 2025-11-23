from typing import List

import numpy as np

from demo.types import DetectionResult, PointCloudFrame, Target3D


class FusionEngine:
    def __init__(self, config) -> None:
        self.config = config

    def _in_danger_zone(self, bbox_xyxy: np.ndarray, image_shape) -> bool:
        h, w = image_shape
        x1, y1, x2, y2 = bbox_xyxy
        cx = (x1 + x2) / 2.0 / w
        cy = (y1 + y2) / 2.0 / h
        dz = self.config.danger_zone
        return dz.x_min <= cx <= dz.x_max and dz.y_min <= cy <= dz.y_max

    def _estimate_distance(self, points: np.ndarray) -> float:
        if points.size == 0:
            return float("nan")
        radii = np.linalg.norm(points, axis=1)
        radii = radii[~np.isnan(radii)]
        if radii.size == 0:
            return float("nan")
        filtered = radii
        if self.config.min_distance_m > 0:
            filtered = filtered[filtered >= self.config.min_distance_m]
        if self.config.max_distance_m > 0:
            filtered = filtered[filtered <= self.config.max_distance_m]
        if filtered.size == 0:
            filtered = radii
        return float(np.median(filtered))

    def fuse(
        self, det: DetectionResult, pcd: PointCloudFrame, image_shape
    ) -> List[Target3D]:
        targets: List[Target3D] = []
        for detection in det.detections:
            in_zone = self._in_danger_zone(detection.bbox_xyxy, image_shape)
            distance = self._estimate_distance(pcd.points)
            targets.append(
                Target3D(
                    class_name=detection.class_name,
                    distance_m=distance,
                    in_danger_zone=in_zone,
                    bbox_xyxy=detection.bbox_xyxy,
                    timestamp=det.timestamp,
                )
            )
        return targets
