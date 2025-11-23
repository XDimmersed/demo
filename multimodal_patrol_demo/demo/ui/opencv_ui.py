from typing import List

import cv2
import numpy as np

from demo.types import DetectionResult, RGBFrame, Target3D


class RGBView:
    def __init__(self, config) -> None:
        self.config = config
        self.window_name = config.ui.window_name_rgb

    def _draw_detections(self, img: np.ndarray, det: DetectionResult):
        for d in det.detections:
            x1, y1, x2, y2 = d.bbox_xyxy.astype(int)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), self.config.ui.line_thickness)
            label = f"{d.class_name} {d.confidence:.2f}"
            cv2.putText(
                img,
                label,
                (x1, max(0, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                self.config.ui.font_scale,
                (0, 255, 0),
                self.config.ui.line_thickness,
                lineType=cv2.LINE_AA,
            )

    def _draw_targets(self, img: np.ndarray, targets: List[Target3D]):
        for t in targets:
            x1, y1, x2, y2 = t.bbox_xyxy.astype(int)
            color = (0, 0, 255) if t.in_danger_zone else (255, 255, 0)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, self.config.ui.line_thickness)
            label = f"{t.class_name} {t.distance_m:.1f}m"
            cv2.putText(
                img,
                label,
                (x1, min(img.shape[0] - 5, y2 + 15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                self.config.ui.font_scale,
                color,
                self.config.ui.line_thickness,
                lineType=cv2.LINE_AA,
            )

    def _draw_status(self, img: np.ndarray, alert: bool):
        text = "ALERT" if alert else "NORMAL"
        color = (0, 0, 255) if alert else (0, 255, 0)
        cv2.putText(
            img,
            text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            self.config.ui.font_scale + 0.4,
            color,
            max(1, self.config.ui.line_thickness),
            lineType=cv2.LINE_AA,
        )

    def render(
        self, frame: RGBFrame, det: DetectionResult, targets: List[Target3D], alert: bool
    ):
        canvas = frame.image.copy()
        self._draw_detections(canvas, det)
        self._draw_targets(canvas, targets)
        self._draw_status(canvas, alert)
        cv2.imshow(self.window_name, canvas)
