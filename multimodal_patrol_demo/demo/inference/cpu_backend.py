from typing import List

import numpy as np
import torch

from demo.inference.backend_base import InferenceBackend
from demo.config import ModelConfig
from demo.types import Detection, DetectionResult, RGBFrame


class CPUBackend(InferenceBackend):
    def __init__(self, config: ModelConfig) -> None:
        self.config = config
        self.model = None
        if config.use_gpu and torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

    def load(self):
        weights_path = self.config.weights_path
        self.model = torch.hub.load(
            "ultralytics/yolov5",
            "custom" if weights_path else "yolov5s",
            path=weights_path if weights_path else None,
            pretrained=weights_path is None,
        )
        self.model.to(self.device)
        self.model.eval()

    def _postprocess(self, results) -> List[Detection]:
        detections: List[Detection] = []
        if results is None:
            return detections
        for *xyxy, conf, cls in results.xyxy[0].tolist():
            class_id = int(cls)
            bbox = np.array(xyxy, dtype=np.float32)
            detections.append(
                Detection(
                    class_id=class_id,
                    class_name=self.model.names[class_id],
                    confidence=float(conf),
                    bbox_xyxy=bbox,
                )
            )
        return detections

    def infer(self, frame: RGBFrame) -> DetectionResult:
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        img_rgb = frame.image[:, :, ::-1]
        tensor = (
            torch.from_numpy(img_rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        )
        tensor = tensor.to(self.device)
        results = self.model(tensor, size=self.config.input_size[0])
        if hasattr(results, "to"):
            results = results.to("cpu")
        detections = self._postprocess(results)
        return DetectionResult(detections=detections, timestamp=frame.timestamp)
