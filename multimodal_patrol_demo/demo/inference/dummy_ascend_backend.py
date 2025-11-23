from demo.inference.backend_base import InferenceBackend
from demo.types import DetectionResult, RGBFrame


class DummyAscendBackend(InferenceBackend):
    def __init__(self, config) -> None:
        self.config = config

    def load(self):
        # Placeholder for Ascend-specific initialization
        return None

    def infer(self, frame: RGBFrame) -> DetectionResult:
        return DetectionResult(detections=[], timestamp=frame.timestamp)
