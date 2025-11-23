from abc import ABC, abstractmethod

from demo.types import DetectionResult, RGBFrame


class InferenceBackend(ABC):
    @abstractmethod
    def load(self):
        raise NotImplementedError

    @abstractmethod
    def infer(self, frame: RGBFrame) -> DetectionResult:
        raise NotImplementedError
