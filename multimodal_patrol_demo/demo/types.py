from dataclasses import dataclass
from typing import List, Optional

import numpy as np


@dataclass
class RGBFrame:
    image: np.ndarray
    timestamp: float


@dataclass
class PointCloudFrame:
    points: np.ndarray
    timestamp: float


@dataclass
class Detection:
    class_id: int
    class_name: str
    confidence: float
    bbox_xyxy: np.ndarray


@dataclass
class DetectionResult:
    detections: List[Detection]
    timestamp: float


@dataclass
class Target3D:
    class_name: str
    distance_m: float
    in_danger_zone: bool
    bbox_xyxy: np.ndarray
    timestamp: float


@dataclass
class AlertEvent:
    timestamp: float
    class_name: str
    distance_m: float
    zone_name: str
    duration_s: float
    extra_info: Optional[str] = None


@dataclass
class PatrolReport:
    events: List[AlertEvent]
    start_time: float
    end_time: float
    summary_text: str = ""
