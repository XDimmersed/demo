from dataclasses import dataclass
from pathlib import Path
from typing import List

import yaml


@dataclass
class DemoConfig:
    sequence_root: str
    rgb_dir: str
    pointcloud_dir: str
    timestamps_file: str
    play_fps: int


@dataclass
class ModelConfig:
    backend: str
    weights_path: str
    input_size: List[int]


@dataclass
class DangerZoneConfig:
    x_min: float
    x_max: float
    y_min: float
    y_max: float


@dataclass
class FusionConfig:
    use_projection: bool
    danger_zone: DangerZoneConfig
    min_distance_m: float
    max_distance_m: float


@dataclass
class AlertConfig:
    stay_time_threshold_s: float


@dataclass
class UIConfig:
    window_name_rgb: str
    window_name_pcd: str
    font_scale: float
    line_thickness: int


@dataclass
class AppConfig:
    demo: DemoConfig
    model: ModelConfig
    fusion: FusionConfig
    alert: AlertConfig
    ui: UIConfig


class ConfigLoader:
    def __init__(self, path: str | Path):
        self.path = Path(path)

    def load(self) -> AppConfig:
        with self.path.open("r", encoding="utf-8") as f:
            cfg_dict = yaml.safe_load(f)
        danger_zone_cfg = DangerZoneConfig(**cfg_dict["fusion"]["danger_zone"])
        return AppConfig(
            demo=DemoConfig(**cfg_dict["demo"]),
            model=ModelConfig(**cfg_dict["model"]),
            fusion=FusionConfig(
                danger_zone=danger_zone_cfg,
                use_projection=cfg_dict["fusion"].get("use_projection", False),
                min_distance_m=cfg_dict["fusion"].get("min_distance_m", 0.0),
                max_distance_m=cfg_dict["fusion"].get("max_distance_m", 0.0),
            ),
            alert=AlertConfig(**cfg_dict["alert"]),
            ui=UIConfig(**cfg_dict["ui"]),
        )


def load_config(path: str | Path = "config.yaml") -> AppConfig:
    return ConfigLoader(path).load()
