from pathlib import Path
from typing import Generator, Iterable, List, Tuple

import cv2
import numpy as np
import open3d as o3d

from demo.types import PointCloudFrame, RGBFrame


class DataSimulator:
    def __init__(self, config) -> None:
        self.rgb_dir = Path(config.rgb_dir)
        self.pointcloud_dir = Path(config.pointcloud_dir)
        self.timestamps_file = Path(config.timestamps_file)
        self.timestamps = self._load_timestamps()
        self.rgb_files = self._gather_files(self.rgb_dir)
        self.pcd_files = self._gather_files(self.pointcloud_dir)
        self._validate_alignment()

    def __len__(self) -> int:
        return len(self.timestamps)

    def _load_timestamps(self) -> List[float]:
        if not self.timestamps_file.exists():
            raise FileNotFoundError(f"Missing timestamps file: {self.timestamps_file}")
        with self.timestamps_file.open("r", encoding="utf-8") as f:
            return [float(line.strip()) for line in f if line.strip()]

    def _gather_files(self, directory: Path) -> List[Path]:
        if not directory.exists():
            raise FileNotFoundError(f"Data directory not found: {directory}")
        return sorted(p for p in directory.iterdir() if p.is_file())

    def _validate_alignment(self) -> None:
        if not (len(self.rgb_files) == len(self.pcd_files) == len(self.timestamps)):
            raise ValueError(
                "Mismatch between RGB frames, point clouds, and timestamps."
                f" RGB: {len(self.rgb_files)}, PCD: {len(self.pcd_files)}, TS: {len(self.timestamps)}"
            )

    def _read_rgb(self, path: Path) -> np.ndarray:
        img = cv2.imread(str(path))
        if img is None:
            raise ValueError(f"Failed to read image: {path}")
        return img

    def _read_pointcloud(self, path: Path) -> np.ndarray:
        if path.suffix.lower() == ".pcd":
            pcd = o3d.io.read_point_cloud(str(path))
            return np.asarray(pcd.points, dtype=np.float32)
        if path.suffix.lower() == ".bin":
            raw = np.fromfile(path, dtype=np.float32)
            points = raw.reshape(-1, 4)[:, :3]
            return points.astype(np.float32)
        raise ValueError(f"Unsupported point cloud format: {path.suffix}")

    def get_frame(self, index: int) -> Tuple[RGBFrame, PointCloudFrame]:
        if index < 0 or index >= len(self):
            raise IndexError("Frame index out of range")
        timestamp = self.timestamps[index]
        rgb = RGBFrame(image=self._read_rgb(self.rgb_files[index]), timestamp=timestamp)
        pcd = PointCloudFrame(points=self._read_pointcloud(self.pcd_files[index]), timestamp=timestamp)
        return rgb, pcd

    def iter_frames(self) -> Iterable[Tuple[RGBFrame, PointCloudFrame]]:
        for i in range(len(self)):
            yield self.get_frame(i)

    def iter_frames_generator(self) -> Generator[Tuple[RGBFrame, PointCloudFrame], None, None]:
        for i in range(len(self)):
            yield self.get_frame(i)
