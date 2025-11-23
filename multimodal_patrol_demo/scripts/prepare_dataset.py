import argparse
import shutil
from datetime import datetime
from pathlib import Path
from typing import List

def parse_timestamps(ts_file: Path) -> List[float]:
    raw_times: List[datetime] = []
    with ts_file.open("r", encoding="utf-8") as f:
        for line in f:
            text = line.strip()
            if not text:
                continue
            if "." in text:
                prefix, frac = text.split(".", maxsplit=1)
                frac = (frac + "000000")[:6]
                text = f"{prefix}.{frac}"
            raw_times.append(datetime.strptime(text, "%Y-%m-%d %H:%M:%S.%f"))
    if not raw_times:
        raise ValueError("No timestamps found in file")
    base = raw_times[0]
    return [float((t - base).total_seconds()) for t in raw_times]


def convert_sequence(kitti_root: Path, output_root: Path):
    image_dir = kitti_root / "image_02" / "data"
    lidar_dir = kitti_root / "velodyne_points" / "data"
    ts_file = kitti_root / "timestamps.txt"

    if not image_dir.exists() or not lidar_dir.exists():
        raise FileNotFoundError("KITTI directories not found. Ensure image_02/data and velodyne_points/data exist.")

    timestamps = parse_timestamps(ts_file)
    image_files = sorted(p for p in image_dir.iterdir() if p.is_file())
    lidar_files = sorted(p for p in lidar_dir.iterdir() if p.is_file())

    if not (len(image_files) == len(lidar_files) == len(timestamps)):
        raise ValueError(
            "Input lengths mismatch: images {}, lidars {}, timestamps {}".format(
                len(image_files), len(lidar_files), len(timestamps)
            )
        )

    rgb_out = output_root / "rgb"
    pcd_out = output_root / "pointcloud"
    rgb_out.mkdir(parents=True, exist_ok=True)
    pcd_out.mkdir(parents=True, exist_ok=True)

    for idx, (img, lidar) in enumerate(zip(image_files, lidar_files)):
        fname = f"{idx:06d}"
        shutil.copy2(img, rgb_out / f"{fname}{img.suffix}")
        # KITTI lidar files are .bin and already in expected format
        shutil.copy2(lidar, pcd_out / f"{fname}{lidar.suffix}")

    ts_out = output_root / "timestamps.txt"
    with ts_out.open("w", encoding="utf-8") as f:
        for ts in timestamps:
            f.write(f"{ts}\n")
    print(f"Converted {len(image_files)} frames to {output_root}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare KITTI raw sequence for demo replay.")
    parser.add_argument("--kitti_root", type=Path, required=True, help="Path to KITTI drive directory (e.g., 2011_09_26_drive_xxxx_sync)")
    parser.add_argument("--output_root", type=Path, required=True, help="Output directory for processed sequence")
    args = parser.parse_args()

    convert_sequence(args.kitti_root, args.output_root)
