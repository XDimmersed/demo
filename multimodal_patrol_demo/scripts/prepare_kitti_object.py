from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
from typing import List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare KITTI object frames for the multimodal demo",
    )
    parser.add_argument(
        "--kitti_object_root",
        required=True,
        help="Root path of the KITTI object dataset (containing training/testing)",
    )
    parser.add_argument(
        "--output_root",
        default="data/demo_sequence",
        help="Output root directory for the demo sequence",
    )
    parser.add_argument(
        "--split",
        default="training",
        choices=["training", "testing"],
        help="Dataset split to use",
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=300,
        help="Maximum number of frames to copy/link",
    )
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def link_or_copy(src: Path, dst: Path) -> None:
    if dst.exists():
        dst.unlink()
    try:
        os.symlink(src, dst)
    except OSError:
        shutil.copyfile(src, dst)


def list_images(image_dir: Path, num_frames: int) -> List[Path]:
    all_images = sorted(image_dir.glob("*.png"))
    if not all_images:
        raise FileNotFoundError(f"No PNG files found in {image_dir}")
    return all_images[:num_frames]


def main() -> None:
    args = parse_args()
    root = Path(args.kitti_object_root)
    image_dir = root / args.split / "image_2"
    velodyne_dir = root / args.split / "velodyne"

    if not image_dir.exists() or not velodyne_dir.exists():
        raise FileNotFoundError(
            "Required KITTI object folders not found. Ensure data_object_image_2.zip "
            "and data_object_velodyne.zip are extracted under the provided root."
        )

    output_root = Path(args.output_root)
    rgb_out = output_root / "rgb"
    pcd_out = output_root / "pointcloud"
    ensure_dir(rgb_out)
    ensure_dir(pcd_out)

    images = list_images(image_dir, args.num_frames)

    kept = 0
    for img_path in images:
        stem = img_path.stem
        bin_path = velodyne_dir / f"{stem}.bin"
        if not bin_path.exists():
            print(f"[skip] Missing velodyne for {stem}")
            continue
        link_or_copy(img_path, rgb_out / f"{stem}.png")
        link_or_copy(bin_path, pcd_out / f"{stem}.bin")
        kept += 1

    if kept == 0:
        raise RuntimeError("No frames were prepared; check dataset paths and content.")

    timestamps_file = output_root / "timestamps.txt"
    with timestamps_file.open("w", encoding="utf-8") as f:
        for i in range(kept):
            f.write(f"{i * 0.1:.1f}\n")

    print(f"Prepared {kept} frames to {output_root} using split '{args.split}'.")


if __name__ == "__main__":
    main()
