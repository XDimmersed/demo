# Multimodal Patrol Demo

This repository provides a lightweight PC demo that replays synchronized RGB images and point clouds, performs 2D detection, fuses simple 3D distance estimates, and triggers an alert when a target lingers inside a configured danger zone. The code is structured to run on CPU today while leaving an interface for a future Ascend backend.

## Project structure
```
multimodal_patrol_demo/
  README.md
  requirements.txt
  config.yaml
  run_demo.py
  demo/
  data/
```

## Setup
1. Create a Python 3.9+ environment and install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Prepare a sequence under `data/` or use the provided KITTI conversion helper:
   ```bash
   python scripts/prepare_dataset.py \
     --kitti_root /path/to/2011_09_26_drive_xxxx_sync \
     --output_root data/demo_sequence
   ```
3. Adjust `config.yaml` paths if needed.

## Running the demo
```bash
python run_demo.py
```
The RGB view shows detections, fusion status, and alert text; the point cloud view is rendered with Open3D. Keyboard controls: `q`/`Esc` quit, `Space` pauses playback.

## Ascend migration
`demo/inference/dummy_ascend_backend.py` defines the placeholder backend that matches the CPU interface, allowing drop-in replacement when integrating Ascend inference in the future.
