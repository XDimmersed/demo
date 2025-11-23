import time

from demo.config import load_config
from demo.data_simulator import DataSimulator
from demo.fusion.fusion_engine import FusionEngine
from demo.fusion.zone_logic import ZoneMonitor
from demo.inference import create_backend
from demo.time_utils import get_play_interval
from demo.ui.controller import handle_keyboard
from demo.ui.o3d_viewer import PointCloudView
from demo.ui.opencv_ui import RGBView


def main():
    config = load_config("config.yaml")

    simulator = DataSimulator(config.demo)
    backend = create_backend(config.model)
    backend.load()

    fusion_engine = FusionEngine(config.fusion)
    zone_monitor = ZoneMonitor(config.alert.stay_time_threshold_s)

    rgb_view = RGBView(config)
    pcd_view = PointCloudView(config.ui.window_name_pcd)

    paused = False
    interval = get_play_interval(config.demo.play_fps)

    for rgb_frame, pcd_frame in simulator.iter_frames():
        action = handle_keyboard()
        if action == "quit":
            break
        if action == "toggle_pause":
            paused = not paused

        if paused:
            time.sleep(interval)
            continue

        det_result = backend.infer(rgb_frame)
        targets = fusion_engine.fuse(det_result, pcd_frame, rgb_frame.image.shape[:2])
        alert = zone_monitor.update(targets, rgb_frame.timestamp)

        rgb_view.render(rgb_frame, det_result, targets, alert)
        pcd_view.render(pcd_frame)

        time.sleep(interval)


if __name__ == "__main__":
    main()
