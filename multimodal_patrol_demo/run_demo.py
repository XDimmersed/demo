import time
from pathlib import Path

from demo.config import load_config
from demo.data_simulator import DataSimulator
from demo.fusion.fusion_engine import FusionEngine
from demo.fusion.zone_logic import ZoneMonitor
from demo.inference import create_backend
from demo.llm.llm_client import create_llm_client
from demo.llm.report_generator import ReportGenerator
from demo.time_utils import get_play_interval
from demo.types import AlertEvent, PatrolReport
from demo.ui.controller import handle_keyboard
from demo.ui.o3d_viewer import PointCloudView
from demo.ui.opencv_ui import RGBView


def main():
    config = load_config("config.yaml")

    llm_client = create_llm_client(config.llm)
    report_generator = ReportGenerator(llm_client)

    simulator = DataSimulator(config.demo)
    backend = create_backend(config.model)
    backend.load()

    fusion_engine = FusionEngine(config.fusion)
    zone_monitor = ZoneMonitor(config.alert.stay_time_threshold_s)

    patrol_report = PatrolReport(events=[], start_time=0.0, end_time=0.0)
    last_event_text = ""
    first_frame_ts = None
    last_frame_ts = None
    prev_alert = False

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

        if first_frame_ts is None:
            first_frame_ts = rgb_frame.timestamp

        det_result = backend.infer(rgb_frame)
        targets = fusion_engine.fuse(det_result, pcd_frame, rgb_frame.image.shape[:2])
        alert, duration_s = zone_monitor.update(targets, rgb_frame.timestamp)

        event_text = ""
        if alert and not prev_alert:
            person_targets = [
                t for t in targets if t.class_name == "person" and t.in_danger_zone
            ]
            if person_targets:
                t = person_targets[0]
                event = AlertEvent(
                    timestamp=rgb_frame.timestamp,
                    class_name=t.class_name,
                    distance_m=t.distance_m,
                    zone_name="danger_zone",
                    duration_s=duration_s,
                )
                patrol_report.events.append(event)
                event_text = report_generator.describe_single_event(event)
                last_event_text = event_text or last_event_text
                print("[LLM 事件描述]", last_event_text)

        last_frame_ts = rgb_frame.timestamp
        prev_alert = alert

        display_text = event_text or last_event_text
        rgb_view.render(rgb_frame, det_result, targets, alert, event_text=display_text)
        pcd_view.render(pcd_frame)

        time.sleep(interval)

    if first_frame_ts is not None and last_frame_ts is not None:
        patrol_report.start_time = first_frame_ts
        patrol_report.end_time = last_frame_ts
        patrol_report = report_generator.summarize_report(patrol_report)
        print("==== 本次巡检总结 ====")
        print(patrol_report.summary_text)

        output_dir = Path("output")
        output_dir.mkdir(parents=True, exist_ok=True)
        report_path = output_dir / "report.txt"
        report_path.write_text(patrol_report.summary_text, encoding="utf-8")


if __name__ == "__main__":
    main()
