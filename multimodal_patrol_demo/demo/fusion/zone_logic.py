from typing import List, Optional

from demo.types import Target3D


class ZoneMonitor:
    def __init__(self, stay_time_threshold_s: float):
        self.stay_time_threshold_s = stay_time_threshold_s
        self.current_target_enter_time: Optional[float] = None
        self.current_in_zone: bool = False

    def update(self, targets: List[Target3D], current_time: float) -> tuple[bool, float]:
        """
        Returns:
            A tuple of (is_alert, dwell_time_seconds).
        """

        person_in_zone = any(
            t.class_name == "person" and t.in_danger_zone for t in targets
        )
        if person_in_zone:
            if not self.current_in_zone:
                self.current_target_enter_time = current_time
                self.current_in_zone = True
        else:
            self.current_target_enter_time = None
            self.current_in_zone = False

        dwell = 0.0
        if self.current_in_zone and self.current_target_enter_time is not None:
            dwell = current_time - self.current_target_enter_time
            if dwell >= self.stay_time_threshold_s:
                return True, dwell
        return False, dwell
