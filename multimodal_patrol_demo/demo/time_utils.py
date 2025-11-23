import time


def get_play_interval(play_fps: int) -> float:
    if play_fps <= 0:
        return 0.0
    return 1.0 / float(play_fps)


def current_time_s() -> float:
    return time.time()
