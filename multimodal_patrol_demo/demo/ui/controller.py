import cv2


def handle_keyboard():
    key = cv2.waitKey(1) & 0xFF
    if key in (ord("q"), 27):
        return "quit"
    if key == ord(" "):
        return "toggle_pause"
    return None
