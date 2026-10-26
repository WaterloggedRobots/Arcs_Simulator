# Test script for the OpenCV camera type: no camera or depth sensor needed.
# Outputs random points so you can check the 3D spheres, the Tracked Points
# list and Look At's primary target button.
#
#   primary_target = [x_px, y_px, depth_m]          -> large red sphere
#   targets        = [[x_px, y_px, depth_m], ...]   -> small yellow spheres
#
# New random points are picked every NEW_EVERY_S seconds, so the spheres jump
# around slowly enough to watch.
import random
import time

import numpy as np

WIDTH, HEIGHT = 640, 480
NUM_TARGETS = 5
DEPTH_M = (0.5, 2.0)
NEW_EVERY_S = 1.0

primary_target = None
targets = []
last_error = ""

_last_pick = 0.0


def _random_point():
    return [
        float(random.uniform(0, WIDTH)),
        float(random.uniform(0, HEIGHT)),
        float(random.uniform(*DEPTH_M)),
    ]


def get_frame():
    """Black frame with the points drawn, plus fresh random targets."""
    global primary_target, targets, _last_pick
    now = time.time()
    if primary_target is None or now - _last_pick >= NEW_EVERY_S:
        primary_target = _random_point()
        targets = [_random_point() for _ in range(NUM_TARGETS)]
        _last_pick = now
    frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    try:
        import cv2
        for x, y, _d in targets:
            cv2.circle(frame, (int(x), int(y)), 6, (40, 220, 255), -1)
        x, y, _d = primary_target
        cv2.circle(frame, (int(x), int(y)), 10, (40, 40, 230), -1)
    except Exception:
        pass
    return frame
