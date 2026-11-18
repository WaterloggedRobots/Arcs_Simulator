# OpenCV camera script for Intel RealSense D435i.
# Attach on a camera object with type = opencv / script = this file.
#
# GUI click (camera pane) sets target_uv in color-frame pixels.
# Each frame: tracked_cam_pos = [[u_px, v_px, depth_m], ...]
# which waypoints.py turns into a world-space track sphere.

tracked_cam_pos = []
target_uv = None  # (u, v) in color pixels, or None

_pipe = None
_align = None


def set_target(u, v):
    global target_uv
    try:
        target_uv = (int(round(float(u))), int(round(float(v))))
    except Exception:
        target_uv = None


def _ensure():
    global _pipe, _align
    if _pipe is not None:
        return True
    try:
        import pyrealsense2 as rs
    except Exception as e:
        print("d435i: pyrealsense2 missing", e)
        return False
    _pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    _pipe.start(cfg)
    _align = rs.align(rs.stream.color)
    return True


def get_frame():
    """BGR frame for the camera pane + tracked_cam_pos [x_px, y_px, dist_m]."""
    global tracked_cam_pos
    tracked_cam_pos = []
    if not _ensure():
        import numpy as np
        return np.zeros((480, 640, 3), dtype=np.uint8)
    import pyrealsense2 as rs
    frames = _pipe.wait_for_frames(timeout_ms=500)
    frames = _align.process(frames)
    depth = frames.get_depth_frame()
    color = frames.get_color_frame()
    if not color:
        return None
    import numpy as np
    bgr = np.asanyarray(color.get_data())
    if target_uv is None or depth is None:
        return bgr
    u, v = target_uv
    h, w = bgr.shape[0], bgr.shape[1]
    if u < 0 or v < 0 or u >= w or v >= h:
        return bgr
    d = float(depth.get_distance(int(u), int(v)))
    if d > 0:
        tracked_cam_pos[:] = [[float(u), float(v), d]]
    return bgr
