"""
RealSense D435i + YOLO person detector for a Camera object (cv script).

Required by the editor:
  tracked_cam_pos  — list of [x_px, y_px, dist_m]
  get_frame()      — HxWx3 BGR uint8

Deps (install on the machine that runs Flask):
  pip install pyrealsense2 ultralytics opencv-python numpy
"""
from __future__ import annotations

tracked_cam_pos = []

_pipe = None
_align = None
_model = None
_started = False


def _start():
    global _pipe, _align, _model, _started
    if _started:
        return
    import pyrealsense2 as rs
    from ultralytics import YOLO

    _pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    _pipe.start(cfg)
    _align = rs.align(rs.stream.color)
    _model = YOLO("yolov8n.pt")
    _started = True


def get_frame():
    """Grab aligned color+depth, run YOLO person class, fill tracked_cam_pos."""
    global tracked_cam_pos
    import numpy as np
    import cv2

    try:
        _start()
    except Exception as e:
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(frame, "RealSense/YOLO unavailable", (20, 240),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80, 80, 80), 1, cv2.LINE_AA)
        cv2.putText(frame, str(e)[:60], (20, 270),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (80, 80, 80), 1, cv2.LINE_AA)
        tracked_cam_pos = []
        return frame

    frames = _pipe.poll_for_frames()
    if not frames:
        frames = _pipe.wait_for_frames(timeout_ms=50)
    frames = _align.process(frames)
    depth = frames.get_depth_frame()
    color = frames.get_color_frame()
    if not depth or not color:
        tracked_cam_pos[:] = []
        return np.zeros((480, 640, 3), dtype=np.uint8)

    bgr = np.asanyarray(color.get_data())
    results = _model.predict(bgr, verbose=False, imgsz=320, conf=0.35, classes=[0])[0]
    hits = []
    if results.boxes is not None:
        for box in results.boxes:
            cls = int(box.cls[0]) if box.cls is not None else -1
            # COCO 0 = person
            if cls != 0:
                continue
            x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]
            cx = 0.5 * (x1 + x2)
            cy = 0.5 * (y1 + y2)
            dist = float(depth.get_distance(int(cx), int(cy)))
            if dist <= 0:
                continue
            hits.append([cx, cy, dist])
            cv2.rectangle(bgr, (int(x1), int(y1)), (int(x2), int(y2)), (0, 220, 255), 2)
            cv2.circle(bgr, (int(cx), int(cy)), 4, (0, 220, 255), -1)
            cv2.putText(bgr, "person %.2fm" % dist, (int(x1), max(16, int(y1) - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 255), 1, cv2.LINE_AA)
    tracked_cam_pos[:] = hits
    return bgr
