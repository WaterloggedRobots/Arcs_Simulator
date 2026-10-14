# OpenCV camera script for Intel RealSense D435i.
# Attach on a camera object with type = opencv / script = this file.
#
# GUI click (camera pane) sets target_uv in color-frame pixels.
# Each frame: tracked_cam_pos = [[u_px, v_px, depth_m], ...]
# which waypoints.py turns into a world-space track sphere.
#
# Standalone:  python d435i_track.py

tracked_cam_pos = []
target_uv = None  # (u, v) in color pixels, or None
last_error = ""

_pipe = None
_align = None
_started = False
_fail_n = 0
_color_rgb = False


def set_target(u, v):
    global target_uv
    try:
        target_uv = (int(round(float(u))), int(round(float(v))))
    except Exception:
        target_uv = None


def _reset():
    global _pipe, _align, _started
    try:
        if _pipe is not None and _started:
            _pipe.stop()
    except Exception:
        pass
    _pipe = None
    _align = None
    _started = False


def _ensure():
    global _pipe, _align, _started, _fail_n, last_error, _color_rgb
    if _started and _pipe is not None:
        return True
    _reset()
    try:
        import pyrealsense2 as rs
    except Exception as e:
        last_error = "pyrealsense2 missing: %s" % e
        if _fail_n == 0:
            print("d435i:", last_error)
        _fail_n += 1
        return False
    try:
        last = None
        used_rgb = False
        for color_fmt in (rs.format.bgr8, rs.format.rgb8):
            pipe = rs.pipeline()
            cfg = rs.config()
            cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
            cfg.enable_stream(rs.stream.color, 640, 480, color_fmt, 30)
            try:
                pipe.start(cfg)
                _pipe = pipe
                last = None
                used_rgb = color_fmt == rs.format.rgb8
                break
            except Exception as e:
                last = e
                # never stop() a pipeline that did not start — that wedges wait_for_frames
                _pipe = None
        if last is not None or _pipe is None:
            raise last if last is not None else RuntimeError("pipeline start failed")
        _align = rs.align(rs.stream.color)
        _color_rgb = used_rgb
        _started = True
        last_error = ""
        _fail_n = 0
        return True
    except Exception as e:
        last_error = "D435i start failed: %s" % e
        if _fail_n < 2:
            print("d435i start failed", e)
        _fail_n += 1
        _reset()
        return False


def get_frame():
    """BGR frame for the camera pane + tracked_cam_pos [x_px, y_px, dist_m]."""
    global tracked_cam_pos, last_error
    tracked_cam_pos = []
    if not _ensure():
        return None
    try:
        frames = None
        try:
            frames = _pipe.poll_for_frames()
        except Exception:
            frames = None
        if not frames:
            frames = _pipe.wait_for_frames(timeout_ms=200)
        if _align is not None:
            frames = _align.process(frames)
    except Exception as e:
        last_error = "D435i frame: %s" % e
        msg = str(e).lower()
        if "before start" in msg or "not start" in msg:
            _reset()
        return None
    depth = frames.get_depth_frame()
    color = frames.get_color_frame()
    if not color:
        last_error = "D435i: no color frame"
        return None
    import numpy as np
    bgr = np.asanyarray(color.get_data())
    if _color_rgb and bgr.ndim == 3 and bgr.shape[2] == 3:
        bgr = bgr[:, :, ::-1].copy()
    last_error = ""
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


if __name__ == "__main__":
    import time

    print("d435i_track standalone — q to quit")
    try:
        import cv2
    except Exception as e:
        cv2 = None
        print("cv2 not available (%s); printing frame info only" % e)

    n = 0
    ok = 0
    t0 = time.time()
    shown = ""
    try:
        while True:
            frame = get_frame()
            n += 1
            dt = max(1e-6, time.time() - t0)
            if frame is None:
                msg = "no frame  %s  tries=%d" % (last_error or "?", n)
            else:
                ok += 1
                h, w = frame.shape[0], frame.shape[1]
                msg = "ok %dx%d  track=%s  frames=%d/%d  %.1f Hz" % (
                    w, h, tracked_cam_pos, ok, n, ok / dt,
                )
                if cv2 is not None:
                    vis = frame
                    if target_uv is not None:
                        cv2.circle(vis, (int(target_uv[0]), int(target_uv[1])), 8, (0, 255, 255), 2)
                    cv2.imshow("d435i_track", vis)
            if msg != shown:
                print(msg)
                shown = msg
            if cv2 is not None:
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q") or key == 27:
                    break
            else:
                time.sleep(0.05)
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        _reset()
        if cv2 is not None:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass
