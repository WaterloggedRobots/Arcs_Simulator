# OpenCV camera script for Orbbec depth cameras (Femto, Astra, Gemini).
# Attach on a Camera object: Type = OpenCV, Script = this file.
#
# In the camera pane, drag a rectangle around the target. The script tracks it
# frame to frame (OpenCV CSRT, else KCF / MIL) and publishes its 3D position
# as the sandbox's primary target (the big red sphere), which Look At can
# follow ("select primary target"). A plain click starts tracking a small box
# around the point clicked.
#
# Position: the median depth inside the middle of the box, turned into metres
# in the camera frame (x right, y down, z forward) with the colour camera's own
# intrinsics, so the Camera object's hfov / vfov don't matter. Without
# intrinsics it falls back to [x_px, y_px, depth_m] (then set the Camera
# object's hfov / vfov to the camera's). The Camera object must sit where the
# Orbbec's colour lens is: the target is placed from its pose.
#
# Needs: pip install pyorbbecsdk2 opencv-contrib-python
# (Orbbec SDK v2 on PyPI is "pyorbbecsdk2", imported as pyorbbecsdk; the old
#  "pyorbbecsdk" 1.x on PyPI has no wheels past Python 3.11.)
# (opencv-python works too, with the weaker MIL tracker only.)
# Without pyorbbecsdk it opens the camera's colour stream as a plain webcam
# (WEBCAM_INDEX): the box is tracked and drawn, but there is no depth, so no
# 3D target unless FALLBACK_DEPTH_M is set.
#
# Standalone (no sandbox):  python orbbec_box_track.py   (drag a box, q quits)

import time

import numpy as np

# --- settings ---------------------------------------------------------------
COLOR_SIZE = (640, 480)      # wanted colour resolution (the nearest the camera has)
FPS = 30
CLICK_BOX_PX = 60            # a click tracks a square this size around it
MIN_BOX_PX = 8               # smaller drags count as a click
DEPTH_CORE = 0.4             # depth from the middle 40 % of the box
DEPTH_RANGE_M = (0.15, 8.0)  # depth readings outside this are ignored
LOST_AFTER_S = 1.0           # tracker failing this long: "lost" (keeps trying)
WEBCAM_INDEX = 0             # fallback colour source without pyorbbecsdk
FALLBACK_DEPTH_M = None      # e.g. 1.0: a fixed distance when there is no depth

# --- read by the sandbox ----------------------------------------------------
primary_target = None        # [x, y, z] (see target_space)
targets = []                 # unused: only the primary target
target_space = "pixel"       # "camera" once the intrinsics are known
last_error = ""

# --- state ------------------------------------------------------------------
_pipe = None
_align = None
_cap = None
_mode = None                 # "orbbec" | "webcam"
_intr = None                 # (fx, fy, cx, cy) of the colour image
_fail_n = 0
_pending = None              # (x0, y0, x1, y1) waiting for the next frame
_tracker = None
_tracker_name = ""
_box = None                  # (x, y, w, h) current box
_ok_at = 0.0
_status = "drag a box around the target"


# ---------------------------------------------------------------------------
# called by the sandbox
# ---------------------------------------------------------------------------
def set_box(x0, y0, x1, y1):
    """A rectangle dragged in the camera pane (image pixels)."""
    global _pending
    try:
        x0, y0, x1, y1 = (float(v) for v in (x0, y0, x1, y1))
    except Exception:
        return
    _pending = (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))


def set_target(u, v):
    """A click in the camera pane: track a small box around it."""
    h = CLICK_BOX_PX / 2.0
    set_box(float(u) - h, float(v) - h, float(u) + h, float(v) + h)


def clear_target():
    global _tracker, _box, _pending, primary_target, _status
    _tracker, _box, _pending, primary_target = None, None, None, None
    _status = "drag a box around the target"


def get_frame():
    """BGR frame for the camera pane; updates primary_target."""
    global last_error
    grabbed = _grab()
    if grabbed is None:
        return None
    bgr, depth_m = grabbed
    last_error = ""
    _track(bgr, depth_m)
    return _draw(bgr)


# ---------------------------------------------------------------------------
# tracking
# ---------------------------------------------------------------------------
def _new_tracker():
    import cv2
    for name in ("TrackerCSRT_create", "TrackerKCF_create", "TrackerMIL_create"):
        for mod in (cv2, getattr(cv2, "legacy", None)):
            fn = getattr(mod, name, None) if mod is not None else None
            if callable(fn):
                try:
                    return fn(), name[len("Tracker"):-len("_create")]
                except Exception:
                    continue
    raise RuntimeError("no OpenCV tracker available (pip install opencv-contrib-python)")


def _track(bgr, depth_m):
    global _pending, _tracker, _tracker_name, _box, _ok_at, primary_target, _status
    h, w = bgr.shape[:2]
    if _pending is not None:
        x0, y0, x1, y1 = _pending
        _pending = None
        x0, y0 = max(0.0, x0), max(0.0, y0)
        x1, y1 = min(w - 1.0, x1), min(h - 1.0, y1)
        if x1 - x0 < MIN_BOX_PX or y1 - y0 < MIN_BOX_PX:
            cx, cy, half = (x0 + x1) / 2, (y0 + y1) / 2, CLICK_BOX_PX / 2
            x0, y0 = max(0.0, cx - half), max(0.0, cy - half)
            x1, y1 = min(w - 1.0, cx + half), min(h - 1.0, cy + half)
        box = (int(x0), int(y0), max(2, int(x1 - x0)), max(2, int(y1 - y0)))
        try:
            _tracker, _tracker_name = _new_tracker()
            _tracker.init(bgr, box)
            _box, _ok_at = box, time.time()
            _status = "tracking (%s)" % _tracker_name
        except Exception as e:
            _tracker, _box = None, None
            _status = "tracker failed: %s" % e
    if _tracker is None:
        primary_target = None
        return
    try:
        ok, box = _tracker.update(bgr)
    except Exception:
        ok, box = False, None
    now = time.time()
    if ok and box is not None and box[2] > 1 and box[3] > 1:
        _box, _ok_at = tuple(int(round(v)) for v in box), now
        _status = "tracking (%s)" % _tracker_name
    else:
        if now - _ok_at > LOST_AFTER_S:
            _status = "lost: drag a new box"
        primary_target = None  # the sandbox keeps aiming at the last place seen
        return
    x, y, bw, bh = _box
    u, v = x + bw / 2.0, y + bh / 2.0
    z = _box_depth(depth_m, _box, (w, h))
    if z is None:
        z = FALLBACK_DEPTH_M
    if z is None:
        primary_target = None
        _status = "tracking (%s), no depth" % _tracker_name
        return
    primary_target = _to_target(u, v, z)


def _box_depth(depth_m, box, size):
    """Median valid depth (m) in the middle of the box, or None."""
    if depth_m is None:
        return None
    x, y, bw, bh = box
    w, h = size
    dh, dw = depth_m.shape[:2]
    sx, sy = dw / float(w), dh / float(h)  # depth may be another resolution
    m = (1.0 - DEPTH_CORE) / 2.0
    xa, xb = int((x + bw * m) * sx), int((x + bw * (1 - m)) * sx) + 1
    ya, yb = int((y + bh * m) * sy), int((y + bh * (1 - m)) * sy) + 1
    patch = depth_m[max(0, ya):min(dh, yb), max(0, xa):min(dw, xb)]
    vals = patch[(patch > DEPTH_RANGE_M[0]) & (patch < DEPTH_RANGE_M[1])]
    if vals.size < 5:
        return None
    return float(np.median(vals))


def _to_target(u, v, z):
    """[x, y, z] for the sandbox: metres in the camera frame when the colour
    intrinsics are known, else [x_px, y_px, depth_m]."""
    global target_space
    if _intr is not None:
        fx, fy, cx, cy = _intr
        target_space = "camera"
        return [(u - cx) * z / fx, (v - cy) * z / fy, z]
    target_space = "pixel"
    return [u, v, z]


def _draw(bgr):
    import cv2
    vis = bgr.copy() if not bgr.flags.writeable else bgr
    lost = _status.startswith("lost")
    if _box is not None:
        x, y, bw, bh = _box
        col = (0, 0, 255) if lost else (0, 220, 0)
        cv2.rectangle(vis, (x, y), (x + bw, y + bh), col, 2)
        cv2.drawMarker(vis, (int(x + bw / 2), int(y + bh / 2)), col, cv2.MARKER_CROSS, 14, 2)
    txt = _status
    if primary_target is not None:
        txt += "  z %.2f m" % float(primary_target[2])
    # Centred at the top: the camera pane crops the image's sides to fill the pane.
    (tw, _th), _b = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
    org = (max(4, (vis.shape[1] - tw) // 2), 22)
    cv2.putText(vis, txt, org, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(vis, txt, org, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return vis


# ---------------------------------------------------------------------------
# camera
# ---------------------------------------------------------------------------
def _close():
    global _pipe, _align, _cap, _mode
    try:
        if _pipe is not None:
            _pipe.stop()
    except Exception:
        pass
    try:
        if _cap is not None:
            _cap.release()
    except Exception:
        pass
    _pipe, _align, _cap, _mode = None, None, None, None


def _pick_profile(plist):
    """The colour / depth profile nearest COLOR_SIZE at FPS, else the default."""
    try:
        return plist.get_video_stream_profile(COLOR_SIZE[0], 0, _any_format(), FPS)
    except Exception:
        pass
    return plist.get_default_video_stream_profile()


def _any_format():
    from pyorbbecsdk import OBFormat
    return getattr(OBFormat, "UNKNOWN_FORMAT", getattr(OBFormat, "ANY", 0))


def _intrinsics(pipe, color_profile):
    """(fx, fy, cx, cy) of the colour stream, or None (SDK v2, then v1)."""
    try:
        i = color_profile.get_intrinsic()
        return float(i.fx), float(i.fy), float(i.cx), float(i.cy)
    except Exception:
        pass
    try:
        i = pipe.get_camera_param().rgb_intrinsic
        return float(i.fx), float(i.fy), float(i.cx), float(i.cy)
    except Exception:
        return None


def _open_orbbec():
    global _pipe, _align, _mode, _intr
    from pyorbbecsdk import Config, OBSensorType, Pipeline
    pipe = Pipeline()
    cfg = Config()
    cprof = _pick_profile(pipe.get_stream_profile_list(OBSensorType.COLOR_SENSOR))
    cfg.enable_stream(cprof)
    has_depth = False
    try:
        cfg.enable_stream(_pick_profile(pipe.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)))
        has_depth = True
    except Exception as e:
        print("orbbec: no depth stream:", e)
    align = None
    if has_depth:
        try:  # SDK v2: depth aligned to colour by a filter
            from pyorbbecsdk import AlignFilter, OBStreamType
            align = AlignFilter(align_to_stream=OBStreamType.COLOR_STREAM)
        except Exception:
            try:  # SDK v1: aligned in the pipeline
                from pyorbbecsdk import OBAlignMode
                cfg.set_align_mode(OBAlignMode.SW_MODE)
            except Exception:
                pass
    pipe.start(cfg)
    _pipe, _align, _mode = pipe, align, "orbbec"
    _intr = _intrinsics(pipe, cprof)
    print("orbbec: started, intrinsics", _intr)


def _open_webcam():
    global _cap, _mode, _intr
    import cv2
    cap = cv2.VideoCapture(WEBCAM_INDEX)
    if not cap.isOpened():
        raise RuntimeError("no camera at index %d" % WEBCAM_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, COLOR_SIZE[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, COLOR_SIZE[1])
    _cap, _mode, _intr = cap, "webcam", None


def _ensure():
    global last_error, _fail_n
    if _mode is not None:
        return True
    try:
        try:
            import pyorbbecsdk  # noqa: F401
        except ImportError:
            _open_webcam()
            last_error = ""
            print("orbbec: pyorbbecsdk missing (pip install pyorbbecsdk2), colour only (webcam %d)" % WEBCAM_INDEX)
        else:
            _open_orbbec()
        _fail_n = 0
        return True
    except Exception as e:
        _close()
        last_error = "Orbbec start failed: %s" % e
        if _fail_n < 2:
            print(last_error)
        _fail_n += 1
        return False


def _color_to_bgr(frame):
    import cv2
    from pyorbbecsdk import OBFormat
    w, h = frame.get_width(), frame.get_height()
    fmt = frame.get_format()
    data = np.asanyarray(frame.get_data())
    if fmt == OBFormat.RGB:
        return cv2.cvtColor(data.reshape(h, w, 3), cv2.COLOR_RGB2BGR)
    if fmt == OBFormat.BGR:
        return data.reshape(h, w, 3).copy()
    if fmt == OBFormat.MJPG:
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    if fmt == OBFormat.YUYV:
        return cv2.cvtColor(data.reshape(h, w, 2), cv2.COLOR_YUV2BGR_YUY2)
    if fmt == OBFormat.UYVY:
        return cv2.cvtColor(data.reshape(h, w, 2), cv2.COLOR_YUV2BGR_UYVY)
    if fmt == OBFormat.NV12:
        return cv2.cvtColor(data.reshape(h * 3 // 2, w), cv2.COLOR_YUV2BGR_NV12)
    if fmt == OBFormat.NV21:
        return cv2.cvtColor(data.reshape(h * 3 // 2, w), cv2.COLOR_YUV2BGR_NV21)
    raise RuntimeError("colour format %s not handled" % fmt)


def _grab():
    """(bgr, depth in metres or None), or None without a frame."""
    global last_error
    if not _ensure():
        return None
    if _mode == "webcam":
        ok, bgr = _cap.read()
        if not ok or bgr is None:
            last_error = "webcam: no frame"
            return None
        return bgr, None
    try:
        frames = _pipe.wait_for_frames(100)
        if frames is None:
            return None
        if _align is not None:
            frames = _align.process(frames)
            frames = frames.as_frame_set() if hasattr(frames, "as_frame_set") else frames
        color = frames.get_color_frame()
        if color is None:
            return None
        bgr = _color_to_bgr(color)
        depth_m = None
        depth = frames.get_depth_frame()
        if depth is not None:
            dw, dh = depth.get_width(), depth.get_height()
            raw = np.frombuffer(depth.get_data(), dtype=np.uint16).reshape(dh, dw)
            depth_m = raw.astype(np.float32) * float(depth.get_depth_scale()) / 1000.0
        return bgr, depth_m
    except Exception as e:
        last_error = "Orbbec frame: %s" % e
        return None


# ---------------------------------------------------------------------------
# standalone test: drag a box with the mouse
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import cv2

    drag = {}

    def _mouse(ev, x, y, flags, _):
        if ev == cv2.EVENT_LBUTTONDOWN:
            drag["a"] = (x, y)
        elif ev == cv2.EVENT_LBUTTONUP and "a" in drag:
            ax, ay = drag.pop("a")
            set_box(ax, ay, x, y)

    cv2.namedWindow("orbbec_box_track")
    cv2.setMouseCallback("orbbec_box_track", _mouse)
    shown = ""
    try:
        while True:
            img = get_frame()
            msg = "%s  target=%s (%s)" % (_status, primary_target, target_space) if img is not None else last_error
            if msg != shown:
                print(msg)
                shown = msg
            if img is not None:
                cv2.imshow("orbbec_box_track", img)
            if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
                break
    finally:
        _close()
        cv2.destroyAllWindows()
