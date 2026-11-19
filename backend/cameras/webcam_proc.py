"""One webcam, read in its own process (started by webcam.py).

`python webcam_proc.py <index>`: opens the camera (DirectShow first on Windows)
and writes messages to stdout, each a type byte, a 4-byte big-endian length
and the payload: b"F" + JPEG (a frame), b"E" + UTF-8 text (an error / state).
`python webcam_proc.py --list`: prints the DirectShow video device names, one
JSON list on stdout (their order is the DirectShow index order).

In its own process a camera that hangs on open or read (a busy device, a
driver that waits) can't hold up the app; the app just sees no frames.
"""
from __future__ import annotations

import json
import os
import struct
import sys
import time

JPEG_QUALITY = 80
MAX_FPS = 30.0
OPEN_RETRY_S = 2.0


def _out(kind, payload):
    sys.stdout.buffer.write(kind + struct.pack(">I", len(payload)) + payload)
    sys.stdout.buffer.flush()


def _err(text):
    _out(b"E", str(text).encode("utf-8", "replace"))


def list_devices():
    """DirectShow video input names in index order ([] when unknown)."""
    if os.name != "nt":
        return []
    try:
        from pygrabber.dshow_graph import FilterGraph
        return [str(n) for n in (FilterGraph().get_input_devices() or [])]
    except Exception:
        pass
    try:
        import re
        import subprocess
        r = subprocess.run(["ffmpeg", "-hide_banner", "-list_devices", "true", "-f", "dshow", "-i", "dummy"],
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=8)
        text = (r.stdout or b"").decode("utf-8", "replace")
        return re.findall(r'"([^"]+)"\s*\(video\)', text)
    except Exception:
        return []


def _open(cv2, index):
    if os.name == "nt":
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if cap is not None and cap.isOpened():
            return cap
        if cap is not None:
            cap.release()
    cap = cv2.VideoCapture(index)
    if cap is not None and cap.isOpened():
        return cap
    if cap is not None:
        cap.release()
    return None


def main():
    if sys.argv[1:2] == ["--list"]:
        sys.stdout.write(json.dumps(list_devices()))
        sys.stdout.flush()
        return
    index = int(sys.argv[1])
    try:
        import cv2
    except ImportError:
        _err("OpenCV (cv2) is not installed")
        return
    cap = None
    while True:
        if cap is None:
            _err("opening webcam %d" % index)
            cap = _open(cv2, index)
            if cap is None:
                _err("no webcam at index %d" % index)
                time.sleep(OPEN_RETRY_S)
                continue
        t0 = time.perf_counter()
        ok, frame = cap.read()
        if not ok or frame is None:
            _err("webcam %d stopped delivering frames" % index)
            cap.release()
            cap = None
            time.sleep(0.5)
            continue
        ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
        if ok:
            _out(b"F", buf.tobytes())
        left = 1.0 / MAX_FPS - (time.perf_counter() - t0)
        if left > 0:
            time.sleep(left)


if __name__ == "__main__":
    try:
        main()
    except (BrokenPipeError, KeyboardInterrupt):
        pass
    finally:
        os._exit(0)
