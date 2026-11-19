"""Webcam camera type: a plain USB / built-in camera by its index (OpenCV).

Each index in use is read by its own process (webcam_proc.py): opening a busy
camera, or one whose driver waits, froze the app when it was read on a thread
here. A thread per process reads its frames (JPEG) from the pipe; `grab(index)`
returns the newest without waiting. `keep_only(indexes)` ends the processes no
camera uses. `devices()` names the cameras (DirectShow order = index), listed
by a process too and cached.
"""
from __future__ import annotations

import json
import os
import struct
import subprocess
import sys
import threading
import time

STALE_S = 2.0            # no new frame for this long: reported as not delivering
LIST_TTL_S = 30.0        # device names are listed again after this
_HERE = os.path.dirname(os.path.abspath(__file__))
_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)

_readers = {}
_lock = threading.Lock()


class _Reader:
    def __init__(self, index):
        self.index = int(index)
        self.jpeg = None
        self.t = 0.0
        self.error = "starting webcam %d" % self.index
        self.proc = None
        self.stop = False
        try:
            self.proc = subprocess.Popen(
                [sys.executable, os.path.join(_HERE, "webcam_proc.py"), str(self.index)],
                stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                creationflags=_NO_WINDOW)
        except Exception as e:
            self.error = "webcam process failed to start: %s" % e
            return
        self.thread = threading.Thread(target=self._read, name="webcam-%d" % self.index, daemon=True)
        self.thread.start()

    def alive(self):
        return self.proc is not None and self.proc.poll() is None

    def _read(self):
        f = self.proc.stdout
        try:
            while not self.stop:
                head = f.read(5)
                if not head or len(head) < 5:
                    break
                n = struct.unpack(">I", head[1:5])[0]
                data = f.read(n)
                if len(data) < n:
                    break
                if head[:1] == b"F":
                    self.jpeg = data  # a new bytes object per frame (the MJPEG stream watches that)
                    self.t = time.time()
                    self.error = ""
                else:
                    self.error = data.decode("utf-8", "replace")
                    if not self.error.startswith("opening"):
                        self.jpeg = None
        except Exception:
            pass
        if not self.stop:
            self.error = "webcam %d process ended" % self.index
            self.jpeg = None

    def close(self):
        self.stop = True
        if self.proc is not None:
            try:
                self.proc.kill()
            except Exception:
                pass


def _reader(index):
    index = int(index)
    with _lock:
        r = _readers.get(index)
        if r is None or (not r.alive() and time.time() - getattr(r, "born", 0) > 3.0):
            if r is not None:
                r.close()
            r = _Reader(index)
            r.born = time.time()
            _readers[index] = r
        return r


def grab(index):
    """(jpeg bytes or None, error text) of the newest frame from webcam `index`."""
    r = _reader(index)
    if r.jpeg is not None and time.time() - r.t > STALE_S:
        return None, r.error or "webcam %d is not delivering frames" % r.index
    return r.jpeg, r.error


def latest_bgr(index):
    r = _readers.get(int(index))
    if r is None or r.jpeg is None:
        return None
    try:
        import cv2
        import numpy as np
        return cv2.imdecode(np.frombuffer(r.jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
    except Exception:
        return None


def keep_only(indexes):
    """End the readers whose index no camera uses any more (frees the device)."""
    keep = {int(i) for i in indexes}
    with _lock:
        for idx in [i for i in _readers if i not in keep]:
            _readers.pop(idx).close()


def close_all():
    keep_only([])


import atexit  # noqa: E402
atexit.register(close_all)  # a camera stuck opening never notices the app is gone


# -- device names -------------------------------------------------------------
_list = {"names": None, "t": 0.0, "busy": False}


def _list_now():
    try:
        r = subprocess.run([sys.executable, os.path.join(_HERE, "webcam_proc.py"), "--list"],
                           stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                           timeout=15, creationflags=_NO_WINDOW)
        names = json.loads((r.stdout or b"[]").decode("utf-8", "replace") or "[]")
        _list["names"] = [str(n) for n in names]
    except Exception:
        _list["names"] = _list["names"] or []
    _list["t"] = time.time()
    _list["busy"] = False


def devices(refresh=False):
    """[{index, name}] of the video devices (empty when they can't be listed);
    listed in the background, so the first call may return [] and list later."""
    stale = _list["names"] is None or refresh or time.time() - _list["t"] > LIST_TTL_S
    if stale and not _list["busy"]:
        _list["busy"] = True
        threading.Thread(target=_list_now, name="webcam-list", daemon=True).start()
    return [{"index": i, "name": n} for i, n in enumerate(_list["names"] or [])]


def device_name(index):
    names = _list["names"] or []
    try:
        return names[int(index)]
    except Exception:
        return ""


DEPTH_CAMERA_WORDS = ("orbbec", "astra", "gemini", "femto", "realsense")


def is_depth_camera(index):
    n = device_name(index).lower()
    return any(w in n for w in DEPTH_CAMERA_WORDS)
