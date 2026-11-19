"""Webcam camera type: a plain USB / built-in camera by its index (OpenCV).

Each index in use gets one reader thread that keeps the newest frame (BGR and
JPEG); `grab(index)` starts the reader if needed and returns the latest JPEG
without waiting, so the camera tick never blocks on the device. Readers whose
index is no longer used by any camera are closed by `keep_only(indexes)`.
"""
from __future__ import annotations

import os
import threading
import time

JPEG_QUALITY = 80
MAX_FPS = 30.0
OPEN_RETRY_S = 2.0       # a camera that would not open is tried again after this
STALE_S = 2.0            # no new frame for this long: reported as not delivering

_readers = {}
_lock = threading.Lock()


class _Reader:
    def __init__(self, index):
        self.index = int(index)
        self.jpeg = None
        self.bgr = None
        self.t = 0.0
        self.error = ""
        self.size = None
        self.stop = False
        self.thread = threading.Thread(target=self._run, name="webcam-%d" % self.index, daemon=True)
        self.thread.start()

    def _open(self, cv2):
        if os.name == "nt":
            # DirectShow opens quickly and gives indexes like the Windows camera list.
            cap = cv2.VideoCapture(self.index, cv2.CAP_DSHOW)
            if cap is not None and cap.isOpened():
                return cap
            if cap is not None:
                cap.release()
        cap = cv2.VideoCapture(self.index)
        if cap is not None and cap.isOpened():
            return cap
        if cap is not None:
            cap.release()
        return None

    def _run(self):
        try:
            import cv2
        except ImportError:
            self.error = "OpenCV (cv2) is not installed"
            return
        cap = None
        try:
            while not self.stop:
                if cap is None:
                    cap = self._open(cv2)
                    if cap is None:
                        self.error = "no webcam at index %d" % self.index
                        self.jpeg = None
                        time.sleep(OPEN_RETRY_S)
                        continue
                    self.error = ""
                t0 = time.perf_counter()
                ok, frame = cap.read()
                if not ok or frame is None:
                    self.error = "webcam %d stopped delivering frames" % self.index
                    cap.release()
                    cap = None
                    time.sleep(0.5)
                    continue
                ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
                if ok:
                    self.bgr = frame
                    self.jpeg = buf.tobytes()  # a new bytes object per frame (the MJPEG stream watches that)
                    self.t = time.time()
                    self.size = (int(frame.shape[1]), int(frame.shape[0]))
                    self.error = ""
                left = 1.0 / MAX_FPS - (time.perf_counter() - t0)
                if left > 0:
                    time.sleep(left)
        finally:
            if cap is not None:
                try:
                    cap.release()
                except Exception:
                    pass


def _reader(index):
    index = int(index)
    with _lock:
        r = _readers.get(index)
        if r is None or not r.thread.is_alive():
            r = _Reader(index)
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
    return None if r is None else r.bgr


def keep_only(indexes):
    """Close the readers whose index no camera uses any more (frees the device)."""
    keep = {int(i) for i in indexes}
    with _lock:
        for idx in [i for i in _readers if i not in keep]:
            _readers.pop(idx).stop = True


def status():
    return {i: {"error": r.error, "size": r.size, "age_s": round(time.time() - r.t, 2) if r.t else None}
            for i, r in list(_readers.items())}
