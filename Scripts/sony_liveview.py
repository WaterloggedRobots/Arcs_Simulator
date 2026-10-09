#!/usr/bin/env python3
"""Sony USB live-view + in-camera movie rec (PTP ControlDevice DPC_MOVIE_REC).

Grab order: OpenCV UVC (USB Streaming) → PTP GetObject liveview → control.exe
Movie rec: release UVC, SDIOControlDevice(0xD2C8, DOWN=2 / UP=1).
Needs Remote Shooting / PTP, not USB Streaming, for on-card REC.
"""
from __future__ import print_function

import argparse
import os
import struct
import sys
import threading
import time
from pathlib import Path

SONY_VID = 0x054C
PTP_CLASS = 6
GET_DEVICE_INFO = 0x1001
GET_OBJECT = 0x1009
OPEN_SESSION = 0x1002
CLOSE_SESSION = 0x1003
LIVEVIEW_HANDLE = 0xFFFFC002
AUTH_A = 0x96FE
AUTH_B = 0x96FD
SET_EXT = 0x9205
SDIO_CONNECT = 0x9201
SDIO_GETEXT = 0x9202
SDIO_CONTROL = 0x9207
DPC_MOVIE_REC = 0xD2C8
DPC_EXPOSURE_MODE = 0x500E
DPC_OPERATING_MODE = 0x5013
DPC_POSITION_KEY = 0xD25A
DPC_MOVIE_TOGGLE = 0xD2EC
MOVIE_DOWN = 0x0002
MOVIE_UP = 0x0001
EXPOSURE_MOVIE = 0x000A8088
SDI_V3 = 0x012C
SDI_V2 = 0x00C8
OP_NOT_SUPPORTED = 0x2005
PTP_OK = 0x2001

_usb_lock = threading.Lock()
_session = None
_cv_cap = None
_cv_cap_index = None
_prefer_ptp = False
_ptp_denied_until = 0.0
_last_err = ""
_backend = None


def _roots():
    here = Path(__file__).resolve().parent
    return [here, here.parent]


def extract_jpeg(blob):
    if not blob:
        return None
    if blob[:2] == b"\xff\xd8":
        return blob
    soi = blob.find(b"\xff\xd8")
    if soi >= 0:
        eoi = blob.rfind(b"\xff\xd9")
        if eoi > soi:
            return blob[soi:eoi + 2]
        return blob[soi:]
    if len(blob) >= 8:
        offset, size = struct.unpack_from("<II", blob, 0)
        if 8 <= offset < len(blob) and size > 0:
            sl = blob[offset:offset + size]
            if sl[:2] == b"\xff\xd8":
                return sl
    return None


class PtpSession(object):
    def __init__(self, dev, ep_out, ep_in):
        self.dev = dev
        self.ep_out = ep_out
        self.ep_in = ep_in
        self.tid = 0
        self.open = False
        self.sdio_ok = False

    def _cmd(self, code, params=(), data_out=None, expect_data=True, timeout=3000):
        params = list(params)[:5]
        n = len(params)
        length = 12 + 4 * n
        buf = struct.pack("<IHH I", length, 1, code, self.tid)
        for p in params:
            buf += struct.pack("<I", int(p) & 0xFFFFFFFF)
        self.tid += 1
        try:
            self.dev.write(self.ep_out, buf, timeout=timeout)
        except Exception as e:
            self._clear_pipes()
            raise RuntimeError("PTP write cmd=0x%04X ep=0x%02X: %s" % (code, self.ep_out, e))
        payload = b""
        if data_out:
            dlen = 12 + len(data_out)
            hdr = struct.pack("<IHH I", dlen, 2, code, self.tid - 1)
            self.dev.write(self.ep_out, hdr + data_out, timeout=timeout)
        resp_code = 0
        while True:
            chunk = bytes(self.dev.read(self.ep_in, 1024 * 256, timeout=timeout))
            if len(chunk) < 12:
                break
            clen, ctype, ccode, _ctid = struct.unpack_from("<IHH I", chunk, 0)
            if ctype == 2:
                payload += chunk[12:clen]
                if clen > len(chunk):
                    remain = clen - len(chunk)
                    while remain > 0:
                        more = bytes(self.dev.read(self.ep_in, min(remain, 1024 * 256), timeout=timeout))
                        payload += more
                        remain -= len(more)
            elif ctype == 3:
                resp_code = ccode
                break
            if not expect_data and ctype == 3:
                break
        return resp_code, payload

    def _clear_pipes(self):
        for ep in (self.ep_out, self.ep_in, 0x02, 0x81, 0x83):
            try:
                self.dev.clear_halt(ep)
            except Exception:
                pass

    def _reclaim(self, hard=False):
        import usb.util
        if hard:
            try:
                self.dev.reset()
                time.sleep(0.5)
            except Exception as e:
                print("sony reset", e)
            try:
                self.dev.set_configuration()
            except Exception as e:
                print("sony set_configuration", e)
        try:
            usb.util.claim_interface(self.dev, 0)
        except Exception as e:
            print("sony re-claim", e)
        self._clear_pipes()

    def _try_open(self, timeout=4000):
        self.tid = 0
        return self._cmd(OPEN_SESSION, [1], expect_data=False, timeout=timeout)

    def start(self):
        pairs = list(getattr(self, "_pipe_pairs", None) or [(self.ep_out, self.ep_in)])
        last = None
        # Do not reset first. WinUSB 0E74 often drops config on reset and
        # every later OUT 0x02 10060s. First OpenSession may 10060; then reset.
        self._reclaim(hard=False)
        for ep_out, ep_in in pairs:
            self.ep_out, self.ep_in = ep_out, ep_in
            try:
                try:
                    code, payload = self._cmd(GET_DEVICE_INFO, expect_data=True, timeout=1500)
                    print("sony GetDeviceInfo", hex(code or 0), "bytes", len(payload or b""))
                except Exception as e:
                    print("sony GetDeviceInfo", e)
                self._try_open()
                self.open = True
                print("sony OpenSession ok", hex(ep_out), hex(ep_in))
                return
            except Exception as e:
                last = e
                print("sony OpenSession fail", hex(ep_out), hex(ep_in), e)
                self._clear_pipes()
        print("sony OpenSession after reset")
        self._reclaim(hard=True)
        for ep_out, ep_in in pairs:
            self.ep_out, self.ep_in = ep_out, ep_in
            try:
                self._try_open(timeout=5000)
                self.open = True
                print("sony OpenSession ok", hex(ep_out), hex(ep_in))
                return
            except Exception as e:
                last = e
                print("sony OpenSession fail", hex(ep_out), hex(ep_in), e)
                self._clear_pipes()
        if last is not None:
            raise last
        self.open = True

    def _sdio_handshake(self):
        if self.sdio_ok:
            return True
        try:
            self._cmd(SDIO_CONNECT, [1, 0, 0], expect_data=True, timeout=800)
            self._cmd(SDIO_CONNECT, [2, 0, 0], expect_data=True, timeout=800)
            ver = None
            for want in (SDI_V3, SDI_V2):
                try:
                    code, payload = self._cmd(SDIO_GETEXT, [want], expect_data=True, timeout=800)
                    if payload and len(payload) >= 2:
                        ver = struct.unpack_from("<H", payload, 0)[0]
                    print("sony SDIO GetExt", hex(code or 0), "ver", hex(ver or 0), "bytes", len(payload or b""))
                    if ver in (SDI_V3, SDI_V2) or code in (PTP_OK, 0):
                        break
                except Exception as e:
                    print("sony SDIO GetExt", hex(want), e)
            self._cmd(SDIO_CONNECT, [3, 0, 0], expect_data=True, timeout=800)
            self.sdio_ok = True
            print("sony SDIO handshake ok", hex(ver or 0))
            return True
        except Exception as e:
            print("sony SDIO handshake", e)
            self.sdio_ok = False
            self._clear_pipes()
            return False

    def get_liveview(self):
        code, payload = self._cmd(GET_OBJECT, [LIVEVIEW_HANDLE], expect_data=True, timeout=5000)
        n = getattr(self, "_lv_n", 0) + 1
        self._lv_n = n
        if n == 1:
            print("sony liveview get", hex(code or 0), "bytes", len(payload or b""))
        if code not in (0x2001, 0) and not payload:
            raise RuntimeError("PTP GetObject liveview resp=0x%04X" % code)
        jpeg = extract_jpeg(payload)
        if not jpeg:
            raise RuntimeError("liveview payload is not JPEG (%d bytes)" % len(payload))
        return jpeg

    def set_movie(self, on):
        # 0x9207 is OperationNotSupported until SDIO 9201/9202 handshake.
        self._sdio_handshake()
        val = MOVIE_DOWN if on else MOVIE_UP
        u16 = struct.pack("<H", val)
        codes = []
        attempts = (
            (SDIO_CONTROL, [DPC_MOVIE_REC], u16),
            (SDIO_CONTROL, [DPC_MOVIE_TOGGLE], struct.pack("<H", MOVIE_DOWN)),
        )
        for opcode, params, payload in attempts:
            if not on and opcode == SDIO_CONTROL and params[0] == DPC_MOVIE_TOGGLE:
                continue
            code, _ = self._cmd(opcode, params, data_out=payload, expect_data=False, timeout=1500)
            codes.append(code)
            if code not in (PTP_OK, 0):
                print("sony movie cmd", hex(opcode), hex(params[0]), "resp", hex(code or 0))
            if code in (PTP_OK, 0):
                return code
        raise RuntimeError(
            "movie rec PTP resp="
            + ",".join("0x%04X" % c for c in codes)
            + " (0x2005=op not supported until SDIO handshake)"
        )

    def close(self):
        try:
            self._cmd(CLOSE_SESSION, expect_data=False)
        except Exception:
            pass
        self.open = False


def _libusb_candidates():
    # libusbK.dll is the intended Windows backend for WinUSB'd 054C:0E74.
    names = (
        "libusbK.dll", "libusbk.dll",
        "libusb-1.0.dll", "libusb-1.0.so", "libusb-1.0.so.0",
        "libusb-0.1.dll", "libusb0.dll",
    )
    extra = []
    env = os.environ.get("LIBUSB_PATH") or os.environ.get("LIBUSB_DIR")
    if env:
        extra.append(Path(env))
    dirs = extra + _roots() + [p / "libusb" for p in _roots()]
    out = []
    for d in dirs:
        if d.is_file():
            out.append(d)
            continue
        if not d.is_dir():
            continue
        for n in names:
            p = d / n
            if p.is_file():
                out.append(p)
    return out


def _usb_backend():
    global _backend
    if _backend is not None:
        return _backend
    try:
        from usb.backend import libusb1, libusb0
    except Exception as e:
        raise RuntimeError("pyusb not installed (%s)" % e)
    found = _libusb_candidates()
    for dll in found:
        os.environ["PATH"] = str(dll.parent) + os.pathsep + os.environ.get("PATH", "")
        try:
            be = libusb1.get_backend(find_library=lambda _: str(dll))
            if be is not None:
                _backend = be
                print("sony usb backend libusb1", dll)
                return be
        except Exception:
            pass
        try:
            be = libusb0.get_backend(find_library=lambda _: str(dll))
            if be is not None:
                _backend = be
                print("sony usb backend libusb0/K", dll)
                return be
        except Exception:
            pass
    for mod in (libusb1, libusb0):
        try:
            be = mod.get_backend()
            if be is not None:
                _backend = be
                return be
        except Exception:
            pass
    hint = "Place libusb-1.0.dll next to webui.py."
    if found:
        hint = "Found %s but pyusb could not load it." % found[0]
    raise RuntimeError("No USB backend (libusb). " + hint)


def _find_sony_device():
    import usb.core
    import usb.util
    backend = _usb_backend()
    devs = list(usb.core.find(find_all=True, idVendor=SONY_VID, backend=backend) or [])
    if not devs:
        for d in usb.core.find(find_all=True, backend=backend) or []:
            try:
                cfg = d.get_active_configuration()
            except Exception:
                continue
            for intf in cfg:
                if int(getattr(intf, "bInterfaceClass", 0)) == PTP_CLASS:
                    devs.append(d)
                    break
    if not devs:
        raise RuntimeError("no Sony / PTP camera on USB")
    # Remote/PTP on ZV-E1 is PID 0x0E74. Never claim 0x0EA5 (UVC Streaming).
    def _rank(d):
        pid = int(getattr(d, "idProduct", 0) or 0)
        if pid == 0x0E74:
            return 0
        if pid == 0x0EA5:
            return 99
        return 1
    keep = []
    for d in devs:
        pid = int(getattr(d, "idProduct", 0) or 0)
        if pid == 0x0EA5:
            print("sony skip UVC 054C:0EA5")
            continue
        keep.append(d)
    if keep:
        devs = keep
    devs = sorted(devs, key=_rank)
    seen = ["%04X:%04X" % (int(d.idVendor), int(d.idProduct)) for d in devs]
    last = None
    for dev in devs:
        try:
            cfg = None
            try:
                if dev.is_kernel_driver_active(0):
                    dev.detach_kernel_driver(0)
            except Exception:
                pass
            cfg = None
            try:
                cfg = dev.get_active_configuration()
            except Exception:
                cfg = None
            if cfg is None:
                # WinUSB: set_configuration often Errno 13; use first config.
                try:
                    cfg = dev[0]
                except Exception:
                    cfg = None
                if cfg is None:
                    try:
                        dev.set_configuration()
                        cfg = dev.get_active_configuration()
                    except Exception as e:
                        last = e
                        continue
            if cfg is None:
                try:
                    cfg = dev[0]
                except Exception as e:
                    last = e
                    continue
            pairs = []
            intf = None
            for cand in cfg:
                cls = int(getattr(cand, "bInterfaceClass", 0) or 0)
                eps = list(cand)
                outs = [e.bEndpointAddress for e in eps if usb.util.endpoint_direction(e.bEndpointAddress) == usb.util.ENDPOINT_OUT]
                ins = [e.bEndpointAddress for e in eps if usb.util.endpoint_direction(e.bEndpointAddress) == usb.util.ENDPOINT_IN]
                if outs and ins:
                    intf = cand
                    for o in outs:
                        for i in ins:
                            pairs.append((o, i, cls))
            if intf is None or not pairs:
                last = RuntimeError("no bulk endpoints on %04X" % int(dev.idProduct))
                continue
            try:
                usb.util.claim_interface(dev, intf.bInterfaceNumber)
            except Exception as e:
                last = e
                continue
            print("sony PTP claimed", "%04X:%04X" % (int(dev.idVendor), int(dev.idProduct)),
                  "if", intf.bInterfaceNumber, "pipes", [(hex(a), hex(b)) for a, b, _ in pairs])
            globals()["_prefer_ptp"] = True
            globals()["_ptp_denied_until"] = 0.0
            sess = PtpSession(dev, pairs[0][0], pairs[0][1])
            sess._pipe_pairs = [(a, b) for a, b, _ in pairs]
            return sess
        except Exception as e:
            last = e
    raise RuntimeError(
        "Access denied on Sony PTP. WinUSB the WPD device VID_054C PID_0E74 "
        "(not the Camera/0EA5 Streaming interface). Devices seen: %s. (%s)"
        % (", ".join(seen) or "none", last)
    )


def close_session():
    """Drop PTP claim so UVC / a new PTP session can reopen after a run."""
    global _session
    with _usb_lock:
        if _session is not None:
            try:
                _session.close()
            except Exception:
                pass
            try:
                import usb.util
                usb.util.dispose_resources(_session.dev)
            except Exception:
                pass
        _session = None
    release_opencv()
    globals()["_prefer_ptp"] = True
    time.sleep(0.4)


def _ensure_usb():
    global _session, _last_err
    with _usb_lock:
        if _session is not None and _session.open:
            return _session
        if _session is not None:
            try:
                import usb.util
                usb.util.dispose_resources(_session.dev)
            except Exception:
                pass
            _session = None
        sess = _find_sony_device()
        try:
            sess.start()
        except Exception as e:
            print("sony PTP session start", e)
            try:
                import usb.util
                usb.util.dispose_resources(sess.dev)
            except Exception:
                pass
            raise
        _session = sess
        _last_err = ""
        return sess


def grab_via_usb():
    global _last_err
    sess = _ensure_usb()
    try:
        return sess.get_liveview()
    except Exception as e:
        _last_err = str(e)
        raise


_SONY_NAME = (
    "sony", "ilce", "ilme", "ilca", "dsc-", "zv-e", "zv-e1", "zve1",
    "zv-1", "zv-f", "fx3", "fx30", "fx6", "alpha",
)
_SKIP_NAME = (
    "integrated", "facetime", "laptop", "built-in", "builtin", "chicony",
    "life cam", "lifecam", "hd webcam", "usb 2.0 camera", "microsof",
    "obs virtual", "droidcam", "iriun",
)


def _forced_cv_index():
    raw = os.environ.get("SONY_OPENCV_INDEX")
    if raw not in (None, ""):
        try:
            return int(raw)
        except Exception:
            pass
    try:
        import json
        data = json.loads((_roots()[0] / "settings.json").read_text(encoding="utf-8"))
        if data.get("sony_opencv_index") is not None:
            return int(data["sony_opencv_index"])
    except Exception:
        pass
    return None


def _list_dshow_names():
    names = []
    try:
        from pygrabber.dshow_graph import FilterGraph
        names = list(FilterGraph().get_input_devices() or [])
        if names:
            return names
    except Exception:
        pass
    try:
        import subprocess
        import re
        r = subprocess.run(
            ["ffmpeg", "-hide_banner", "-list_devices", "true", "-f", "dshow", "-i", "dummy"],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=8,
        )
        text = (r.stdout or b"").decode("utf-8", "replace")
        found = re.findall(r'\"([^\"]+)\"\s*\(video\)', text)
        if not found:
            found = re.findall(r'\]\s*\"([^\"]+)\"', text)
        if found:
            return found
    except Exception:
        pass
    try:
        import subprocess
        ps = (
            "Get-CimInstance Win32_PnPEntity | "
            "Where-Object { $_.PNPClass -eq 'Camera' -or $_.PNPClass -eq 'Image' "
            "-or $_.Name -match 'Camera|Imaging|Sony|ZV-E|ILME' } | "
            "Select-Object -ExpandProperty Name"
        )
        r = subprocess.run(
            ["powershell", "-NoProfile", "-Command", ps],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, timeout=8,
        )
        extra = [ln.strip() for ln in (r.stdout or b"").decode("utf-8", "replace").splitlines() if ln.strip()]
        if extra:
            return extra
    except Exception:
        pass
    return names


def _is_sony_cam_name(name):
    n = (name or "").lower()
    if not n:
        return False
    if any(s in n for s in _SKIP_NAME):
        return False
    return any(s in n for s in _SONY_NAME)


def release_opencv():
    global _cv_cap, _cv_cap_index
    if _cv_cap is not None:
        try:
            _cv_cap.release()
        except Exception:
            pass
    _cv_cap = None
    _cv_cap_index = None


def grab_via_opencv(index=None):
    global _cv_cap, _cv_cap_index, _last_err
    try:
        import cv2
    except Exception as e:
        raise RuntimeError("opencv not available (%s)" % e)
    forced = index if index is not None else _forced_cv_index()
    if forced is not None:
        try:
            forced = int(forced)
        except Exception:
            forced = None
    if _cv_cap is not None and _cv_cap.isOpened():
        if forced is None or _cv_cap_index == forced:
            ok, frame = _cv_cap.read()
            if ok and frame is not None:
                ok2, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
                if ok2:
                    _last_err = ""
                    return bytes(buf)
        release_opencv()
    names = _list_dshow_names()
    indices = []
    if forced is not None:
        if forced == 0 and names:
            n0 = (names[0] or "").lower()
            if any(s in n0 for s in _SKIP_NAME) or not _is_sony_cam_name(names[0]):
                raise RuntimeError("index 0 is %s, not Sony — using PTP" % (names[0],))
        indices = [forced]
    else:
        for i, name in enumerate(names):
            if _is_sony_cam_name(name):
                indices.append(i)
        if not indices:
            raise RuntimeError(
                "no Sony-named camera; set the index box. Devices: %s"
                % (names or "(unlisted)")
            )
    apis = []
    if hasattr(cv2, "CAP_DSHOW"):
        apis.append(cv2.CAP_DSHOW)
    if hasattr(cv2, "CAP_MSMF"):
        apis.append(cv2.CAP_MSMF)
    apis.append(None)
    for idx in indices:
        for api in apis:
            try:
                cap = cv2.VideoCapture(idx) if api is None else cv2.VideoCapture(idx, api)
            except Exception:
                cap = None
            if cap is not None and cap.isOpened():
                ok, frame = cap.read()
                if ok and frame is not None:
                    _cv_cap = cap
                    _cv_cap_index = idx
                    ok2, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
                    if ok2:
                        _last_err = ""
                        return bytes(buf)
                cap.release()
    raise RuntimeError("OpenCV device failed to open (indices %s)" % indices)


def grab_via_control(out_path=None):
    ctrl = None
    for base in _roots():
        for name in ("control.exe", "control", "sony_liveview.exe"):
            p = base / name
            if p.is_file():
                ctrl = p
                break
        if ctrl is not None:
            break
    env_ctrl = os.environ.get("SONY_PTP_CONTROL")
    if env_ctrl and Path(env_ctrl).exists():
        ctrl = Path(env_ctrl)
    if ctrl is None:
        raise RuntimeError("no USB camera claimed and no `control` binary")
    import subprocess
    import tempfile
    dest = out_path
    tmp = None
    if not dest:
        fd, tmp = tempfile.mkstemp(suffix=".jpg")
        os.close(fd)
        dest = tmp
    try:
        env = os.environ.copy()
        env["LD_LIBRARY_PATH"] = str(Path(ctrl).parent) + os.pathsep + env.get("LD_LIBRARY_PATH", "")
        subprocess.run(
            [str(ctrl), "getliveview", "--of=" + str(dest)],
            cwd=str(Path(ctrl).parent), env=env, timeout=6, check=False,
        )
        data = Path(dest).read_bytes()
        jpeg = extract_jpeg(data)
        if not jpeg:
            raise RuntimeError("control getliveview produced no JPEG")
        return jpeg
    finally:
        if tmp:
            try:
                os.remove(tmp)
            except Exception:
                pass


def grab_jpeg(index=None, prefer_ptp=None):
    global _last_err, _prefer_ptp
    errs = []
    steps = []
    if time.time() < float(_ptp_denied_until or 0):
        use_ptp = False
    else:
        use_ptp = _prefer_ptp if prefer_ptp is None else bool(prefer_ptp)
    if use_ptp:
        steps = [grab_via_usb, grab_via_control]
    else:
        steps = [grab_via_usb, grab_via_control]
    for fn in steps:
        try:
            blob = fn()
            if blob:
                _last_err = ""
                if fn is grab_via_usb:
                    _prefer_ptp = True
                return blob
        except Exception as e:
            errs.append(str(e))
    _last_err = " | ".join(errs)
    return None


def set_recording(on):
    global _last_err, _prefer_ptp, _ptp_denied_until
    on = bool(on)
    last = None
    if time.time() < float(_ptp_denied_until or 0) and on:
        _last_err = "PTP denied; skip REC until cooldown"
        return False
    for attempt in range(2):
        try:
            if attempt:
                time.sleep(0.25)
            sess = _ensure_usb()
            sess.set_movie(on)
            _last_err = ""
            print("sony movie", "DOWN/start" if on else "UP/stop")
            return True
        except Exception as e:
            last = e
    msg = str(last)
    if "13" in msg or "Access denied" in msg:
        _ptp_denied_until = time.time() + 15.0
        _prefer_ptp = False
    low = msg.lower()
    if "no sony" in low or "endpoints" in low:
        msg = (
            "USB is Streaming/UVC, not PTP. "
            "In-camera REC needs USB = Remote Shooting. (%s)" % last
        )
    elif "0x2005" in msg:
        msg = "PTP 0x9207 not enabled (0x2005). SDIO handshake required. (%s)" % last
    _last_err = "rec: " + msg
    print("sony movie failed", _last_err)
    return False


_names_cache = ([], 0.0)


def status():
    global _names_cache
    now = time.time()
    names, ts = _names_cache
    if now - ts > 5.0:
        try:
            names = _list_dshow_names()
        except Exception:
            names = names or []
        _names_cache = (names, now)
    return {
        "error": _last_err,
        "usb_session": bool(_session and _session.open),
        "opencv": bool(_cv_cap is not None and _cv_cap.isOpened()),
        "cv_index": _cv_cap_index,
        "devices": names,
    }


def main(argv=None):
    p = argparse.ArgumentParser(description="Sony PTP live view → JPEG")
    p.add_argument("--once", action="store_true")
    p.add_argument("--of", dest="out", default="sony_live.jpg")
    args = p.parse_args(argv)
    blob = grab_jpeg()
    if not blob:
        print(_last_err or "no frame", file=sys.stderr)
        return 1
    Path(args.out).write_bytes(blob)
    print("wrote", args.out, "bytes", len(blob))
    return 0


if __name__ == "__main__":
    sys.exit(main())
