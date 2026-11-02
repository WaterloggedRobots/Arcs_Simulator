"""Game controller (Xbox, PlayStation 4 / 5, ...) for jogging the arm from the
sandbox on a Windows PC, wired (USB) or wireless (Bluetooth / Xbox Wireless
Adapter).

Same controls as the Steam Deck edition, with the shoulder buttons in place of
the Deck's back buttons (L1 / L2 for L4 / L5, R1 / R2 for R4 / R5):

    left stick     up = +x, left = +y
    L1 / L2        +z / -z      (Xbox LB / LT, PlayStation L1 / L2)
    right stick    left = +roll (about x), up = +pitch (about y)
    R1 / R2        +yaw / -yaw  (Xbox RB / RT, PlayStation R1 / R2)
    X              switch between the sim bot and the real bot
                   (the left face button: Xbox X, PlayStation Square)

Read through SDL2 (pygame), which knows Xbox, PlayStation and most other pads,
names the device and maps every one to the same button positions. Without
pygame it falls back to XInput (Xbox and XInput-compatible pads only, no
name). The first controller found is used; a new one is picked up when it is
plugged in or turned on.
"""
from __future__ import annotations

import math
import os
import threading
import time

VERSION = "controller-1.0"
MAPPING = ("left stick up +x, left +y; L1/L2 +z/-z; right stick left +roll, up +pitch; "
           "R1/R2 +yaw/-yaw; X (PlayStation Square) Sim / Real")

DEADZONE = 0.12
TRIGGER_ON = 0.5      # trigger pulled past half counts as pressed
LOOP_S = 0.03
SCAN_S = 1.0          # look for a controller this often while none is connected

# Button labels per controller family, for the UI.
LABELS = {
    "xbox": {"X": "X", "L1": "LB", "L2": "LT", "R1": "RB", "R2": "RT"},
    "playstation": {"X": "Square", "L1": "L1", "L2": "L2", "R1": "R1", "R2": "R2"},
    "nintendo": {"X": "Y", "L1": "L", "L2": "ZL", "R1": "R", "R2": "ZR"},
    "other": {"X": "X", "L1": "L1", "L2": "L2", "R1": "R1", "R2": "R2"},
}

_status = {"backend": None, "connected": False, "name": "", "type": "", "family": "",
           "vendor": "", "product": "", "error": ""}
_lock = threading.Lock()


def _axis(x, power=2.0):
    """-1..1 with a deadzone; power 2 (squared) for fine moves."""
    x = max(-1.0, min(1.0, float(x)))
    a = abs(x)
    if a < DEADZONE:
        return 0.0
    a = (a - DEADZONE) / (1.0 - DEADZONE)
    return math.copysign(a ** power, x)


def map_controls(lx, ly, rx, ry, l1, l2, r1, r2):
    """Sticks in -1..1 (y up = +1), shoulders as booleans, triggers 0..1 ->
    {lin: [x, y, z], rot: [roll, pitch, yaw]} in -1..1, as the Deck's pad."""
    up = (1.0 if l1 else 0.0) - (1.0 if l2 >= TRIGGER_ON else 0.0)
    yaw = (1.0 if r1 else 0.0) - (1.0 if r2 >= TRIGGER_ON else 0.0)
    return {
        # left stick up = +x, left = +y (linear: half tilt is half speed)
        "lin": [_axis(ly, 1.0), -_axis(lx, 1.0), up],
        # right stick left = +roll (about x), up = +pitch (about y)
        "rot": [-_axis(rx), _axis(ry), yaw],
    }


# ---------------------------------------------------------------- device id

# USB vendor / product ids -> (family, type)
_VENDORS = {0x045E: "xbox", 0x054C: "playstation", 0x057E: "nintendo"}
_SONY = {
    0x05C4: "PS4 (DualShock 4)", 0x09CC: "PS4 (DualShock 4 v2)", 0x0BA0: "PS4 (wireless adapter)",
    0x0CE6: "PS5 (DualSense)", 0x0DF2: "PS5 (DualSense Edge)",
}
_MICROSOFT = {
    0x028E: "Xbox 360", 0x0719: "Xbox 360 (wireless receiver)", 0x02D1: "Xbox One",
    0x02DD: "Xbox One", 0x02EA: "Xbox One S", 0x02E0: "Xbox One S (Bluetooth)",
    0x02FD: "Xbox One S (Bluetooth)", 0x0B00: "Xbox Elite 2", 0x0B05: "Xbox Elite 2 (Bluetooth)",
    0x0B12: "Xbox Series X|S", 0x0B13: "Xbox Series X|S (Bluetooth)",
    0x02E6: "Xbox (Wireless Adapter)", 0x02FE: "Xbox (Wireless Adapter)",
}


def identify(name, guid=""):
    """(family, type, vendor id, product id) from the SDL GUID (USB ids) and
    the device name."""
    vid = pid = 0
    g = str(guid or "")
    if len(g) >= 20:
        try:
            # SDL GUID: bus (2 bytes), crc (2), vendor (2, little endian), 0, product (2), ...
            vid = int(g[10:12] + g[8:10], 16)
            pid = int(g[18:20] + g[16:18], 16)
        except ValueError:
            vid = pid = 0
    family = _VENDORS.get(vid, "")
    n = str(name or "").lower()
    if not family:
        if "xbox" in n or "xinput" in n or "x-box" in n:
            family = "xbox"
        elif "ps4" in n or "ps5" in n or "dualshock" in n or "dualsense" in n or "playstation" in n \
                or "wireless controller" == n.strip():
            family = "playstation"
        elif "nintendo" in n or "switch" in n or "joy-con" in n:
            family = "nintendo"
        else:
            family = "other"
    kind = ""
    if family == "playstation":
        kind = _SONY.get(pid, "PlayStation")
    elif family == "xbox":
        kind = _MICROSOFT.get(pid, "Xbox")
    elif family == "nintendo":
        kind = "Nintendo"
    else:
        kind = "Other"
    return family, kind, ("%04x" % vid if vid else ""), ("%04x" % pid if pid else "")


def _set_device(backend, name, guid=""):
    family, kind, vid, pid = identify(name, guid)
    changed = not _status["connected"] or _status["name"] != name
    _status.update(backend=backend, connected=True, name=name, family=family, type=kind,
                   vendor=vid, product=pid, error="")
    if changed:
        print("controller: connected %s [%s%s] via %s" % (
            name, kind, (", %s:%s" % (vid, pid)) if vid else "", backend))


def _drop_device(reason=""):
    if _status["connected"]:
        print("controller: disconnected %s%s" % (_status["name"], (" (%s)" % reason) if reason else ""))
    _status.update(connected=False, name="", family="", type="", vendor="", product="")


# ---------------------------------------------------------------- SDL (pygame)

class _SDL:
    name = "SDL"

    def __init__(self):
        # Keep reading while the browser (not a pygame window) has focus.
        os.environ.setdefault("SDL_JOYSTICK_ALLOW_BACKGROUND_EVENTS", "1")
        os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
        import pygame
        from pygame._sdl2 import controller
        self.pg = pygame
        self.c = controller
        pygame.joystick.init()
        controller.init()
        self.ctrl = None
        self.last_scan = 0.0

    def _open(self):
        now = time.time()
        if now - self.last_scan < SCAN_S:
            return None
        self.last_scan = now
        for i in range(self.c.get_count()):
            if not self.c.is_controller(i):
                continue
            try:
                ctrl = self.c.Controller(i)
                guid = ""
                try:
                    guid = ctrl.as_joystick().get_guid()
                except Exception:
                    pass
                _set_device("SDL", ctrl.name or "Game controller", guid)
                return ctrl
            except Exception as e:
                _status["error"] = "open failed: %s" % e
        return None

    def read(self):
        self.c.update()  # also notices controllers being plugged in or removed
        if self.ctrl is not None and not self.ctrl.attached():
            try:
                self.ctrl.quit()
            except Exception:
                pass
            self.ctrl = None
            _drop_device("unplugged or turned off")
        if self.ctrl is None:
            self.ctrl = self._open()
            if self.ctrl is None:
                return None
        pg, c = self.pg, self.ctrl

        def ax(k):
            return c.get_axis(k) / 32767.0

        pad = map_controls(
            ax(pg.CONTROLLER_AXIS_LEFTX), -ax(pg.CONTROLLER_AXIS_LEFTY),
            ax(pg.CONTROLLER_AXIS_RIGHTX), -ax(pg.CONTROLLER_AXIS_RIGHTY),
            c.get_button(pg.CONTROLLER_BUTTON_LEFTSHOULDER), ax(pg.CONTROLLER_AXIS_TRIGGERLEFT),
            c.get_button(pg.CONTROLLER_BUTTON_RIGHTSHOULDER), ax(pg.CONTROLLER_AXIS_TRIGGERRIGHT),
        )
        pad["buttons"] = {"X"} if c.get_button(pg.CONTROLLER_BUTTON_X) else set()
        return pad


# ---------------------------------------------------------------- XInput fallback

class _XInput:
    name = "XInput"

    def __init__(self):
        from deck import pad as xpad
        if xpad._load_xinput() is None:
            raise RuntimeError("XInput not found")
        self.x = xpad

    def read(self):
        p = self.x.read()
        if p is None:
            _drop_device()
            return None
        _set_device("XInput", "XInput controller #%d" % (int(self.x._status.get("index") or 0) + 1))
        p["buttons"] = {"X"} if p["buttons"] & self.x.BUTTONS["X"] else set()
        return p


def _open_backend():
    errors = []
    for cls in (_SDL, _XInput):
        try:
            b = cls()
            _status["backend"] = b.name
            return b
        except Exception as e:
            errors.append("%s: %s" % (cls.name, e))
    _status["error"] = "; ".join(errors) or "no controller backend"
    return None


def status():
    out = dict(_status)
    out["labels"] = dict(LABELS.get(_status.get("family") or "other", LABELS["other"]))
    out["version"] = VERSION
    return out


def start(step, on_button=None):
    """Poll the controller and call step(lin, rot, dt) while any control is
    held, and step(None, None, 0) once when they are all let go. on_button(name)
    runs once per press of a button ("X")."""

    def _loop():
        backend = _open_backend()
        if backend is None:
            print("controller:", _status["error"])
            return
        print("controller: reading pads via %s" % backend.name)
        last = time.time()
        moving = False
        was = set()
        while True:
            now = time.time()
            dt = min(0.1, now - last)
            last = now
            try:
                pad = backend.read()
                now_b = pad["buttons"] if pad else set()
                if on_button is not None:
                    for name in now_b - was:
                        on_button(name)
                was = now_b
                held = bool(pad) and any(abs(v) > 0 for v in pad["lin"] + pad["rot"])
                if held:
                    step(pad["lin"], pad["rot"], dt)
                    moving = True
                elif moving:
                    step(None, None, 0.0)
                    moving = False
            except Exception as e:
                print("controller step failed", e)
                time.sleep(0.5)
            time.sleep(LOOP_S)

    t = threading.Thread(target=_loop, daemon=True, name="controller")
    t.start()
    return t
