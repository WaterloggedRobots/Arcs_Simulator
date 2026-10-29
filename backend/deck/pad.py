"""Steam Deck controls for the deck edition, read over XInput.

With the Arcs Deck profile (Steam Deck Tools) the Deck shows up as an X360
pad: sticks as sticks, L4 / R4 as LB / RB, L5 / R5 as LT / RT. Trackpad mouse
and L2 / R2 clicks are done by SteamController.exe itself, not here.

    left stick     up = +x, left = +y
    L4 / L5        +z / -z
    right stick    left = +roll (about x), up = +pitch (about y)
    R4 / R5        +yaw / -yaw (about z)
    X              switch the sticks between the sim bot and the real bot

Axes follow the Axes Mode toggle: Local = the sim cursor's own axes,
Global = world axes.
"""
from __future__ import annotations

import ctypes
import math
import os
import threading
import time

VERSION = "deck-1.7"
MAPPING = "left stick up +x, left +y; L4/L5 +z/-z; right stick left +roll, up +pitch; R4/R5 +yaw/-yaw"

LB = 0x0100
RB = 0x0200
# Buttons reported to on_button (pressed, not held).
BUTTONS = {"X": 0x4000}
DEADZONE = 0.12
TRIGGER_ON = 128
LOOP_S = 0.03


class _Gamepad(ctypes.Structure):
    _fields_ = [
        ("wButtons", ctypes.c_ushort),
        ("bLeftTrigger", ctypes.c_ubyte),
        ("bRightTrigger", ctypes.c_ubyte),
        ("sThumbLX", ctypes.c_short),
        ("sThumbLY", ctypes.c_short),
        ("sThumbRX", ctypes.c_short),
        ("sThumbRY", ctypes.c_short),
    ]


class _State(ctypes.Structure):
    _fields_ = [("dwPacketNumber", ctypes.c_uint), ("Gamepad", _Gamepad)]


_xinput = None
_status = {"backend": None, "connected": False, "index": None, "error": ""}


def _load_xinput():
    global _xinput
    if _xinput is not None or os.name != "nt":
        return _xinput
    for name in ("xinput1_4", "xinput1_3", "xinput9_1_0"):
        try:
            _xinput = getattr(ctypes.windll, name)
            _status["backend"] = name
            return _xinput
        except Exception:
            continue
    _status["error"] = "XInput not found"
    return None


def _axis(v, power=2.0):
    """-32768..32767 to -1..1 with a deadzone; power 2 (squared) for fine moves."""
    x = max(-1.0, min(1.0, float(v) / 32767.0))
    a = abs(x)
    if a < DEADZONE:
        return 0.0
    a = (a - DEADZONE) / (1.0 - DEADZONE)
    return math.copysign(a ** power, x)


def read():
    """The first connected pad as {lin: [x, y, z], rot: [roll, pitch, yaw]}, or None."""
    dll = _load_xinput()
    if dll is None:
        _status["connected"] = False
        return None
    st = _State()
    order = [_status["index"]] if _status["index"] is not None else []
    order += [i for i in range(4) if i not in order]
    for i in order:
        if dll.XInputGetState(i, ctypes.byref(st)) != 0:
            continue
        _status["connected"] = True
        _status["index"] = i
        g = st.Gamepad
        up = (1.0 if g.wButtons & LB else 0.0) - (1.0 if g.bLeftTrigger >= TRIGGER_ON else 0.0)
        yaw = (1.0 if g.wButtons & RB else 0.0) - (1.0 if g.bRightTrigger >= TRIGGER_ON else 0.0)
        return {
            "buttons": int(g.wButtons),
            # left stick up = +x, left = +y (linear: half tilt is half speed);
            # right stick left = +roll (about x), up = +pitch (about y)
            "lin": [_axis(g.sThumbLY, 1.0), -_axis(g.sThumbLX, 1.0), up],
            "rot": [-_axis(g.sThumbRX), _axis(g.sThumbRY), yaw],
        }
    _status["connected"] = False
    _status["index"] = None
    return None


def status():
    return dict(_status)


def start(step, on_button=None):
    """Poll the pad and call step(lin, rot, dt) while any control is held,
    and step(None, None, 0) once when they are all let go. on_button(name)
    runs once per press of a button in BUTTONS."""
    if os.name != "nt":
        _status["error"] = "not Windows: no controller input"
        print("deck pad:", _status["error"])
        return None

    def _loop():
        last = time.time()
        moving = False
        was = 0
        while True:
            now = time.time()
            dt = min(0.1, now - last)
            last = now
            try:
                pad = read()
                now_b = pad["buttons"] if pad else 0
                if on_button is not None:
                    for name, bit in BUTTONS.items():
                        if now_b & bit and not was & bit:
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
                print("deck pad step failed", e)
            time.sleep(LOOP_S)

    t = threading.Thread(target=_loop, daemon=True, name="deck-pad")
    t.start()
    return t
