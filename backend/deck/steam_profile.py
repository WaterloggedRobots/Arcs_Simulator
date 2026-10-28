"""Steam Deck (Windows): switch Steam Deck Tools' SteamController.exe to its
"Arcs Deck" profile while the app runs, and back to the previous profile on
exit.

SteamController.exe shares a SteamControllerSetting struct in a named memory
map (CommonHelpers/SharedData.cs, GlobalConfig.cs): three UTF-16 strings,
CurrentProfile (128 chars), SelectableProfiles (2048), DesiredProfile (128).
Writing DesiredProfile makes it select that profile on its next tick (the same
way PowerControl switches profiles).

The Arcs Deck profile is only selectable while this process holds the
"ArcsDeckActive" mutex, so a crash still gives the user their controls back.
"""
from __future__ import annotations

import atexit
import os
import time

PROFILE = "Arcs Deck"
MUTEX_NAME = "ArcsDeckActive"
SHARED_NAME = "Global_SteamControllerSetting_Setting"
_CUR = (0, 128)
_SEL = (128, 2048)
_DES = (128 + 2048, 128)
# Struct size rounded up to SharedData's 256-byte alignment.
SHARED_SIZE = ((128 + 2048 + 128) * 2 + 255) // 256 * 256

_state = {"mutex": None, "previous": None, "active": False, "message": ""}


def _is_windows():
    return os.name == "nt"


def _open_map():
    import mmap
    return mmap.mmap(-1, SHARED_SIZE, tagname=SHARED_NAME)


def _read_str(m, field):
    off, n = field
    raw = m[off * 2:(off + n) * 2]
    return raw.decode("utf-16-le", errors="ignore").split("\x00", 1)[0]


def _write_str(m, field, text):
    off, n = field
    data = str(text)[: n - 1].encode("utf-16-le")
    data = data + b"\x00" * (n * 2 - len(data))
    m[off * 2:(off + n) * 2] = data


def current_profile():
    """SteamController.exe's current profile name, or "" if it is not running."""
    if not _is_windows():
        return ""
    try:
        m = _open_map()
        try:
            return _read_str(m, _CUR)
        finally:
            m.close()
    except Exception:
        return ""


def selectable_profiles():
    if not _is_windows():
        return []
    try:
        m = _open_map()
        try:
            return [p for p in _read_str(m, _SEL).split("\n") if p]
        finally:
            m.close()
    except Exception:
        return []


def _select(name, timeout=3.0):
    """Ask SteamController.exe for a profile; True once it reports it."""
    m = _open_map()
    try:
        t0 = time.time()
        while time.time() - t0 < timeout:
            if _read_str(m, _CUR) == name:
                return True
            # It rewrites the whole struct every tick, so a write can be lost: retry.
            _write_str(m, _DES, name)
            time.sleep(0.15)
        return _read_str(m, _CUR) == name
    finally:
        m.close()


def _hold_mutex():
    import ctypes
    k32 = ctypes.windll.kernel32
    k32.CreateMutexW.restype = ctypes.c_void_p
    h = k32.CreateMutexW(None, False, MUTEX_NAME)
    return h or None


def _drop_mutex():
    h = _state.get("mutex")
    _state["mutex"] = None
    if not h:
        return
    try:
        import ctypes
        ctypes.windll.kernel32.CloseHandle(ctypes.c_void_p(h))
    except Exception:
        pass


def activate():
    """Switch to the Arcs Deck profile. Returns a status dict."""
    if not _is_windows():
        _state["message"] = "not Windows: controller profile not switched"
        print("steam profile:", _state["message"])
        return status()
    prev = current_profile()
    if not prev:
        _state["message"] = "SteamController.exe is not running: start it (Steam Deck Tools) for the deck controls"
        print("steam profile:", _state["message"])
        return status()
    if prev != PROFILE:
        _state["previous"] = prev
    _state["mutex"] = _hold_mutex()
    # It checks the mutex every 0.5 s before the profile becomes selectable.
    time.sleep(0.6)
    if _select(PROFILE):
        _state["active"] = True
        _state["message"] = "switched from %s" % (_state["previous"] or "?")
        _install_exit_hooks()
    else:
        have = selectable_profiles()
        _state["message"] = (
            "SteamController.exe has no %r profile (have: %s). Install the Steam Deck Tools build "
            "with the Arcs Deck profile, and pick 'Use X360 Controller with Steam'." % (PROFILE, ", ".join(have) or "none")
        )
        _drop_mutex()
    print("steam profile:", _state["message"])
    return status()


def restore():
    """Select the profile that was active before activate(); safe to call twice."""
    if not _state.get("active"):
        _drop_mutex()
        return
    _state["active"] = False
    prev = _state.get("previous") or "Desktop"
    try:
        ok = _select(prev, timeout=2.0)
        print("steam profile: restored", prev if ok else "(failed: %s)" % prev)
    except Exception as e:
        print("steam profile: restore failed", e)
    # Without the mutex the Arcs Deck profile stops being selectable anyway.
    _drop_mutex()


def status():
    return {
        "profile": PROFILE,
        "active": bool(_state.get("active")),
        "previous": _state.get("previous"),
        "message": _state.get("message") or "",
    }


_hooks_installed = False
_ctrl_handler = None


def _install_exit_hooks():
    """Restore on normal exit, Ctrl+C and on closing the console window."""
    global _hooks_installed, _ctrl_handler
    if _hooks_installed:
        return
    _hooks_installed = True
    atexit.register(restore)
    try:
        import ctypes
        from ctypes import wintypes

        HANDLER = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.DWORD)

        def _on_ctrl(_event):
            # Ctrl+C, Ctrl+Break, close, logoff, shutdown
            restore()
            return False

        _ctrl_handler = HANDLER(_on_ctrl)
        ctypes.windll.kernel32.SetConsoleCtrlHandler(_ctrl_handler, True)
    except Exception as e:
        print("steam profile: console handler failed", e)
