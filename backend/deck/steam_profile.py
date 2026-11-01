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
import threading
import time

PROFILE = "Arcs Deck"
MUTEX_NAME = "ArcsDeckActive"
SHARED_NAME = "Global_SteamControllerSetting_Setting"
_CUR = (0, 128)
_SEL = (128, 2048)
_DES = (128 + 2048, 128)
# Struct size rounded up to SharedData's 256-byte alignment.
SHARED_SIZE = ((128 + 2048 + 128) * 2 + 255) // 256 * 256

# problem: "" (fine), "not_windows", "not_running" (SteamController.exe is
# not running), "no_profile" (a build without Arcs Deck), "lost" (it switched
# to another profile while the app runs). The Deck page shows it as a banner.
_state = {"mutex": None, "previous": None, "active": False, "message": "",
          "problem": "", "current": ""}
_lock = threading.RLock()
WATCH_S = 3.0


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


def _set(problem, message, current=""):
    _state["problem"] = problem
    _state["message"] = message
    _state["current"] = current
    print("steam profile:", message)


def activate():
    """Switch to the Arcs Deck profile. Returns a status dict. Safe to call
    again (Retry on the Deck page, the watcher)."""
    with _lock:
        if not _is_windows():
            _set("not_windows", "not Windows: controller profile not switched")
            return status()
        prev = current_profile()
        if not prev:
            _set("not_running", "SteamController.exe (Steam Deck Tools) is not running")
            return status()
        if prev != PROFILE:
            _state["previous"] = prev
        if not _state.get("mutex"):
            _state["mutex"] = _hold_mutex()
            # It checks the mutex every 0.5 s before the profile becomes selectable.
            time.sleep(0.6)
        if _select(PROFILE):
            _state["active"] = True
            _set("", "switched from %s" % (_state["previous"] or "?"), PROFILE)
            _install_exit_hooks()
        else:
            _state["active"] = False
            cur = current_profile()
            _set("no_profile",
                 "this SteamController.exe has no %r profile (current profile: %s). Install the "
                 "Steam Deck Tools build with the Arcs Deck profile." % (PROFILE, cur or "?"), cur)
            _drop_mutex()
        return status()


def _check():
    """One watcher pass: notice SteamController.exe starting later, or
    leaving the Arcs Deck profile."""
    if not _lock.acquire(blocking=False):
        return
    try:
        cur = current_profile()
        if _state["active"]:
            if not cur:
                _state["active"] = False
                _set("not_running", "SteamController.exe (Steam Deck Tools) stopped")
            elif cur != PROFILE:
                _state["active"] = False
                _set("lost", "SteamController.exe switched to %r" % cur, cur)
        elif _state["problem"] == "not_running" and cur:
            activate()  # it was started after the app
        elif _state["problem"] == "lost" and cur == PROFILE:
            _state["active"] = True
            _set("", "back on %s" % PROFILE, PROFILE)
    finally:
        _lock.release()


def start_watch():
    """Check the profile every WATCH_S seconds (Windows only)."""
    if not _is_windows():
        return

    def loop():
        while True:
            time.sleep(WATCH_S)
            try:
                _check()
            except Exception as e:
                print("steam profile: watch failed", e)

    threading.Thread(target=loop, daemon=True, name="steam-profile").start()


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
        "problem": _state.get("problem") or "",
        "current": _state.get("current") or "",
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
