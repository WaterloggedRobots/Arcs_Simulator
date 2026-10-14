"""Heavy session debug. Writes Scripts/debug_run.log and prints [DBG]."""
from __future__ import annotations

import functools
import sys
import threading
import time
import traceback
from pathlib import Path

_LOCK = threading.Lock()
_LOG = Path(__file__).resolve().parent.parent / "debug_run.log"
_last = {}
_t0 = time.perf_counter()
_counts = {}


def _now():
    return time.strftime("%H:%M:%S") + ".%03d" % (int((time.time() % 1) * 1000),)


def _short(v, n=160):
    try:
        if hasattr(v, "shape"):
            return "ndarray%s" % (tuple(v.shape),)
        if isinstance(v, (list, tuple)) and len(v) > 12:
            return "%s[len=%d]" % (type(v).__name__, len(v))
        if isinstance(v, dict) and len(v) > 12:
            return "dict[keys=%d]" % len(v)
        s = repr(v)
    except Exception:
        s = "<%s>" % type(v).__name__
    s = s.replace("\n", " ")
    if len(s) > n:
        return s[:n] + "..."
    return s


def log(tag, msg="", **kv):
    extra = ""
    if kv:
        bits = []
        for k, v in kv.items():
            if isinstance(v, float):
                bits.append("%s=%.4f" % (k, v))
            elif isinstance(v, bool) or v is None or isinstance(v, (int, str)):
                bits.append("%s=%s" % (k, v))
            else:
                bits.append("%s=%s" % (k, _short(v)))
        extra = " " + " ".join(bits)
    line = "[DBG +%.3fs] %s %s%s" % (time.perf_counter() - _t0, tag, msg, extra)
    with _LOCK:
        try:
            with _LOG.open("a", encoding="utf-8") as f:
                f.write(_now() + " " + line + "\n")
        except Exception:
            pass
    try:
        print(line)
    except Exception:
        pass


def every(key, period_s, tag, msg="", **kv):
    now = time.perf_counter()
    prev = _last.get(key, 0.0)
    if now - prev < float(period_s):
        return False
    _last[key] = now
    log(tag, msg, **kv)
    return True


def count(tag, msg=""):
    k = tag + "|" + str(msg)
    _counts[k] = _counts.get(k, 0) + 1
    n = _counts[k]
    if n in (1, 2, 5, 10, 25, 50) or n % 100 == 0:
        log(tag, msg, n=n)


def catch(tag, fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except Exception as e:
        log(tag, "EXCEPTION %s" % e)
        log(tag, traceback.format_exc())
        raise


def wrap(tag, every_s=0.0, log_args=True):
    def deco(fn):
        if getattr(fn, "_dbg_wrapped", False):
            return fn
        name = getattr(fn, "__name__", "fn")

        @functools.wraps(fn)
        def inner(*a, **k):
            t0 = time.perf_counter()
            kv = {}
            if log_args:
                if a:
                    kv["args"] = a[1:] if (a and getattr(a[0], "__class__", None) and name in dir(a[0].__class__)) else a
                if k:
                    kv["kwargs"] = k
            try:
                out = fn(*a, **k)
            except Exception as e:
                log(tag, name + " EXCEPTION", err=str(e), dt=time.perf_counter() - t0, **kv)
                log(tag, traceback.format_exc())
                raise
            dt = time.perf_counter() - t0
            kv["dt"] = dt
            kv["ret"] = out
            if every_s:
                if dt >= 0.08:
                    log(tag, name + " SLOW", **kv)
                else:
                    every("wrap:%s:%s" % (tag, name), every_s, tag, name, **kv)
            else:
                log(tag, name, **kv)
            return out

        inner._dbg_wrapped = True
        return inner

    return deco


def patch(mod, names, tag, every_s=0.0, log_args=True):
    n_ok = 0
    for n in names:
        fn = getattr(mod, n, None)
        if not callable(fn):
            log("patch", "missing", mod=getattr(mod, "__name__", mod), name=n)
            continue
        setattr(mod, n, wrap(tag, every_s=every_s, log_args=log_args)(fn))
        n_ok += 1
    log("patch", getattr(mod, "__name__", str(mod)), n=n_ok)


def patch_methods(obj, names, tag, every_s=0.0):
    for n in names:
        fn = getattr(obj, n, None)
        if not callable(fn):
            continue
        setattr(obj, n, wrap(tag, every_s=every_s)(fn))


def sample_gaps(samples, tag="path"):
    if not samples:
        log(tag, "no samples")
        return
    ts = []
    for s in samples:
        try:
            ts.append(float(s.get("t") if isinstance(s, dict) else s["t"]))
        except Exception:
            pass
    if len(ts) < 2:
        log(tag, "samples", n=len(samples), t0=ts[0] if ts else None)
        return
    dts = [ts[i + 1] - ts[i] for i in range(len(ts) - 1)]
    back = sum(1 for d in dts if d < -1e-6)
    big = max(dts) if dts else 0
    log(
        tag,
        "timing",
        n=len(ts),
        t0=ts[0],
        t1=ts[-1],
        dt_min=min(dts),
        dt_max=big,
        dt_mean=sum(dts) / len(dts),
        backward=back,
    )
    if big > 0.15:
        log(tag, "LARGE SAMPLE GAP", dt_max=big)


def install_hooks():
    prev = sys.excepthook

    def _hook(t, v, tb):
        log("crash", "".join(traceback.format_exception(t, v, tb)))
        try:
            prev(t, v, tb)
        except Exception:
            pass

    sys.excepthook = _hook

    if hasattr(threading, "excepthook"):
        def _th(args):
            log(
                "thread_crash",
                str(args.exc_value),
                thread=getattr(args.thread, "name", None),
            )
            log("thread_crash", "".join(traceback.format_exception(args.exc_type, args.exc_value, args.exc_traceback)))

        threading.excepthook = _th


def reset():
    global _t0, _counts
    _t0 = time.perf_counter()
    _counts = {}
    try:
        _LOG.write_text("=== debug_run start %s ===\n" % time.strftime("%Y-%m-%d %H:%M:%S"), encoding="utf-8")
    except Exception:
        pass
    install_hooks()
    log("boot", "debug_trace ready", path=str(_LOG), py=sys.version.split()[0])


def install_app_hooks():
    """Call after robot_runtime / bake / waypoints / file_manage imported."""
    try:
        import robot_runtime as rr
        patch(
            rr,
            [
                "enter_simulation_mode",
                "exit_simulation_mode",
                "toggle_sim_pause",
                "set_sim_time",
                "upload_real_path",
                "fire_real_path",
                "pause_real_path",
                "stop_real_path",
                "start_real_path",
                "start_hold_move",
                "stop_hold_move",
                "start_goto_path_start",
                "reset_sim_to_real",
                "reset_sim_to_home",
                "request_halt",
                "drop_uploaded_buffer",
                "plan_joint_path",
                "set_hide_real_robot",
                "reload_robot_models",
                "set_track_object",
                "apply_camera_view",
            ],
            "rr",
            every_s=0.0,
        )
    except Exception as e:
        log("patch", "robot_runtime failed", err=str(e))
    try:
        import path_bake as bake
        patch(
            bake,
            [
                "run_bake",
                "clear_bake",
                "write_overspeed_report",
                "export_overspeed",
            ],
            "bake",
        )
        patch(
            bake,
            ["evaluate_timing", "rebuild_preview"],
            "bake_tick",
            every_s=0.8,
            log_args=False,
        )
    except Exception as e:
        log("patch", "path_bake failed", err=str(e))
    try:
        import waypoints as wp
        patch(
            wp,
            [
                "delete_selected",
                "select_waypoint",
                "deselect_all",
                "undo",
                "redo",
                "clear_waypoints",
                "capture_saved_path",
                "place_saved_path",
                "add_sim_pose",
                "set_mode",
                "apply_gizmo_space",
            ],
            "wp",
            every_s=0.0,
        )
    except Exception as e:
        log("patch", "waypoints failed", err=str(e))
    try:
        import file_manage as fman
        patch(
            fman,
            [
                "load_from_path",
                "write_save_file",
                "write_export_json",
                "apply_loaded_doc",
                "new_project",
                "write_autosave",
                "parse_project_doc",
            ],
            "file",
        )
    except Exception as e:
        log("patch", "file_manage failed", err=str(e))
    try:
        import arcscontrol as ac
        if hasattr(ac, "RobotArcs"):
            cls = ac.RobotArcs
            patch_methods(
                cls,
                [
                    "connect",
                    "reconnect",
                    "MoveLAwait",
                    "MoveLAsync",
                    "MoveJAwait",
                    "MoveJAsync",
                    "StopMove",
                    "PauseMove",
                    "AbortPathBuffer",
                    "StartMove",
                    "GetIK",
                    "prepare_path_buffer",
                    "run_path_buffer",
                    "plan_joint_path",
                ],
                "rpc",
            )
    except Exception as e:
        log("patch", "arcscontrol failed", err=str(e))
    log("boot", "function hooks installed")
