"""Gimbal objects: a DJI RS4 (or any gimbal with a driver script) in the scene.

A Gimbal is one scene object (kind "gimbal", one row in the object list) drawn
as two scene nodes:

- the mount: an empty (grey sphere + arrows) at the object's transform. The
  user moves it like any other object (gizmo, Properties, parenting).
- the head: a cube (with an orange lens arrow) parented to the mount. Its local
  rotation is the gimbal's yaw / roll / pitch: the angles read from the gimbal
  while it is connected, otherwise the target angles (a preview).

Angles are the gimbal's joint angles in degrees. The head's frame matches the
mount's: z forward (lens), x right, y down. Yaw turns about y (positive turns
right), roll about the yawed z (positive drops the right side), pitch about
the yawed and rolled x (positive tilts up), in the motor order yaw > roll > pitch.

Timeline: every gimbal gets a track "gimbal_<id>" with frames
{t, yaw, roll, pitch}. The target follows the track (linear between frames)
whenever the playhead moves, during Simulate and during Run. During a real
Run the gimbal is driven along the track; before it, while the arm moves to
the path start, the gimbal goes to its initial pose (the track at t = 0).

All gimbal I/O runs on one thread (_loop): requests from the web server are
queued in each gimbal's runtime record. The driver is a Python script with
connect(cfg), get_angles(), move_to(yaw, roll, pitch, duration) and close();
the default is user_files/gimbal_scripts/rs4_can.py.
"""
import importlib.util
import math
import os
import threading
import time

import numpy as np

import app_paths
import arms
import waypoints as wp

KIND = "gimbal"
TRACK_PREFIX = "gimbal_"
DEFAULT_SCRIPT = "rs4_can.py"
PERIOD_S = 0.1
RECONNECT_S = 2.0
MAX_MOVE_S = 25.0  # the SDK's time_for_action tops out at 25.5 s
HEAD_RGB = (70, 70, 78)
LENS_RGB = (255, 150, 40)
HEAD_SIZE = 0.06
AXES = ("yaw", "roll", "pitch")
LIMITS = {"yaw": (-180.0, 180.0), "roll": (-30.0, 30.0), "pitch": (-56.0, 146.0)}

DEFAULT_CFG = {
    "connect": False,          # talk to the gimbal (off: preview only)
    "script": "",              # driver script; "" = the bundled RS4 script
    "interface": "canalystii",  # python-can interface
    "channel": 0,
    "bitrate": 1000000,
    "move_s": 2.0,             # time for Move and the move to the initial pose
}

# Runtime per gimbal id (not saved, not in undo).
_rt = {}
_rt_lock = threading.RLock()
_thread = None


# ---------------------------------------------------------------------------
# Document data
# ---------------------------------------------------------------------------
def _f(v, default=0.0):
    try:
        v = float(v)
        return v if math.isfinite(v) else default
    except Exception:
        return default


def clamp_angles(a):
    a = list(a or [0.0, 0.0, 0.0]) + [0.0, 0.0, 0.0]
    out = []
    for i, axis in enumerate(AXES):
        lo, hi = LIMITS[axis]
        out.append(max(lo, min(hi, _f(a[i]))))
    return out


def clean_gimbal(rec):
    """The saved part of a gimbal: settings and target angles."""
    rec = rec if isinstance(rec, dict) else {}
    cfg = dict(DEFAULT_CFG)
    src = rec.get("cfg") if isinstance(rec.get("cfg"), dict) else {}
    for k in DEFAULT_CFG:
        if k in src and src[k] is not None:
            cfg[k] = src[k]
    cfg["connect"] = bool(cfg["connect"])
    cfg["script"] = str(cfg["script"] or "")
    cfg["interface"] = str(cfg["interface"] or "canalystii")
    cfg["channel"] = int(_f(cfg["channel"], 0))
    cfg["bitrate"] = int(_f(cfg["bitrate"], 1000000))
    cfg["move_s"] = max(0.1, min(MAX_MOVE_S, _f(cfg["move_s"], 2.0)))
    return {"cfg": cfg, "target": clamp_angles(rec.get("target"))}


def is_gimbal(obj):
    return bool(obj) and str(obj.get("kind") or "") == KIND


def gimbal_objects():
    return [o for o in list(wp.scene_objects or []) if is_gimbal(o)]


def _find(oid):
    for o in list(wp.scene_objects or []):
        if str(o.get("id")) == str(oid) and is_gimbal(o):
            return o
    return None


def track_id(oid):
    return TRACK_PREFIX + str(oid)


def _track_name(obj):
    raw = str((obj or {}).get("name") or (obj or {}).get("id") or "Gimbal").strip()
    return raw if raw.lower().endswith("gimbal") else raw + " gimbal"


def sync_tracks(tracks):
    """Keep one track per gimbal object (frames kept), drop orphans."""
    objs = {str(o.get("id")): o for o in gimbal_objects()}
    out, seen = [], set()
    for tr in list(tracks or []):
        tid = str((tr or {}).get("id") or "") if isinstance(tr, dict) else ""
        if tid.startswith(TRACK_PREFIX):
            oid = tid[len(TRACK_PREFIX):]
            if oid not in objs or oid in seen:
                continue
            tr["name"] = _track_name(objs[oid])
            tr["gimbal"] = True
            tr["keys"] = clean_keys(tr.get("keys"))
            seen.add(oid)
        out.append(tr)
    for oid, obj in objs.items():
        if oid not in seen:
            out.append({"id": track_id(oid), "name": _track_name(obj), "gimbal": True, "keys": []})
    return out


def clean_keys(keys):
    out = []
    for k in list(keys or []):
        if not isinstance(k, dict) or k.get("t") is None:
            continue
        y, r, p = clamp_angles([k.get("yaw"), k.get("roll"), k.get("pitch")])
        out.append({"t": max(0.0, _f(k.get("t"))), "yaw": y, "roll": r, "pitch": p})
    out.sort(key=lambda k: k["t"])
    return out


def _track(oid, create=False):
    tid = track_id(oid)
    for tr in list(wp.extra_tracks or []):
        if isinstance(tr, dict) and str(tr.get("id") or "") == tid:
            return tr
    if not create:
        return None
    obj = _find(oid)
    tr = {"id": tid, "name": _track_name(obj), "gimbal": True, "keys": []}
    wp.extra_tracks = list(wp.extra_tracks or []) + [tr]
    return tr


def keys_for(oid):
    tr = _track(oid)
    return clean_keys((tr or {}).get("keys"))


def angles_at(keys, t):
    """Track value at time t: linear between frames, held before / after."""
    if not keys:
        return None
    t = _f(t)
    if t <= keys[0]["t"]:
        k = keys[0]
        return [k["yaw"], k["roll"], k["pitch"]]
    for a, b in zip(keys, keys[1:]):
        if t <= b["t"]:
            span = b["t"] - a["t"]
            u = 1.0 if span <= 1e-9 else (t - a["t"]) / span
            return [a[x] + (b[x] - a[x]) * u for x in AXES]
    k = keys[-1]
    return [k["yaw"], k["roll"], k["pitch"]]


def initial_angles(oid):
    return angles_at(keys_for(oid), 0.0)


FRAME_EPS = 1e-4


def insert_frame(oid, t, angles=None):
    """Frame at t with the given (or the target) angles; replaces one at t."""
    obj = _find(oid)
    if obj is None:
        return False
    g = obj.setdefault("gimbal", clean_gimbal(None))
    ang = clamp_angles(angles if angles is not None else g.get("target"))
    g["target"] = ang
    t = max(0.0, _f(t))
    tr = _track(oid, create=True)
    keys = [k for k in clean_keys(tr.get("keys")) if abs(k["t"] - t) > FRAME_EPS]
    keys.append({"t": t, "yaw": ang[0], "roll": ang[1], "pitch": ang[2]})
    tr["keys"] = clean_keys(keys)
    _note_keys_changed(oid)
    return True


def delete_frame(oid, t=None, index=None):
    tr = _track(oid)
    if tr is None:
        return False
    keys = clean_keys(tr.get("keys"))
    n = len(keys)
    if index is not None:
        i = int(_f(index, -1))
        keys = [k for j, k in enumerate(keys) if j != i]
    elif t is not None:
        keys = [k for k in keys if abs(k["t"] - _f(t)) > FRAME_EPS]
    tr["keys"] = keys
    _note_keys_changed(oid)
    return len(keys) != n


def _note_keys_changed(oid):
    with _rt_lock:
        rt = _rt.get(str(oid))
        if rt is not None:
            rt["follow"] = True


# ---------------------------------------------------------------------------
# Scene: the head cube under the mount empty
# ---------------------------------------------------------------------------
def _quat_axis(axis, deg):
    h = math.radians(_f(deg)) * 0.5
    s = math.sin(h)
    return np.array([math.cos(h), axis[0] * s, axis[1] * s, axis[2] * s], dtype=float)


def head_wxyz(angles):
    """Local rotation of the head for (yaw, roll, pitch) in degrees."""
    import simulatorview as sv
    y, r, p = (list(angles or [0, 0, 0]) + [0, 0, 0])[:3]
    q = sv._qmul(sv._qmul(_quat_axis((0, 1, 0), y), _quat_axis((0, 0, 1), r)), _quat_axis((1, 0, 0), p))
    n = float(np.linalg.norm(q)) or 1.0
    return q / n


def spawn_head(obj):
    """Cube + lens arrow as children of the mount's node (follow its pose and visibility)."""
    if wp.server is None or obj.get("handle") is None:
        return
    name = "/obj_%s/mesh/head" % obj["id"]
    sc = np.asarray(obj.get("scale", [1, 1, 1]), dtype=float).reshape(3)
    try:
        head = wp.server.scene.add_box(
            name=name,
            dimensions=tuple(HEAD_SIZE * float(v) for v in sc),
            color=HEAD_RGB,
            wxyz=tuple(float(v) for v in head_wxyz((obj.get("gimbal") or {}).get("target"))),
            position=(0.0, 0.0, 0.0),
        )
        wp.add_forward_arrow(name, length=0.11, colors=LENS_RGB)
    except Exception as e:
        print("gimbal head failed", e)
        return
    obj["head"] = head
    oid = obj["id"]

    def _on_click(_e=None, _oid=oid):
        import simulatorview as sv
        mode = "add" if wp._select_ctrl else ("range" if wp._select_shift else "replace")
        sv.select_scene_object(_oid, mode=mode)
    try:
        head.on_click(_on_click)
    except Exception:
        pass


def scale_head(obj):
    h = obj.get("head")
    if h is None:
        return
    sc = np.asarray(obj.get("scale", [1, 1, 1]), dtype=float).reshape(3)
    try:
        h.dimensions = tuple(HEAD_SIZE * float(v) for v in sc)
    except Exception:
        pass


def _set_head(obj, angles):
    h = obj.get("head")
    if h is None or angles is None:
        return
    key = tuple(round(float(v), 3) for v in angles)
    if obj.get("_head_shown") == key:
        return
    obj["_head_shown"] = key
    try:
        h.wxyz = tuple(float(v) for v in head_wxyz(angles))
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Runtime
# ---------------------------------------------------------------------------
def _new_rt():
    return {
        "mod": None, "mod_path": None, "connected": False, "conn_err": "",
        "next_try": 0.0, "read": None, "read_ok": False, "read_err": "",
        "read_at": 0.0, "fails": 0, "requests": [], "running": False,
        "cmd_end": -1.0, "last_play": None, "follow": True, "cfg_sig": None,
        "last_cmd": None, "cmd_err": "",
    }


def _rt_for(oid):
    with _rt_lock:
        rt = _rt.get(str(oid))
        if rt is None:
            rt = _rt[str(oid)] = _new_rt()
        return rt


def resolve_script(cfg):
    path = str((cfg or {}).get("script") or "").strip()
    if path:
        moved = app_paths.relocate(path, app_paths.GIMBAL_SCRIPTS_DIR)
        if os.path.isfile(moved):
            return moved
        return path
    return str(app_paths.GIMBAL_SCRIPTS_DIR / DEFAULT_SCRIPT)


def _load_module(oid, path):
    spec = importlib.util.spec_from_file_location("user_gimbal_%s" % oid, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    for fn in ("connect", "get_angles", "move_to"):
        if not callable(getattr(mod, fn, None)):
            raise AttributeError("script has no %s()" % fn)
    return mod


def _disconnect(rt):
    mod = rt.get("mod")
    if mod is not None and rt.get("connected") and callable(getattr(mod, "close", None)):
        try:
            mod.close()
        except Exception as e:
            print("gimbal close failed", e)
    rt["connected"] = False
    rt["read_ok"] = False


def _cfg_sig(cfg):
    return (resolve_script(cfg), cfg.get("interface"), cfg.get("channel"), cfg.get("bitrate"))


def _ensure_connected(oid, cfg, rt):
    sig = _cfg_sig(cfg)
    if rt["connected"] and rt["cfg_sig"] == sig:
        return True
    if rt["connected"]:
        _disconnect(rt)
    now = time.time()
    if now < rt["next_try"]:
        return False
    rt["next_try"] = now + RECONNECT_S
    rt["cfg_sig"] = sig
    path = sig[0]
    try:
        if rt["mod"] is None or rt["mod_path"] != path:
            if not os.path.isfile(path):
                raise FileNotFoundError("script not found: %s" % path)
            rt["mod"] = _load_module(oid, path)
            rt["mod_path"] = path
        rt["mod"].connect(dict(cfg))
    except Exception as e:
        rt["conn_err"] = "%s: %s" % (type(e).__name__, e)
        rt["read_ok"] = False
        rt["read_err"] = "not connected (%s)" % rt["conn_err"]
        return False
    rt["connected"] = True
    rt["conn_err"] = ""
    rt["fails"] = 0
    print("gimbal connected", oid, path)
    return True


def _run_states():
    """[(_rp, _pb)] of every arm, read without swapping arms."""
    import robot_runtime as rr
    out = []
    with arms.LOCK:
        cur = arms.current()
        for aid in arms.each():
            if aid == cur:
                out.append((getattr(rr, "_rp", None) or {}, getattr(rr, "_pb", None) or {}))
                continue
            st = ((arms._arms.get(aid) or {}).get("state") or {}).get("robot_runtime") or {}
            out.append((st.get("_rp") or {}, st.get("_pb") or {}))
    return out


def _clock():
    """(timeline t, real run active, sim playing)."""
    import simulatorview as sv
    real_t, real_on, sim_on = None, False, False
    try:
        for rp, pb in _run_states():
            if rp.get("active") and rp.get("armed") and not rp.get("paused"):
                real_on = True
                real_t = _f(rp.get("t"))
            if pb.get("active") and not pb.get("paused"):
                sim_on = True
    except Exception:
        pass
    t = real_t if real_on else _f(sv._playhead_t())
    return t, real_on, sim_on


def _move(oid, rt, angles, duration):
    mod = rt["mod"]
    y, r, p = clamp_angles(angles)
    try:
        ok = mod.move_to(y, r, p, duration=max(0.1, min(MAX_MOVE_S, float(duration))))
        rt["last_cmd"] = [y, r, p]
        rt["cmd_err"] = "" if ok is not False else "gimbal refused the move"
        return ok is not False
    except Exception as e:
        rt["cmd_err"] = "move failed: %s" % e
        return False


def _drive_run(oid, rt, keys, t):
    """During a real run: command the next frame, timed to arrive on it."""
    if not keys or t < rt["cmd_end"] - 1e-3:
        return
    nxt = next((k for k in keys if k["t"] > t + 0.05), None)
    if nxt is None:
        return  # past the last frame: hold
    tt = min(nxt["t"], t + MAX_MOVE_S)
    dur = max(0.1, round(tt - t, 1))
    if _move(oid, rt, angles_at(keys, tt), dur):
        rt["cmd_end"] = t + dur
    else:
        rt["cmd_end"] = t + 0.2  # retry soon


def _read(oid, rt):
    try:
        a = rt["mod"].get_angles()
        if a is None:
            raise RuntimeError("no angles returned")
        a = [float(v) for v in list(a)[:3]]
        if len(a) < 3 or not all(math.isfinite(v) for v in a):
            raise RuntimeError("bad angles %r" % (a,))
        rt["read"] = a
        rt["read_ok"] = True
        rt["read_err"] = ""
        rt["read_at"] = time.time()
        rt["fails"] = 0
    except Exception as e:
        rt["read_ok"] = False
        rt["read_err"] = str(e) or type(e).__name__
        rt["fails"] += 1
        if rt["fails"] >= 20:  # ~2 s of failures: reopen the bus
            _disconnect(rt)
            rt["read_err"] = "%s (reconnecting)" % rt["read_err"]


def _tick_one(obj, clock):
    oid = str(obj.get("id"))
    g = obj.get("gimbal")
    if not isinstance(g, dict):
        g = obj["gimbal"] = clean_gimbal(g)
    cfg = g.get("cfg") or dict(DEFAULT_CFG)
    rt = _rt_for(oid)
    t, real_on, sim_on = clock
    keys = keys_for(oid)
    # Target follows the track while it plays or when the playhead moves.
    moved = rt["last_play"] is None or abs(t - rt["last_play"]) > 1e-6
    if keys and (real_on or sim_on or moved or rt["follow"]):
        g["target"] = clamp_angles(angles_at(keys, t))
    rt["follow"] = False
    rt["last_play"] = t
    with _rt_lock:
        reqs, rt["requests"] = rt["requests"], []
    if cfg.get("connect"):
        if _ensure_connected(oid, cfg, rt):
            for kind, angles, dur in reqs:
                _move(oid, rt, angles, dur)
            if real_on and not rt["running"]:
                rt["cmd_end"] = -1.0
            if real_on:
                _drive_run(oid, rt, keys, t)
            _read(oid, rt)
    elif rt["connected"]:
        _disconnect(rt)
        rt["read_err"] = ""
        rt["read"] = None
        print("gimbal disconnected", oid)
    rt["running"] = real_on
    connected = cfg.get("connect") and rt["connected"]
    shown = rt["read"] if (connected and rt["read"] is not None) else g.get("target")
    _set_head(obj, shown)


def tick():
    objs = gimbal_objects()
    with _rt_lock:
        live = {str(o.get("id")) for o in objs}
        for oid in [k for k in _rt if k not in live]:
            _disconnect(_rt.pop(oid))
    if not objs:
        return
    clock = _clock()
    for obj in objs:
        try:
            _tick_one(obj, clock)
        except Exception as e:
            rt = _rt_for(obj.get("id"))
            if rt.get("_last_exc") != str(e):
                print("gimbal tick failed", obj.get("id"), e)
            rt["_last_exc"] = str(e)


def _loop():
    while True:
        t0 = time.time()
        try:
            tick()
        except Exception as e:
            print("gimbal loop", e)
        time.sleep(max(0.01, PERIOD_S - (time.time() - t0)))


def ensure_thread():
    global _thread
    if _thread is not None and _thread.is_alive():
        return
    _thread = threading.Thread(target=_loop, name="gimbal", daemon=True)
    _thread.start()


# ---------------------------------------------------------------------------
# Requests from the web server
# ---------------------------------------------------------------------------
def request_move(oid, angles=None, duration=None):
    obj = _find(oid)
    if obj is None:
        return False
    g = obj.setdefault("gimbal", clean_gimbal(None))
    if angles is not None:
        g["target"] = clamp_angles(angles)
    dur = _f(duration, 0.0) or _f((g.get("cfg") or {}).get("move_s"), 2.0)
    rt = _rt_for(oid)
    with _rt_lock:
        rt["requests"] = [("move", list(g["target"]), dur)]
    return True


def go_to_initial_pose():
    """Before a run (the arm moving to the path start): every gimbal with
    frames moves to the track's pose at t = 0."""
    n = 0
    for obj in gimbal_objects():
        ang = initial_angles(obj.get("id"))
        if ang is None:
            continue
        if request_move(obj.get("id"), ang):
            n += 1
    return n


def update(oid, data):
    """Settings / target from the Properties panel."""
    obj = _find(oid)
    if obj is None:
        return False
    g = clean_gimbal(obj.get("gimbal"))
    if isinstance(data.get("cfg"), dict):
        cfg = dict(g["cfg"])
        cfg.update({k: v for k, v in data["cfg"].items() if k in DEFAULT_CFG})
        g["cfg"] = cfg
    if data.get("target") is not None:
        g["target"] = data.get("target")
    obj["gimbal"] = clean_gimbal(g)
    if not obj["gimbal"]["cfg"]["connect"]:
        _rt_for(oid)["next_try"] = 0.0
    return True


def status(oid):
    """What the Properties panel shows."""
    obj = _find(oid)
    if obj is None:
        return None
    g = clean_gimbal(obj.get("gimbal"))
    rt = _rt_for(oid)
    cfg = g["cfg"]
    connected = bool(cfg["connect"] and rt["connected"])
    err = ""
    if cfg["connect"]:
        if not rt["connected"]:
            err = "Angle read failed: not connected" + (" (%s)" % rt["conn_err"] if rt["conn_err"] else "")
        elif not rt["read_ok"] and rt["read_err"]:
            err = "Angle read failed: %s" % rt["read_err"]
    return {
        "cfg": cfg,
        "target": g["target"],
        "script_path": resolve_script(cfg),
        "connected": connected,
        "read": rt["read"] if connected else None,
        "read_ok": bool(connected and rt["read_ok"]),
        "read_age": (time.time() - rt["read_at"]) if rt["read_at"] else None,
        "read_err": err,
        "cmd_err": rt["cmd_err"] if connected else "",
        "keys": keys_for(oid),
        "track_id": track_id(oid),
        "limits": LIMITS,
    }

