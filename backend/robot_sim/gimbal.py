"""Gimbal objects: a DJI RS4 (or any gimbal with a driver script) in the scene.

A Gimbal is one scene object (kind "gimbal") with two twins under it, like the
Aubo Arm's Sim and Real cursors (list rows "<oid>#sim" and "<oid>#real"):

- the mount: the gimbal's HANDLE at the object's transform. The user moves
  it like any other object (gizmo, Properties, parenting): it stores the
  pose of the handle.
- the Sim gimbal: the gimbal's moving parts and camera in blue, posed by the
  target angles (the track).
- the Real gimbal: the same in orange, posed by the angles read from the
  gimbal.

Shape, joint axes and offsets come from a MODEL FILE (cfg["model"], in
user_files/gimbal_models/, read by gimbal_model.py); nothing about a
particular gimbal is in the code. It is shown only while reading the angles works. Reading is tried
  all the time (reconnecting every 2 s).

Angles are the gimbal's joint angles in degrees: J1 (handle motor), J2
(middle motor), J3 (camera motor); the SDK and the saved data call them
yaw, roll, pitch. They give the camera's pose relative to the handle through
the model's joint axes (the RS 4 file: handle along z, hand end up, camera
looking along +x at 0, J3 = -90 looking down the handle).

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
SIM_RGB = (80, 200, 255)    # as the Sim cursor
REAL_RGB = (255, 180, 80)   # as the Real cursor
PARTS = ("sim", "real")
AXES = ("yaw", "roll", "pitch")
# Stored J1 / J2 / J3 (keys yaw / roll / pitch) are joint readings (+/-180
# as read; frames may go further round, below). The SDK's -56..146 "pitch" and +/-30 "roll" are limits for MOVE commands,
# not for the joints: an RS4 with the camera looking along the handle reads
# J3 = -90. (Clamping to them turned a copied J3 of -90 into -56.)
# Joints go on past +/-180 on the timeline (J1 frames 0 -> 360 = one full
# turn; a frame's CW / CCW / Shortest direction picks the turn, also for J2 /
# J3); the gimbal itself reads them in +/-180 and moves are sent in +/-180.
YAW_TURNS_MAX = 20
_TURNS = (-360.0 * YAW_TURNS_MAX, 360.0 * YAW_TURNS_MAX)
LIMITS = {"yaw": _TURNS, "roll": _TURNS, "pitch": _TURNS}
# A frame's direction per joint (how the joint gets there from the frame
# before; the first frame's: from the real gimbal's reading, for the move to
# the initial pose): "" = as set (straight between the stored values).
FRAME_DIRS = ("", "short", "cw", "ccw")

DEFAULT_CFG = {
    "drive": True,             # send moves (Move, initial pose, Run); reading never stops
    "script": "",              # driver script; "" = the model's driver (rs4_can.py)
    "model": "",               # model file in user_files/gimbal_models; "" = dji_rs4.json
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
    cfg["drive"] = bool(cfg["drive"])
    cfg["script"] = str(cfg["script"] or "")
    cfg["model"] = str(cfg["model"] or "")
    cfg["interface"] = str(cfg["interface"] or "canalystii")
    cfg["channel"] = int(_f(cfg["channel"], 0))
    cfg["bitrate"] = int(_f(cfg["bitrate"], 1000000))
    cfg["move_s"] = max(0.1, min(MAX_MOVE_S, _f(cfg["move_s"], 2.0)))
    out = {"cfg": cfg, "target": clamp_angles(rec.get("target")), "look": clean_look(rec.get("look"))}
    col = clean_collider(rec.get("collider"))
    if col:
        out["collider"] = col
    return out


def clean_collider(rec):
    """The Sim gimbal's collider: {enabled, stop, margin} (shaped by the model's
    moving parts, not the handle), or None."""
    if not isinstance(rec, dict):
        return None
    return {"enabled": bool(rec.get("enabled", True)), "stop": bool(rec.get("stop", False)),
            "margin": max(0.0, _f(rec.get("margin"), 0.0))}


def clean_look(rec):
    """Look At on the Sim gimbal: {added, on, target (a name)}."""
    rec = rec if isinstance(rec, dict) else {}
    return {"added": bool(rec.get("added")), "on": bool(rec.get("on", True)),
            "target": str(rec.get("target") or "").strip()}


# ---------------------------------------------------------------------------
# Look At: the Sim gimbal aims its lens at a target
# ---------------------------------------------------------------------------
SIM_CURSOR_TARGET = "Sim Arm Cursor"
SIM_CURSOR_OLD_NAMES = ("sim cursor",)  # Look At targets saved before the rename
PRIMARY_TARGET = "Primary target"


def look_targets(oid):
    """Names the Look At target list offers: scene objects (not this gimbal),
    the Sim cursor and the camera scripts' primary target."""
    names = [str(o.get("name") or "") for o in list(wp.scene_objects or [])
             if str(o.get("id")) != str(oid) and o.get("name")]
    return names + [SIM_CURSOR_TARGET, PRIMARY_TARGET]


def look_target_pos(oid, name):
    """World position of a Look At target, or None. Read-only, safe on the gimbal thread."""
    key = str(name or "").strip().lower()
    if not key:
        return None
    if key == PRIMARY_TARGET.lower():
        try:
            p = wp.primary_target_pos()
            return None if p is None else np.asarray(p, dtype=float).reshape(3)
        except Exception:
            return None
    if key == SIM_CURSOR_TARGET.lower() or key in SIM_CURSOR_OLD_NAMES:
        try:
            import robot_runtime as rr
            with arms.LOCK:
                p, _ = rr.get_sim_cursor_pose()
            return None if p is None else np.asarray(p, dtype=float).reshape(3)
        except Exception:
            return None
    for o in list(wp.scene_objects or []):
        if str(o.get("id")) != str(oid) and str(o.get("name") or "").strip().lower() == key:
            return np.asarray(o.get("pos"), dtype=float).reshape(3)
    return None


def aim_angles(obj, world_pos, prev=None):
    """Stored [yaw, roll, pitch] pointing the lens at world_pos, turning only
    the model's aim joints (gimbal_model.aim). None if the target is at the camera."""
    model, _ = model_of(obj)
    out = gmodel.aim(model, obj.get("pos", [0, 0, 0]), obj.get("wxyz", [1, 0, 0, 0]), world_pos, prev)
    if out is None:
        return None
    out = list(out)
    out[0] = unwrap_yaw(out[0], (prev or [None])[0])
    return clamp_angles(out)


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


def numbered_name(i):
    """'Gimbal' for the first gimbal, then 'Gimbal 1', 'Gimbal 2' ... (as Aubo Arms)."""
    return "Gimbal" if i <= 0 else "Gimbal %d" % i


def _track_name(obj):
    """Track name by the gimbal's place among the gimbals in the scene."""
    ids = [str(o.get("id")) for o in gimbal_objects()]
    oid = str((obj or {}).get("id"))
    return numbered_name(ids.index(oid) if oid in ids else len(ids))


def default_name():
    """Name for a new gimbal object: the first free of Gimbal, Gimbal 1, ..."""
    used = {str(o.get("name") or "") for o in list(wp.scene_objects or [])}
    i = 0
    while numbered_name(i) in used:
        i += 1
    return numbered_name(i)


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


def unwrap_yaw(yaw, ref):
    """J1 angle equal to yaw (mod 360) nearest ref: readings and IK / Look At
    results are in +/-180, the track's J1 may be turns further on."""
    if ref is None:
        return _f(yaw)
    return _f(ref) + ((_f(yaw) - _f(ref) + 180.0) % 360.0) - 180.0


def clean_keys(keys):
    rows = [k for k in list(keys or []) if isinstance(k, dict) and k.get("t") is not None]
    rows.sort(key=lambda k: _f(k.get("t")))
    out = []
    turns = 0.0
    for k in rows:
        # Older files: "spin" = extra J1 turns into this frame -> J1 that much further on.
        try:
            turns += 360.0 * int(round(float(k.get("spin") or 0)))
        except Exception:
            pass
        y, r, p = clamp_angles([_f(k.get("yaw")) + turns, k.get("roll"), k.get("pitch")])
        row = {"t": max(0.0, _f(k.get("t"))), "yaw": y, "roll": r, "pitch": p}
        d = clean_dirs(k.get("dir"))
        if any(d):
            row["dir"] = d
        out.append(row)
    return out


def clean_dirs(d):
    """[J1, J2, J3] directions, each one of FRAME_DIRS."""
    d = list(d) if isinstance(d, (list, tuple)) else []
    d = (d + ["", "", ""])[:3]
    return [x if x in FRAME_DIRS else "" for x in (str(v or "") for v in d)]


def _turn_to(v, prev, d, rising_ccw):
    """The value equal to v (mod 360) reached from prev turning way d."""
    if d == "short":
        return prev + ((v - prev + 180.0) % 360.0) - 180.0
    if d in ("cw", "ccw"):
        up = (v - prev) % 360.0
        if up < 1e-6 or up > 360.0 - 1e-6:
            return prev  # already there: no full turn
        if (d == "ccw") == rising_ccw:
            return prev + up
        return prev - (360.0 - up)
    return v


def resolve_keys(obj, keys):
    """Frames with their directions applied, in order: a tagged joint takes
    the turn of its value its direction gives from the frame before; an
    untagged one keeps its step from the frame before (so it moves with it)."""
    rising = [rising_is_ccw(obj, i) for i in range(3)] if obj is not None else [True] * 3
    out, prev_old = [], None
    for k in keys:
        n = dict(k)
        if out:
            d = clean_dirs(k.get("dir"))
            for i, a in enumerate(AXES):
                if d[i]:
                    n[a] = _turn_to(k[a], out[-1][a], d[i], rising[i])
                else:
                    n[a] = k[a] + (out[-1][a] - prev_old[a])
        prev_old = k
        out.append(n)
    return out


def _set_keys(oid, tr, keys):
    tr["keys"] = clean_keys(resolve_keys(_find(oid), clean_keys(keys)))


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


def insert_frame(oid, t, angles=None, reading=False):
    """Frame at t with the given (or the target) angles; replaces one at t."""
    obj = _find(oid)
    if obj is None:
        return False
    g = obj.setdefault("gimbal", clean_gimbal(None))
    if angles is not None and reading:
        # A reading (+/-180): each joint nearest the track (or target) there, not turns back.
        here = angles_at(keys_for(oid), t) or g.get("target") or [None] * 3
        angles = [unwrap_yaw(a, h) for a, h in zip(list(angles)[:3], list(here)[:3])]
    ang = clamp_angles(angles if angles is not None else g.get("target"))
    g["target"] = ang
    t = max(0.0, _f(t))
    tr = _track(oid, create=True)
    old = clean_keys(tr.get("keys"))
    was = next((k for k in old if abs(k["t"] - t) <= FRAME_EPS), None)
    keys = [k for k in old if abs(k["t"] - t) > FRAME_EPS]
    row = {"t": t, "yaw": ang[0], "roll": ang[1], "pitch": ang[2]}
    if was is not None and was.get("dir"):
        row["dir"] = was["dir"]  # replacing a frame keeps its directions
    keys.append(row)
    _set_keys(oid, tr, keys)
    _note_keys_changed(oid)
    return True


def set_frame_dir(oid, dirs, index=None, t=None):
    """A frame's direction per joint (dirs: [J1, J2, J3], None keeps one)."""
    tr = _track(oid)
    if tr is None:
        return False
    keys = clean_keys(tr.get("keys"))
    i = int(_f(index, -1)) if index is not None else next(
        (j for j, k in enumerate(keys) if t is not None and abs(k["t"] - _f(t)) <= FRAME_EPS), -1)
    if not 0 <= i < len(keys):
        return False
    cur = clean_dirs(keys[i].get("dir"))
    new = list(dirs) if isinstance(dirs, (list, tuple)) else []
    new = (new + [None, None, None])[:3]
    keys[i]["dir"] = clean_dirs([c if n is None else n for c, n in zip(cur, new)])
    _set_keys(oid, tr, keys)
    _note_keys_changed(oid)
    return True


def edit_frame(oid, index, t=None, angles=None, dirs=None):
    """The timeline's Edit window: one frame's time, angles and directions
    (each left as it is when None). A frame already at the new time is replaced."""
    tr = _track(oid)
    if tr is None:
        return False
    keys = clean_keys(tr.get("keys"))
    i = int(_f(index, -1))
    if not 0 <= i < len(keys):
        return False
    k = dict(keys[i])
    if t is not None:
        k["t"] = max(0.0, _f(t))
    if angles is not None:
        a = list(angles) + [None] * 3
        cur = [k["yaw"], k["roll"], k["pitch"]]
        k["yaw"], k["roll"], k["pitch"] = clamp_angles([c if v is None else v for c, v in zip(cur, a[:3])])
    if dirs is not None:
        cur = clean_dirs(k.get("dir"))
        new = (list(dirs) + [None] * 3)[:3]
        k["dir"] = clean_dirs([c if n is None else n for c, n in zip(cur, new)])
    rest = [x for j, x in enumerate(keys) if j != i and abs(x["t"] - k["t"]) > FRAME_EPS]
    rest.append(k)
    _set_keys(oid, tr, rest)
    _note_keys_changed(oid)
    return True


def paste_frames(oid, frames):
    """The timeline's Paste: frames [{t, yaw, roll, pitch, dir?}] (absolute
    times), each replacing a frame at its time."""
    rows = clean_keys(frames)
    if not rows:
        return False
    tr = _track(oid, create=True)
    keys = [k for k in clean_keys(tr.get("keys"))
            if all(abs(k["t"] - r["t"]) > FRAME_EPS for r in rows)]
    _set_keys(oid, tr, keys + rows)
    _note_keys_changed(oid)
    return True


EXPORT_STEP_S = 0.1


def export_tracks(step=EXPORT_STEP_S):
    """For the Export file: each gimbal with frames, its frames (angles in
    degrees, J1-J3 as yaw / roll / pitch, unwrapped, with directions) and its
    pose every `step` s from 0 to the last frame (what Simulate plays)."""
    out = []
    for obj in gimbal_objects():
        oid = str(obj.get("id"))
        keys = keys_for(oid)
        if not keys:
            continue
        model, _ = model_of(obj)
        end = keys[-1]["t"]
        n = int(math.floor(end / step + 1e-9))
        ts = [round(i * step, 6) for i in range(n + 1)]
        if ts[-1] < end - 1e-9:
            ts.append(round(end, 6))
        poses = []
        for t in ts:
            a = angles_at(keys, t)
            poses.append({"t": t, "yaw": round(a[0], 3), "roll": round(a[1], 3), "pitch": round(a[2], 3)})
        out.append({"id": oid, "name": obj.get("name") or oid, "model": model.get("name"),
                    "joints": {"J1": "yaw", "J2": "roll", "J3": "pitch"}, "units": "deg",
                    "frames": keys, "poses": poses})
    return out


def angles_now(oid, t):
    """[J1, J2, J3] of a gimbal's track at time t (None without frames)."""
    a = angles_at(keys_for(oid), t)
    return None if a is None else [round(v, 3) for v in a]


def first_dirs(oid):
    """The first frame's directions: the move to the initial pose."""
    keys = keys_for(oid)
    return clean_dirs(keys[0].get("dir")) if keys else ["", "", ""]


# ---------------------------------------------------------------------------
# Speed check (run with each bake): every segment between frames, each joint's
# speed against the model's max_dps (J1 also against the run drive's own cap).
# ---------------------------------------------------------------------------
MIN_SEGMENT_S = 0.1  # the SDK's absolute move takes at least 0.1 s
_speed = {}          # track id -> {"segments": [...], "over": n, "time": t}


def speed_check():
    """Check every gimbal track; returns {track id: report} (kept for the page)."""
    out = {}
    for obj in gimbal_objects():
        oid = obj.get("id")
        keys = keys_for(oid)
        model, _ = model_of(obj)
        limits = {j["sdk"]: j.get("max_dps", 360.0) for j in model["joints"]}
        names = {j["sdk"]: j["name"] for j in model["joints"]}
        limits["yaw"] = min(limits.get("yaw", 360.0), TRACK_MAX_DPS)
        segs = []
        for a, b in zip(keys, keys[1:]):
            dt = b["t"] - a["t"]
            rows = []
            for k in AXES:
                d = abs(b[k] - a[k])  # J1 unwrapped: turns count
                dps = d / dt if dt > 1e-9 else (0.0 if d < 1e-6 else float("inf"))
                rows.append({"joint": names.get(k, k), "dps": round(dps, 1), "max": limits.get(k, 360.0),
                             "over": dps > limits.get(k, 360.0) + 1e-6})
            short = 0 < dt < MIN_SEGMENT_S and any(r["dps"] > 0 for r in rows)
            worst = max(rows, key=lambda r: r["dps"] / max(1e-9, r["max"]))
            segs.append({"t0": a["t"], "t1": b["t"], "joints": rows, "over": any(r["over"] for r in rows) or short,
                         "short": short, "worst": worst})
        rep = {"segments": segs, "over": sum(1 for s in segs if s["over"]), "time": time.time(),
               "name": obj.get("name") or oid}
        _speed[track_id(oid)] = rep
        out[track_id(oid)] = rep
    for tid in list(_speed):
        if tid not in out:
            _speed.pop(tid, None)
    return out


def tracks_baked():
    """Every gimbal track with frames to play has been checked by a bake since
    it last changed (edits drop the report: _note_keys_changed)."""
    for obj in gimbal_objects():
        if len(keys_for(obj.get("id"))) >= 2 and track_id(obj.get("id")) not in _speed:
            return False
    return True


def tracks_ok():
    """Baked and no segment too fast: what Upload / Run need."""
    return tracks_baked() and not any(r.get("over") for r in _speed.values())


def speed_status():
    """{track id: report} of the last check, for tracks unchanged since."""
    return dict(_speed)


def speed_summary():
    """One line for the bake hint, or ''."""
    if not tracks_baked():
        return "gimbal frames not baked: bake"
    bad = [(r["name"], s) for r in _speed.values() for s in r["segments"] if s["over"]]
    if not bad:
        return ""
    name, s = bad[0]
    w = s["worst"]
    why = ("shorter than %.1f s" % MIN_SEGMENT_S) if s["short"] and not w["over"] else (
        "%s %.0f°/s > %.0f°/s" % (w["joint"], w["dps"], w["max"]))
    return "gimbal too fast: %d segment%s (%s %.2f–%.2fs: %s)" % (
        len(bad), "" if len(bad) == 1 else "s", name, s["t0"], s["t1"], why)


def restore_track_keys(tracks):
    """Undo / redo: every gimbal's frames as in the snapshot {track id: keys}."""
    for obj in gimbal_objects():
        oid = obj.get("id")
        tr = _track(oid, create=True)
        keys = clean_keys(tracks.get(track_id(oid)) or [])
        if keys != clean_keys(tr.get("keys")):
            tr["keys"] = keys
            _note_keys_changed(oid)  # the target follows the restored frames


def delete_frame(oid, t=None, index=None, ts=None):
    """Drop the frame at index, at t, or at each time in ts (the timeline's Delete)."""
    tr = _track(oid)
    if tr is None:
        return False
    keys = clean_keys(tr.get("keys"))
    n = len(keys)
    if index is not None:
        i = int(_f(index, -1))
        keys = [k for j, k in enumerate(keys) if j != i]
    elif ts is not None:
        gone = [_f(x) for x in list(ts or [])]
        keys = [k for k in keys if all(abs(k["t"] - g) > FRAME_EPS for g in gone)]
    elif t is not None:
        keys = [k for k in keys if abs(k["t"] - _f(t)) > FRAME_EPS]
    _set_keys(oid, tr, keys)
    _note_keys_changed(oid)
    return len(keys) != n


def move_frames(oid, moves):
    """Retime frames (the timeline's drag): moves = [{t0, t}]. A frame left
    where a moved one lands is replaced by it. Returns the new times."""
    tr = _track(oid)
    if tr is None:
        return None
    keys = clean_keys(tr.get("keys"))
    moved, out = [], []
    for mv in list(moves or []):
        t0, t1 = _f((mv or {}).get("t0"), -1.0), max(0.0, _f((mv or {}).get("t")))
        k = next((k for k in keys if abs(k["t"] - t0) <= FRAME_EPS and k not in moved), None)
        if k is None:
            continue
        moved.append(k)
        out.append(t1)
    if not moved:
        return []
    stay = [k for k in keys if k not in moved
            and all(abs(k["t"] - t1) > FRAME_EPS for t1 in out)]
    for k, t1 in zip(moved, out):
        stay.append(dict(k, t=t1))
    _set_keys(oid, tr, stay)
    _note_keys_changed(oid)
    return out


def _note_keys_changed(oid):
    _speed.pop(track_id(oid), None)  # the bake's speed check no longer holds
    with _rt_lock:
        rt = _rt.get(str(oid))
        if rt is not None:
            rt["follow"] = True


# ---------------------------------------------------------------------------
# Scene: the gimbal drawn from its model file (gimbal_model.py)
# ---------------------------------------------------------------------------
import gimbal_model as gmodel  # noqa: E402

HANDLE_RGB = (150, 150, 158)


def _quat_axis(axis, deg):
    return gmodel.qaxis(axis, deg)


def model_of(obj):
    """(model, error) for a gimbal object, from cfg["model"] (a file name)."""
    cfg = ((obj or {}).get("gimbal") or {}).get("cfg") or {}
    m, err, _ = gmodel.load(cfg.get("model"))
    return m, err


def head_wxyz(angles, model=None):
    """Camera rotation relative to the handle for stored [yaw, roll, pitch]."""
    if model is None:
        model, _, _ = gmodel.load(None)
    q, _ = gmodel.camera_pose(model, gmodel.joint_values(model, angles))
    return q


def part_id(oid, part):
    return "%s#%s" % (oid, part)


def split_part(oid):
    """'o3#sim' -> ('o3', 'sim'); anything else -> (oid, None)."""
    base, _, part = str(oid or "").partition("#")
    return (base, part) if part in PARTS else (str(oid or ""), None)


def _clicker(oid, part=None):
    def _on_click(_e=None):
        import simulatorview as sv
        if part:
            sv.select_scene_object(part_id(oid, part))
        else:
            mode = "add" if wp._select_ctrl else ("range" if wp._select_shift else "replace")
            sv.select_scene_object(oid, mode=mode)
    return _on_click


def _mount_name(obj):
    return "/obj_%s/mesh" % obj["id"]


def spawn_mount_marker(obj, name=None):
    """The handle (the model's link-0 parts), under the mount node."""
    if wp.server is None:
        return
    model, _ = model_of(obj)
    old = obj.get("base_root")
    if old is not None:
        try:
            old.remove()
        except Exception:
            pass
    try:
        obj["base_root"] = gmodel.build_base(wp.server, (name or _mount_name(obj)) + "/handle", model,
                                             HANDLE_RGB, _clicker(obj["id"]))
    except Exception as e:
        obj["base_root"] = None
        print("gimbal handle failed", e)


def spawn_head(obj):
    """The Sim and Real gimbals (the model's moving links and camera), under
    the mount node. The Real one starts hidden."""
    if wp.server is None or obj.get("handle") is None:
        return
    model, _ = model_of(obj)
    target = (obj.get("gimbal") or {}).get("target")
    for part in PARTS:
        old = obj.get(part + "_head")
        if old is not None:
            try:
                old.remove()
            except Exception:
                pass
        # The Sim gimbal hangs under the mount node; the Real one is a node of
        # its own, placed every viewer tick by real_mirror (on the mount, or
        # on the real arm when the gimbal rides on a Sim Arm Cursor).
        node_name = "%s/%s" % (_mount_name(obj), part) if part == "sim" else "/gimbal_real_%s" % obj["id"]
        try:
            root, nodes, handles = gmodel.build_twin(
                wp.server, node_name, model,
                SIM_RGB if part == "sim" else REAL_RGB, gmodel.joint_values(model, target),
                _clicker(obj["id"], part))
            obj[part + "_head"], obj[part + "_nodes"], obj[part + "_parts"] = root, nodes, handles
        except Exception as e:
            print("gimbal %s twin failed" % part, e)
            obj[part + "_head"], obj[part + "_nodes"], obj[part + "_parts"] = None, [], []
        obj.pop("_shown_" + part, None)
        obj.pop("_vis_" + part, None)
    obj.pop("_real_root_set", None)
    place_real_root(obj, *(obj.get("_real_root") or (obj.get("pos", [0, 0, 0]), obj.get("wxyz", [1, 0, 0, 0]))))
    obj["_model_used"] = model
    obj["_model_sig"] = _model_sig(obj)
    _set_visible(obj, "real", False)


def _model_sig(obj):
    cfg = ((obj or {}).get("gimbal") or {}).get("cfg") or {}
    _, _, mtime = gmodel.load(cfg.get("model"))
    return (gmodel.path_of(cfg.get("model")), mtime)


def place_real_root(obj, pos, wxyz):
    """Where the Real gimbal's handle frame is in the world (the mount, or the
    mount mirrored onto the real arm: real_mirror.tick)."""
    pos = np.asarray(pos, dtype=float).reshape(3)
    q = np.asarray(wxyz, dtype=float).reshape(4)
    q = q / (np.linalg.norm(q) or 1.0)
    obj["_real_root"] = (pos, q)
    key = tuple(np.round(np.concatenate([pos, q]), 6))
    h = obj.get("real_head")
    if h is None or obj.get("_real_root_set") == key:
        return
    obj["_real_root_set"] = key
    try:
        h.position = tuple(float(v) for v in pos)
        h.wxyz = tuple(float(v) for v in q)
    except Exception:
        pass


def twin_camera_world(obj, part="sim"):
    """World (pos, wxyz) of a twin's camera frame: z lens, x right, y image
    down (a Camera object, which looks along -z, sits on it turned 180 deg about x)."""
    model = obj.get("_model_used") or model_of(obj)[0]
    g = obj.get("gimbal") or {}
    angles = g.get("target") if part == "sim" else (_rt.get(str(obj.get("id"))) or {}).get("read")
    q, p = gmodel.camera_pose(model, gmodel.joint_values(model, angles or g.get("target")))
    if part == "real" and obj.get("_real_root") is not None:
        rp, mq = obj["_real_root"]  # the Real gimbal may sit on the real arm
        pos = np.asarray(rp, dtype=float) + gmodel.qrot(mq, p)
        wq = gmodel.qmul(mq, q)
        return pos, wq / (np.linalg.norm(wq) or 1.0)
    mq = np.asarray(obj.get("wxyz", [1, 0, 0, 0]), dtype=float)
    pos = np.asarray(obj.get("pos", [0, 0, 0]), dtype=float) + gmodel.qrot(mq, p)
    wq = gmodel.qmul(mq, q)
    return pos, wq / (np.linalg.norm(wq) or 1.0)


def sim_collider_shapes(obj, margin=0.0):
    """World shapes of the Sim gimbal's moving parts (links 1..n, not the
    handle, which sits on the arm) for the collision check."""
    model = obj.get("_model_used") or model_of(obj)[0]
    g = obj.get("gimbal") or {}
    return gmodel.world_parts(model, gmodel.joint_values(model, g.get("target")),
                              obj.get("pos", [0, 0, 0]), obj.get("wxyz", [1, 0, 0, 0]), margin=margin)


def set_twin_colour(obj, part, rgb=None):
    """Tint a twin's parts (red on contact); rgb None = its own colour."""
    rgb = rgb or (SIM_RGB if part == "sim" else REAL_RGB)
    for h in obj.get(part + "_parts") or []:
        try:
            h.color = tuple(int(v) for v in rgb)
        except Exception:
            pass


def world_angles(obj, part="sim"):
    """World yaw / pitch / roll of a twin's camera (gimbal_model.world_ypr)."""
    _, q = twin_camera_world(obj, part)
    return [round(v, 2) for v in gmodel.world_ypr(q)]


def _aiming(obj):
    look = clean_look((obj.get("gimbal") or {}).get("look"))
    return look["added"] and look["on"]


def solve_world(obj, cam_world_wxyz):
    """Set the Sim gimbal's target so its camera frame turns to cam_world_wxyz
    (inverse kinematics from the current angles). Returns the error in degrees,
    or None when Look At drives the target."""
    if _aiming(obj):
        return None
    model = obj.get("_model_used") or model_of(obj)[0]
    g = obj.setdefault("gimbal", clean_gimbal(None))
    angles, err = gmodel.ik(model, obj.get("wxyz", [1, 0, 0, 0]), cam_world_wxyz, g.get("target"))
    angles = list(angles)
    angles[0] = unwrap_yaw(angles[0], (g.get("target") or [None])[0])
    g["target"] = clamp_angles(angles)
    g["_ik_err"] = round(float(err), 2)
    return err


def set_world_target(oid, ypr):
    """World-mode Target: camera yaw / pitch / roll in the world -> J1-J3."""
    obj = _find(oid)
    if obj is None:
        return None
    y, p, r = (list(ypr) + [0, 0, 0])[:3]
    err = solve_world(obj, gmodel.world_q(_f(y), _f(p), _f(r)))
    _follow_now(obj)
    return err


# The Sim Gimbal Cursor's gizmo: at its camera frame (the frame its children,
# the Camera, hang on), rotate only. Turning it turns the gimbal (J1-J3 solved);
# its children never drive the gimbal.
_twin_giz = {}  # oid -> gizmo handle


def _twin_gizmo_pose(obj):
    pos, q = twin_camera_world(obj, "sim")
    return tuple(float(v) for v in pos), tuple(float(v) for v in q)


def show_twin_gizmo(obj):
    if wp.server is None:
        return
    oid = str(obj.get("id"))
    pos, q = _twin_gizmo_pose(obj)
    giz = _twin_giz.get(oid)
    if giz is None:
        try:
            giz = wp.server.scene.add_transform_controls(
                "/gimbal_giz_%s" % oid, scale=0.18, disable_axes=True, disable_sliders=True,
                position=pos, wxyz=q, visible=True)
        except Exception as e:
            print("gimbal gizmo failed", e)
            return
        _twin_giz[oid] = giz

        @giz.on_update
        def _(_e=None, _oid=oid, _giz=giz):
            o = _find(_oid)
            if o is None or wp._suspend_gizmo_cb:
                return
            if _aiming(o):
                show_twin_gizmo(o)  # Look At aims it: snap back
                return
            if not wp._drag_checkpointed and not wp._suspend_history:
                wp._push_history()  # one undo step per drag
                wp._drag_checkpointed = True
            solve_world(o, np.asarray(_giz.wxyz, dtype=float))
            _follow_now(o)
            wp._arm_drag_release(_giz)
    try:
        giz.position = pos
        giz.wxyz = q
        giz.visible = True
    except Exception:
        pass


def place_twin_gizmo(obj):
    """Keep a shown gizmo on the camera frame (the target moved)."""
    giz = _twin_giz.get(str(obj.get("id")))
    if giz is None or not getattr(giz, "visible", False):
        return
    pos, q = _twin_gizmo_pose(obj)
    try:
        giz.position = pos
        if not wp._recent_gizmo_drag():
            giz.wxyz = q
    except Exception:
        pass


def drop_twin_gizmo(oid):
    giz = _twin_giz.pop(str(oid), None)
    if giz is not None:
        try:
            giz.remove()
        except Exception:
            pass


def hide_twin_gizmos():
    for giz in _twin_giz.values():
        try:
            giz.visible = False
        except Exception:
            pass


def _follow_now(obj):
    """Show the new target at once (the gimbal thread would on its next tick)."""
    _set_twin(obj, "sim", (obj.get("gimbal") or {}).get("target"))
    _follow_sim(obj)
    place_twin_gizmo(obj)


def add_gimbal_camera(obj):
    """A Camera object on a new gimbal's Sim gimbal, at its camera frame.
    (Called for new gimbals only: the Sim gimbal also takes the real pose.)"""
    _rt_for(str(obj.get("id")))["adopt_read"] = True
    import simulatorview as sv
    cam = sv.add_scene_object("camera", name="%s camera" % (obj.get("name") or "Gimbal"))
    if cam is None:
        return None
    cam["parent_id"] = part_id(obj["id"], "sim")
    cam["space"] = "local"
    cam["local_pos"] = np.zeros(3)
    # Camera objects look along their local -z with +y up (OpenGL); the
    # gimbal's camera frame looks along +z with y down: 180 deg about x.
    cam["local_wxyz"] = np.array([0.0, 1.0, 0.0, 0.0])
    sv._set_world_from_local(cam)
    sv._apply_object_xform(cam)
    sv._refresh_camera_viz(cam)
    try:
        sv.select_scene_object(obj["id"])
    except Exception:
        pass
    return cam


def _follow_sim(obj):
    """Children of the Sim gimbal (its Camera) follow the twin."""
    import simulatorview as sv
    sid = part_id(obj["id"], "sim")
    if any(str(o.get("parent_id") or "") == sid for o in list(wp.scene_objects or [])):
        sv._propagate_children(sid)


def rebuild(obj):
    """Redraw handle and twins (model file chosen or changed)."""
    spawn_mount_marker(obj)
    spawn_head(obj)
    _follow_sim(obj)


def scale_head(obj):
    """Twins keep the model's size; only the mount sphere scales."""
    return None


def _set_twin(obj, part, angles):
    nodes = obj.get(part + "_nodes")
    model = obj.get("_model_used")
    if not nodes or model is None or angles is None:
        return
    key = tuple(round(float(v), 3) for v in angles)
    if obj.get("_shown_" + part) == key:
        return
    obj["_shown_" + part] = key
    gmodel.pose_twin(model, nodes, gmodel.joint_values(model, angles))
    if part == "sim":
        _follow_sim(obj)
        place_twin_gizmo(obj)


def _user_shows(obj, part):
    """The cursor's eye in the object list (item_vis "obj:<oid>#sim" / "#real")."""
    try:
        return (wp.item_vis or {}).get("obj:%s" % part_id(obj.get("id"), part), True) is not False
    except Exception:
        return True


def _set_visible(obj, part, on):
    h = obj.get(part + "_head")
    if h is None or obj.get("_vis_" + part) == bool(on):
        return
    obj["_vis_" + part] = bool(on)
    try:
        h.visible = bool(on)
    except Exception:
        pass


# Which twin is selected: kept with its object id, so selecting anything else
# (selected_obj_id changes) or the mount itself drops it.
_part_sel = {"oid": None, "part": None}


def set_selected_part(oid, part):
    _part_sel["oid"] = str(oid) if part else None
    _part_sel["part"] = part if part in PARTS else None


def part_of(oid):
    if oid is None or _part_sel["oid"] != str(oid) or str(getattr(wp, "selected_obj_id", None)) != str(oid):
        return None
    return _part_sel["part"]


def selected_part():
    return part_of(getattr(wp, "selected_obj_id", None))


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
    """Driver script: cfg["script"], else the model's driver, else rs4_can.py."""
    path = str((cfg or {}).get("script") or "").strip()
    if not path:
        model, _, _ = gmodel.load((cfg or {}).get("model"))
        path = model.get("driver") or DEFAULT_SCRIPT
    moved = app_paths.relocate(path, app_paths.GIMBAL_SCRIPTS_DIR)
    if os.path.isfile(moved):
        return moved
    return path if os.path.isabs(path) else str(app_paths.GIMBAL_SCRIPTS_DIR / path)


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
    rt.pop("track", None)  # a run in progress starts its J1 tracking afresh


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


# ---------------------------------------------------------------------------
# Gimbal-only playback: a timeline with gimbal frames and no arm path plays on
# its own clock (Simulate; Upload = move the gimbals to their initial pose,
# then Run). The page sees it as the sim / real status (webui._merged_status).
# ---------------------------------------------------------------------------
_solo_lock = threading.RLock()
_solo = {"kind": None, "active": False, "paused": False, "t": 0.0, "t_end": 0.0, "wall": None,
         "plays": 1, "played": 0, "delay": 0.0, "uploaded": False, "ready_at": None, "restore": None}


def solo_t_end():
    """Length of the gimbal-only timeline: the last frame of a gimbal track
    with 2 frames or more, or None when there is nothing to play."""
    ends = []
    for obj in gimbal_objects():
        keys = keys_for(obj.get("id"))
        if len(keys) >= 2:
            ends.append(max(k["t"] for k in keys))
    return max(ends) if ends and max(ends) > 0 else None


def solo_playable():
    return solo_t_end() is not None


def _solo_tick_locked(now=None):
    """Advance the clock (looping / ending); call with _solo_lock held."""
    st = _solo
    if not st["active"] or st["paused"] or st["wall"] is None:
        return
    now = time.time() if now is None else now
    t = now - st["wall"]
    if t < st["t_end"]:
        st["t"] = max(0.0, t)
        return
    st["t"] = st["t_end"]
    if t < st["t_end"] + st["delay"]:
        return  # holding at the end between plays
    st["played"] += 1
    if st["plays"] < 0 or st["played"] < st["plays"]:
        st["wall"] = now
        st["t"] = 0.0
        if st["kind"] == "real":
            # Between plays the run drive is off (_clock): the gimbals go back
            # to the initial pose with an absolute move, then the next play starts.
            go_to_initial_pose()
            st["wall"] = now + max(_move_time_all(), 0.0)
        return
    st["active"] = False
    st["uploaded"] = False


def _move_time_all():
    ts = [_f(((o.get("gimbal") or {}).get("cfg") or {}).get("move_s"), 2.0) for o in gimbal_objects()]
    return max(ts) + 0.3 if ts else 0.0


def solo_active():
    with _solo_lock:
        _solo_tick_locked()
        return bool(_solo["active"])


def _plays(loops):
    try:
        n = int(loops)
    except Exception:
        n = 0
    return -1 if n < 0 else 1 + n


def solo_sim_start(loops=0, delay=0.0, restore=None):
    t_end = solo_t_end()
    if t_end is None:
        return False
    with _solo_lock:
        _solo.update(kind="sim", active=True, paused=False, t=0.0, t_end=t_end, wall=time.time(),
                     plays=_plays(loops), played=0, delay=max(0.0, _f(delay)), uploaded=False,
                     ready_at=None, restore=restore)
    return True


def solo_upload(loops=0, delay=0.0):
    """Real, step 1: the gimbals go to their initial pose; Run once there."""
    t_end = solo_t_end()
    if t_end is None or not any(real_ok(o.get("id")) for o in gimbal_objects()):
        return False
    go_to_initial_pose()
    with _solo_lock:
        _solo.update(kind="real", active=False, paused=False, t=0.0, t_end=t_end, wall=None,
                     plays=_plays(loops), played=0, delay=max(0.0, _f(delay)), uploaded=True,
                     ready_at=time.time() + _move_time_all(), restore=None)
    return True


def solo_real_start():
    with _solo_lock:
        if _solo["kind"] != "real" or not _solo["uploaded"] or _solo["active"]:
            return False
        if _solo["ready_at"] is not None and time.time() < _solo["ready_at"]:
            return False
        _solo.update(active=True, paused=False, t=0.0, wall=time.time())
    return True


def solo_pause():
    with _solo_lock:
        _solo_tick_locked()
        if not _solo["active"]:
            return None
        if _solo["paused"]:
            _solo["paused"] = False
            _solo["wall"] = time.time() - _solo["t"]
        else:
            _solo["paused"] = True
        return _solo["paused"]


def solo_stop():
    with _solo_lock:
        was = _solo["active"] or _solo["uploaded"]
        _solo.update(active=False, paused=False, uploaded=False, ready_at=None)
        return was


def solo_status(kind):
    """Like robot_runtime.sim_status / real_path_status, for the gimbal-only
    playback of that kind, or None when it isn't the one in use."""
    with _solo_lock:
        _solo_tick_locked()
        st = dict(_solo)
    if st["kind"] != kind or not (st["active"] or st["uploaded"]):
        return None
    out = {"active": st["active"], "paused": st["paused"], "t": st["t"], "t_end": st["t_end"],
           "gimbal_only": True}
    if kind == "sim":
        out.update(lin_mps=0.0, playhead_restore=st["restore"])
    else:
        moving = (not st["active"]) and st["ready_at"] is not None and time.time() < st["ready_at"]
        out.update(armed=st["active"], uploading=moving, uploaded=st["uploaded"] and not moving,
                   phase="gimbal moving to its initial pose" if moving else "", at_start=not moving,
                   start_q=None)
    return out


def _clock():
    """(timeline t, real run active, sim playing)."""
    import simulatorview as sv
    with _solo_lock:
        _solo_tick_locked()
        if _solo["active"]:
            # Not running while paused or while waiting to start the next play.
            on = not _solo["paused"] and time.time() >= (_solo["wall"] or 0.0)
            return _solo["t"], on and _solo["kind"] == "real", on and _solo["kind"] == "sim"
    real_t, real_on, sim_on = None, False, False
    try:
        for rp, pb in _run_states():
            if rp.get("active") and rp.get("armed") and not rp.get("paused"):
                real_on = True
                # The run's clock starts at the camera preroll; the arm's path
                # (and so the timeline) starts after it.
                real_t = _f(rp.get("t")) - _f(rp.get("preroll"))
            if pb.get("active") and not pb.get("paused"):
                sim_on = True
    except Exception:
        pass
    t = real_t if real_on else _f(sv._playhead_t())
    return t, real_on, sim_on


def _sim_playing():
    """A Simulate is on (paused included): an arm's playback or the gimbal-only one."""
    with _solo_lock:
        if _solo["active"] and _solo["kind"] == "sim":
            return True
    try:
        return any(pb.get("active") for _rp, pb in _run_states())
    except Exception:
        return False


RESTORE_HOLD_S = 0.5


# The SDK's absolute move is in camera ATTITUDE (angles against gravity, the
# gimbal holding its heading), not joints. The two only match with the handle
# upright: tilted (e.g. horizontal on the arm's flange) an absolute pitch made
# the RS 4 swing J1 to reach it. Past this tilt no absolute move is sent.
ATTITUDE_TILT_MAX = 20.0


def handle_tilt(obj):
    """Degrees between the real gimbal's hand end (handle +z) and world up:
    its mount on the real arm (real_mirror), else its own mount."""
    root = (obj or {}).get("_real_root")
    q = root[1] if root else (obj or {}).get("wxyz", [1, 0, 0, 0])
    z = gmodel.qrot(np.asarray(q, float), [0.0, 0.0, 1.0])
    return math.degrees(math.acos(max(-1.0, min(1.0, float(z[2])))))


def _move(oid, rt, angles, duration, yaw=True, roll=True):
    """Absolute move; yaw=False leaves J1 out (the run drive turns it by speed),
    roll=False leaves the roll axis out. Not sent with the handle tilted past
    ATTITUDE_TILT_MAX (it would go to the wrong pose): False, and said."""
    tilt = handle_tilt(_find(oid))
    if tilt > ATTITUDE_TILT_MAX:
        rt["cmd_err"] = ("handle %.0f° from upright: the RS 4's absolute moves are camera angles against gravity, "
                         "not joints, so they would turn the wrong joints (J1 swings); not sent" % tilt)
        return False
    mod = rt["mod"]
    w = lambda a: ((a + 180.0) % 360.0) - 180.0  # the gimbal takes +/-180
    y, r, p = clamp_angles(angles)
    y = w(y) if yaw else None
    r = w(r) if roll else None
    p = w(p)
    try:
        ok = mod.move_to(y, r, p, duration=max(0.1, min(MAX_MOVE_S, float(duration))))
        rt["last_cmd"] = [y, r, p]
        rt["cmd_err"] = "" if ok is not False else "gimbal refused the move"
        return ok is not False
    except Exception as e:
        rt["cmd_err"] = "move failed: %s" % e
        return False


TRACK_LOOKAHEAD_S = 0.3
TRACK_MAX_DPS = 180.0
TRACK_FUNCS = ("track_begin", "track_speed", "track_end")


def _can_track(rt):
    return all(callable(getattr(rt.get("mod"), f, None)) for f in TRACK_FUNCS)


def _can_j3(rt):
    """The driver can turn J3 alone during a run (track_j3: RS 4 incremental
    pitch in 3D ROLL 360 mode, which the axis probe showed moves J3 only)."""
    return callable(getattr(rt.get("mod"), "track_j3", None))


def _joint_mode(obj, rt):
    """Drive J1 and J3 as joints (J1 by speed, J3 by track_j3 steps, J2 held),
    with no absolute moves: the handle is tilted, so absolute moves (camera
    angles against gravity) would turn the wrong joints."""
    return handle_tilt(obj) > ATTITUDE_TILT_MAX and _can_track(rt) and _can_j3(rt)


J3_STEP_S = 0.4     # each J3 step aims this far ahead on the track
J3_MIN_DEG = 0.5    # smaller corrections are not sent
J3_MAX_DPS = 180.0
J3_SETTLE_S = 0.15  # after a step, before the next reading counts (reads come at 10 Hz)


def _track_j3(rt, keys, t):
    """J3 during a run in joint mode: a step from the J3 reading to the track
    J3_STEP_S ahead, sent when the last one is done (closed loop on the reading)."""
    if "track" not in rt or not (rt["read_ok"] and rt["read"]):
        return  # 3D ROLL 360 is set by _track_j1's track_begin first
    now = time.time()
    if now < rt.get("j3_busy", 0.0):
        return
    want = angles_at(keys, t + J3_STEP_S)[2]
    d = want - unwrap_yaw(rt["read"][2], want)
    if abs(d) < J3_MIN_DEG:
        return
    d = max(-STEP_MAX_DEG, min(STEP_MAX_DEG, d))
    dur = max(J3_STEP_S, abs(d) / J3_MAX_DPS)
    try:
        rt["mod"].track_j3(d, dur)
        rt["j3_busy"] = now + dur + J3_SETTLE_S  # the next step reads the finished one
    except Exception as e:
        rt["cmd_err"] = "J3 step failed: %s" % e
        rt["j3_busy"] = now + 0.5


def _track_end(rt):
    """End of a run (or pause): stop J1, hand control back, normal mode."""
    tk = rt.pop("track", None)
    if tk is None:
        return
    try:
        rt["mod"].track_end()
    except Exception as e:
        rt["cmd_err"] = "run end failed: %s" % e


def _track_j1(rt, keys, t):
    """J1 during a run: follow the track (any number of turns) with speed
    commands, resent every tick and closing on the unwrapped J1 reading."""
    mod = rt["mod"]
    j1 = rt["read"][0] if rt["read_ok"] and rt["read"] else None
    want_now = angles_at(keys, t)[0]
    tk = rt.get("track")
    if tk is None:
        try:
            mod.track_begin()
        except Exception as e:
            rt["cmd_err"] = "run start failed: %s" % e
            return
        start = want_now if j1 is None else unwrap_yaw(j1, want_now)
        tk = rt["track"] = {"have": start, "last": j1, "sign": rt.get("j1_sign", 1.0), "cmd": 0.0,
                            "moved": 0.0, "learnt": "j1_sign" in rt, "settle": time.time() + 0.3, "t": t}
    if t < tk["t"] - 0.5:
        # The timeline started again (Repeat): J1 counts from the nearest turn.
        tk["have"] = unwrap_yaw(tk["have"], want_now)
    tk["t"] = t
    if j1 is not None and tk["last"] is not None:
        d = ((j1 - tk["last"] + 180.0) % 360.0) - 180.0
        tk["have"] += d
        tk["moved"] += d
        # J1 turning against the command: the command is the other way round
        # (learnt once, not from readings older than the last change).
        if (not tk["learnt"] and time.time() >= tk["settle"] and abs(tk["moved"]) > 6
                and abs(tk["cmd"]) > 5 and (tk["moved"] > 0) != (tk["cmd"] > 0)):
            tk["sign"] = -tk["sign"]
            tk["learnt"] = True
            rt["j1_sign"] = tk["sign"]
            tk["settle"] = time.time() + 0.5
    if j1 is not None:
        tk["last"] = j1
    want = angles_at(keys, t + TRACK_LOOKAHEAD_S)[0]
    rate = max(-TRACK_MAX_DPS, min(TRACK_MAX_DPS, (want - tk["have"]) / TRACK_LOOKAHEAD_S))
    if abs(rate) < 0.2:
        rate = 0.0
    if (rate > 0) != (tk["cmd"] > 0) or not tk["cmd"]:
        tk["moved"] = 0.0
    try:
        mod.track_speed(rate * tk["sign"])
        tk["cmd"] = rate
    except Exception as e:
        rt["cmd_err"] = "J1 speed failed: %s" % e


def _drive_run(oid, rt, keys, t):
    """During a real run: J1 follows the track with speed commands (the
    driver's track_begin / track_speed / track_end; any number of turns), and
    J2 / J3 get the next frame, timed to arrive on it. A driver without them:
    absolute moves for all three (J1 then can't turn more than 180 deg per move)."""
    if not keys:
        return
    if _joint_mode(_find(oid), rt):
        # Handle tilted: J1 by speed, J3 by steps, nothing absolute (J2 holds).
        _track_j1(rt, keys, t)
        _track_j3(rt, keys, t)
        return
    # J1 by speed only when it turns in these frames: otherwise all three get
    # absolute moves (J2 too, which is left out while J1 turns by speed).
    tracking = _can_track(rt) and any(abs(b["yaw"] - a["yaw"]) > 0.5 for a, b in zip(keys, keys[1:]))
    if tracking:
        _track_j1(rt, keys, t)
    if t < rt["cmd_end"] - 1e-3:
        return
    nxt = next((k for k in keys if k["t"] > t + 0.05), None)
    if nxt is None:
        return  # past the last frame: hold
    tt = min(nxt["t"], t + MAX_MOVE_S)
    # An absolute move goes the short way: one is at most STEP_MAX_DEG for a
    # joint it drives, so a longer turn (a CW / CCW frame) goes the frames' way.
    now = angles_at(keys, t)
    goal = angles_at(keys, tt)
    driven = (2,) if tracking else (0, 1, 2)
    step = max(abs(goal[i] - now[i]) for i in driven)
    if step > STEP_MAX_DEG:
        tt = t + (tt - t) * STEP_MAX_DEG / step
        goal = angles_at(keys, tt)
    dur = max(0.1, round(tt - t, 1))
    if tracking:
        # The SDK's absolute move is in camera ATTITUDE angles, not joints: with
        # J1 turning the camera (a roll), an absolute roll pulls it back the
        # short way once it passes 180 deg. So while J1 turns by speed nothing
        # absolute goes out unless J3 changes, and then pitch only.
        if abs(goal[2] - now[2]) < 0.5:
            rt["cmd_end"] = tt
            return
        ok = _move(oid, rt, goal, dur, yaw=False, roll=False)
    else:
        ok = _move(oid, rt, goal, dur)
    if ok:
        rt["cmd_end"] = t + dur
    else:
        rt["cmd_end"] = t + 0.2  # retry soon


STEP_MAX_DEG = 90.0

LOOK_STEP_DEG = 0.3
LOOK_MOVE_S = 0.2


def _drive_look(oid, rt, angles):
    """During a run with Look At on: follow the aim live, in short moves."""
    last = rt.get("last_cmd")
    if last is not None and max(abs(a - b) for a, b in zip(angles, last)) < LOOK_STEP_DEG:
        return
    _move(oid, rt, angles, LOOK_MOVE_S)


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
    # Model file chosen or edited on disk: redraw.
    if wp.server is not None and obj.get("handle") is not None and obj.get("_model_sig") != _model_sig(obj):
        rebuild(obj)
    keys = keys_for(oid)
    # Simulate: the Sim gimbal's pose before it is put back when it ends, as the
    # arm's is (the playhead's jump back used to set it from the track there).
    playing = _sim_playing()
    if playing and not rt.get("sim_play"):
        rt["sim_play"] = True
        rt["pre_play"] = list(g.get("target") or [])
        rt.pop("restore_until", None)
    elif not playing and rt.get("sim_play"):
        rt["sim_play"] = False
        if rt.get("pre_play"):
            # Held a moment: the playhead is set back just after the playback ends.
            rt["restore_until"] = time.time() + RESTORE_HOLD_S
    # Target follows the track while it plays or when the playhead moves.
    moved = rt["last_play"] is None or abs(t - rt["last_play"]) > 1e-6
    if keys and (real_on or sim_on or moved or rt["follow"]):
        g["target"] = clamp_angles(angles_at(keys, t))
    rt["follow"] = False
    rt["last_play"] = t
    if rt.get("restore_until"):
        if time.time() < rt["restore_until"] and rt.get("pre_play"):
            g["target"] = clamp_angles(rt["pre_play"])
        else:
            rt.pop("restore_until", None)
            rt.pop("pre_play", None)
    # Look At overrides the track: the target aims at the Look At target, live.
    look = clean_look(g.get("look"))
    aiming = False
    rt["look_err"] = ""
    if look["added"] and look["on"]:
        tp = look_target_pos(oid, look["target"])
        aim = None if tp is None else aim_angles(obj, tp, g.get("target"))
        if aim is None:
            rt["look_err"] = "no valid look at target."
        else:
            g["target"] = aim
            aiming = True
    with _rt_lock:
        reqs, rt["requests"] = rt["requests"], []
    # Reading is always attempted; moves go out only with Drive on.
    calib = rt.pop("calib_req", None)
    if _ensure_connected(oid, cfg, rt):
        if not (real_on and not aiming) and "track" in rt and "jmove" not in rt:
            _track_end(rt)  # run stopped or paused: normal mode before any other move
        if cfg.get("drive", True):
            for kind, angles, dur, direction in reqs:
                if real_on:
                    continue
                turning = any(d in ("cw", "ccw") for d in direction)
                # Handle tilted: every move as joints (the timed move below), none absolute.
                if (turning or _joint_mode(obj, rt)) and rt["read_ok"]:
                    _jmove_start(rt, angles, dur, direction, obj)
                else:
                    rt.pop("jmove", None)
                    _track_end(rt)
                    _move(oid, rt, angles, dur)
                    if turning:
                        # Said, not hidden: the move went the short way instead.
                        rt["cmd_err"] = "CW / CCW needs a reading from the gimbal: moved the short way"
            if "jmove" in rt and not real_on:
                _jmove_tick(oid, rt)
            if real_on and not rt["running"]:
                rt["cmd_end"] = -1.0
            if real_on and aiming:
                _drive_look(oid, rt, g["target"])
            elif real_on:
                _drive_run(oid, rt, keys, t)
        _read(oid, rt)
        # A new gimbal starts where the real one is (first good reading), unless
        # its pose was set by hand, frames were added or Look At aims it.
        if rt.get("adopt_read") and rt["read_ok"]:
            rt["adopt_read"] = False
            if not keys_for(oid) and not aiming:
                g["target"] = clamp_angles(rt["read"])
        # A snap of sim to real that came before this gimbal read: its first good reading.
        if rt.get("snap_until") and rt["read_ok"]:
            if time.time() < rt.pop("snap_until") and not aiming:
                _snap_one(obj, rt)
        if calib:
            _calibrate_home(obj, cfg, rt)  # after the read: the reading is fresh
    elif calib:
        rt["calib"] = {"ok": False, "time": time.time(),
                       "text": "Recalibrate failed: the real gimbal is not connected (%s)" % (rt.get("conn_err") or "no reply")}
    rt["running"] = real_on
    _set_twin(obj, "sim", g.get("target"))
    ok = real_ok(oid)
    if ok:
        _set_twin(obj, "real", rt["read"])
    # The Real gimbal is its own node: it also hides with its mount.
    _set_visible(obj, "real", ok and _user_shows(obj, "real") and obj.get("visible", True) is not False)
    _set_visible(obj, "sim", _user_shows(obj, "sim"))


SDK_ORDER = ("yaw", "roll", "pitch")


def _calibrate_home(obj, cfg, rt):
    """Recalibrate: the gimbal is in its home pose (handle vertical, hand end
    up, camera looking at +x, level, no roll); its reading becomes the model's
    all-zero pose (each joint's offset; signs and axes stay). No move is sent."""
    try:
        if not rt["read_ok"] or not rt["read"]:
            raise ValueError("the real gimbal is not reading")
        reading = [float(v) for v in rt["read"]]
        world = None
        if callable(getattr(rt["mod"], "get_world", None)):
            try:
                world = [float(v) for v in list(rt["mod"].get_world())[:3]]
            except Exception:
                world = None
        model, err, _ = gmodel.load(cfg.get("model"))
        if err:
            raise ValueError("the model file can't be read: %s" % err)
        was = {j["sdk"]: j["offset"] for j in model["joints"]}
        # Home = the camera looking along WORLD +x, level, with the handle where
        # the gimbal object is in the scene (unrotated: the all-zero pose).
        offs, ik_err = gmodel.home_offsets(model, reading, obj.get("wxyz", [1, 0, 0, 0]))
        if ik_err > 2.0:
            raise ValueError("with the gimbal object turned as it is in the scene, the model can't look along "
                             "world +x level (%.1f deg off): set its rotation to match the real handle" % ik_err)
        note = "%s: home pose (camera at +x, level) read J %.1f / %.1f / %.1f%s" % (
            time.strftime("%Y-%m-%d %H:%M"), reading[0], reading[1], reading[2],
            "" if world is None else ", world roll %.1f pitch %.1f" % (world[1], world[2]))
        path = gmodel.write_joints(cfg.get("model"), {k: {"offset": v} for k, v in offs.items()}, note)
        names = {j["sdk"]: j["name"] for j in model["joints"]}
        text = "Recalibrated to the home pose: read %s; offsets %s (were %s). Saved to %s (previous in .bak, the original in .orig)." % (
            " / ".join("%s %.1f" % (names.get(k, k), reading[i]) for i, k in enumerate(SDK_ORDER)),
            " / ".join("%s %.1f" % (names.get(k, k), offs[k]) for k in SDK_ORDER if k in offs),
            " / ".join("%.1f" % was[k] for k in SDK_ORDER if k in was), os.path.basename(path))
        if world is not None and (abs(world[1]) > 3 or abs(world[2]) > 3):
            text += (" The gimbal reads roll %.1f, pitch %.1f against gravity: the camera wasn't level, so the "
                     "Sim gimbal will be off by that much; level it and recalibrate." % (world[1], world[2]))
        g = obj.get("gimbal") or {}
        if not _aiming(obj):
            g["target"] = clamp_angles(reading)  # the Sim gimbal at the home pose too (frames follow when the playhead moves)
        rt["calib"] = {"ok": True, "time": time.time(), "reading": reading, "world": world, "offsets": offs, "text": text}
        print("gimbal calibrate:", text)
    except Exception as e:
        rt["calib"] = {"ok": False, "time": time.time(), "text": "Recalibrate failed: %s" % e}
        print("gimbal calibrate failed:", e)


def request_calibrate(oid):
    if _find(oid) is None:
        return False
    rt = _rt_for(oid)
    rt["calib_req"] = True
    rt["calib"] = {"ok": None, "time": time.time(), "text": "Reading the home pose…"}
    return True


def real_ok(oid):
    """The last angle read worked: the Real gimbal is shown."""
    rt = _rt.get(str(oid))
    return bool(rt and rt["connected"] and rt["read_ok"] and rt["read"] is not None)


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
MOVE_DIRS = ("short", "cw", "ccw")
JMOVE_DPS = 60.0  # J1 speed of a CW / CCW move (at least Move time)


def move_dirs(direction):
    """[J1, J2, J3] move directions (short / cw / ccw) from a list, or from
    one word (older pages: J1's, J2 / J3 the short way)."""
    if isinstance(direction, (list, tuple)):
        d = [str(v or "short") for v in list(direction)[:3]] + ["short"] * 3
    else:
        d = [str(direction or "short"), "short", "short"]
    return [x if x in MOVE_DIRS else "short" for x in d[:3]]


def request_move(oid, angles=None, duration=None, direction="short"):
    """Move the real gimbal to the target, each joint its way (move_dirs):
    all Shortest = one absolute move; a CW / CCW joint = a timed move from the
    reading (_jmove_start) that turns it that way round: J1 by speed (any
    amount, as the run drive does), J3 in absolute steps of at most
    STEP_MAX_DEG. J2 gets no absolute move while J1 turns by speed (the
    attitude problem), so its direction only counts on the timeline."""
    obj = _find(oid)
    if obj is None:
        return False
    g = obj.setdefault("gimbal", clean_gimbal(None))
    if angles is not None:
        g["target"] = clamp_angles(angles)
    dur = _f(duration, 0.0) or _f((g.get("cfg") or {}).get("move_s"), 2.0)
    direction = move_dirs(direction)
    rt = _rt_for(oid)
    with _rt_lock:
        rt["requests"] = [("move", list(g["target"]), dur, direction)]
    return True


def _joint(obj, i):
    """The model joint driven by SDK key AXES[i]."""
    model = obj.get("_model_used") or model_of(obj)[0]
    return next((j for j in model["joints"] if j["sdk"] == AXES[i]), model["joints"][min(i, len(model["joints"]) - 1)])


def rising_is_ccw(obj, i=0):
    """Does a rising reading of joint i (AXES order) turn it anticlockwise as
    seen looking along the model's view for it (J1: from the hand end, down
    the handle)? A turn about +axis looks anticlockwise from where the axis
    points, i.e. looking along -axis. (RS 4 J1: axis -z, sign +1 -> clockwise.)"""
    j = _joint(obj, i)
    view = np.asarray(j.get("view", [0.0, 0.0, -1.0]), float)
    return float(np.dot(np.asarray(j["axis"], float), -view)) * float(j.get("sign", 1.0)) > 0


def _jmove_start(rt, angles, dur, direction, obj=None):
    """A timed move from the reading to the target, each joint its way
    (move_dirs: short / cw / ccw, seen as the model's view for that joint),
    driven like a run (_drive_run)."""
    r = list(rt["read"])
    dirs = move_dirs(direction)
    end = []
    for i in range(3):
        rising = rising_is_ccw(obj, i) if obj is not None else True
        e = _turn_to(_f(angles[i]), r[i], dirs[i], rising)
        if dirs[i] != "short" and abs(e - r[i]) > 359.5:
            e = r[i]  # past it by under half a degree: no full turn
        end.append(e)
    delta = max(abs(e - a) for e, a in zip(end, r))
    dur = max(_f(dur, 2.0), delta / JMOVE_DPS)
    _track_end(rt)
    rt["jmove"] = {"t0": time.time(), "dur": dur, "keys": [
        {"t": 0.0, "yaw": r[0], "roll": r[1], "pitch": r[2]},
        {"t": dur, "yaw": end[0], "roll": end[1], "pitch": end[2]}]}
    rt["cmd_end"] = -1.0


def _jmove_tick(oid, rt):
    jm = rt["jmove"]
    t = time.time() - jm["t0"]
    if t > jm["dur"] + 1.0:
        rt.pop("jmove", None)
        _track_end(rt)
        return
    _drive_run(oid, rt, jm["keys"], min(t, jm["dur"]))


SNAP_WAIT_S = 30.0


def _snap_one(obj, rt):
    read = list(rt["read"])
    g = obj.setdefault("gimbal", clean_gimbal(None))
    read[0] = unwrap_yaw(read[0], (g.get("target") or [None])[0])
    g["target"] = clamp_angles(read)


def snap_to_real():
    """Snap sim to real (the arm's): every gimbal whose angles read takes them
    as its Sim target (J1 to the turn nearest the target). A gimbal not reading
    yet (the arm's snap when the robot connects comes before the gimbal's first
    reading) takes its first good reading in the next SNAP_WAIT_S. Returns how
    many snapped now."""
    n = 0
    for obj in gimbal_objects():
        oid = obj.get("id")
        if _aiming(obj):
            continue
        rt = _rt_for(oid)
        if not real_ok(oid):
            rt["snap_until"] = time.time() + SNAP_WAIT_S
            continue
        rt.pop("snap_until", None)
        _snap_one(obj, rt)
        _follow_now(obj)
        n += 1
    return n


def go_to_initial_pose():
    """Before a run (the arm moving to the path start): every gimbal with
    frames moves to the track's pose at t = 0."""
    n = 0
    for obj in gimbal_objects():
        ang = initial_angles(obj.get("id"))
        if ang is None:
            continue
        # The first frame's directions: from the reading to the starting pose.
        d = [x or "short" for x in first_dirs(obj.get("id"))]
        if request_move(obj.get("id"), ang, direction=d):
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
        _rt_for(oid)["adopt_read"] = False  # set by hand: don't take the real pose
    if isinstance(data.get("look"), dict):
        look = dict(g.get("look") or {})
        look.update(data["look"])
        g["look"] = look
    obj["gimbal"] = clean_gimbal(g)
    if isinstance(data.get("cfg"), dict) and set(data["cfg"]) & {"script", "interface", "channel", "bitrate"}:
        _rt_for(oid)["next_try"] = 0.0  # try the new settings now
    return True


def _model_status(obj):
    cfg = ((obj or {}).get("gimbal") or {}).get("cfg") or {}
    model, err, _ = gmodel.load(cfg.get("model"))
    return {"file": cfg.get("model") or gmodel.DEFAULT_FILE, "name": model["name"], "error": err,
            "joints": [{"name": j["name"], "sdk": j["sdk"], "range": j["range"], "view": j.get("view_name") or ""}
                       for j in model["joints"]],
            "folder": str(gmodel.models_dir())}


def status(oid):
    """What the Properties panel shows."""
    obj = _find(oid)
    if obj is None:
        return None
    g = clean_gimbal(obj.get("gimbal"))
    rt = _rt_for(oid)
    cfg = g["cfg"]
    connected = bool(rt["connected"])
    err = ""
    if not rt["connected"]:
        err = "Angle read failed: not connected" + (" (%s)" % rt["conn_err"] if rt["conn_err"] else "")
    elif not rt["read_ok"]:
        err = "Angle read failed: %s" % (rt["read_err"] or "no angles yet")
    return {
        "cfg": cfg,
        "target": g["target"],
        "script_path": resolve_script(cfg),
        "model": _model_status(obj),
        "world": world_angles(obj, "sim"),
        "world_real": world_angles(obj, "real") if real_ok(oid) else None,
        "ik_err": (obj.get("gimbal") or {}).get("_ik_err"),
        "calib": rt.get("calib"),
        "models": gmodel.list_models(),
        "connected": connected,
        "read": rt["read"] if real_ok(oid) else None,
        "read_ok": real_ok(oid),
        "part": part_of(oid),
        "look": clean_look(g.get("look")),
        "look_err": rt.get("look_err", ""),
        "look_targets": look_targets(oid),
        "read_age": (time.time() - rt["read_at"]) if rt["read_at"] else None,
        "read_err": err,
        "cmd_err": rt["cmd_err"] if connected else "",
        # Handle tilted: J1 / J3 driven as joints (no absolute moves), J2 held.
        "tilt": round(handle_tilt(obj), 1),
        "joint_drive": bool(connected and _joint_mode(obj, rt)),
        "keys": keys_for(oid),
        "track_id": track_id(oid),
        "limits": LIMITS,
    }

