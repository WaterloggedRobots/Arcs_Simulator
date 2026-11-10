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
# Stored J1 / J2 / J3 (keys yaw / roll / pitch) are kept as read, in -180..180.
# The SDK's -56..146 "pitch" and +/-30 "roll" are limits for MOVE commands,
# not for the joints: an RS4 with the camera looking along the handle reads
# J3 = -90. (Clamping to them turned a copied J3 of -90 into -56.)
LIMITS = {"yaw": (-180.0, 180.0), "roll": (-180.0, 180.0), "pitch": (-180.0, 180.0)}

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
SIM_CURSOR_TARGET = "Sim cursor"
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
    if key == SIM_CURSOR_TARGET.lower():
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
    return None if out is None else clamp_angles(out)


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


MAX_SPIN = 10  # turns


def clean_spin(v):
    try:
        n = int(round(float(v or 0)))
    except Exception:
        n = 0
    return max(-MAX_SPIN, min(MAX_SPIN, n))


def _seg_yaw(a, b):
    """J1 change from frame a to frame b: straight from a's to b's angle, plus
    b's spin turns (360 deg each, + = rising J1)."""
    return (b["yaw"] - a["yaw"]) + 360.0 * clean_spin(b.get("spin"))


def spin_segment(keys, t):
    """(a, b) when t is inside a segment into a frame with spin turns, else None."""
    for a, b in zip(keys, keys[1:]):
        if a["t"] - 1e-6 <= t < b["t"] and clean_spin(b.get("spin")):
            return a, b
    return None


def clean_keys(keys):
    out = []
    for k in list(keys or []):
        if not isinstance(k, dict) or k.get("t") is None:
            continue
        y, r, p = clamp_angles([k.get("yaw"), k.get("roll"), k.get("pitch")])
        row = {"t": max(0.0, _f(k.get("t"))), "yaw": y, "roll": r, "pitch": p}
        spin = clean_spin(k.get("spin"))
        if spin:
            row["spin"] = spin  # extra J1 turns on the way into this frame
        out.append(row)
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
            out = [a[x] + (b[x] - a[x]) * u for x in AXES]
            if clean_spin(b.get("spin")):
                # Spinning: J1 goes round (stored values stay in +/-180).
                out[0] = ((a["yaw"] + _seg_yaw(a, b) * u + 180.0) % 360.0) - 180.0
            return out
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
    old = clean_keys(tr.get("keys"))
    same = next((k for k in old if abs(k["t"] - t) <= FRAME_EPS), None)
    keys = [k for k in old if abs(k["t"] - t) > FRAME_EPS]
    row = {"t": t, "yaw": ang[0], "roll": ang[1], "pitch": ang[2]}
    if same is not None and same.get("spin"):
        row["spin"] = same["spin"]  # a replaced frame keeps its spin
    keys.append(row)
    tr["keys"] = clean_keys(keys)
    _note_keys_changed(oid)
    return True


def set_spin(oid, index, spin):
    """Spin turns on the way into frame `index` (not the first frame)."""
    tr = _track(oid)
    if tr is None:
        return False
    keys = clean_keys(tr.get("keys"))
    i = int(_f(index, -1))
    if not 0 < i < len(keys):
        return False
    n = clean_spin(spin)
    if n:
        keys[i]["spin"] = n
    else:
        keys[i].pop("spin", None)
    tr["keys"] = keys
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
        try:
            root, nodes, handles = gmodel.build_twin(
                wp.server, "%s/%s" % (_mount_name(obj), part), model,
                SIM_RGB if part == "sim" else REAL_RGB, gmodel.joint_values(model, target),
                _clicker(obj["id"], part))
            obj[part + "_head"], obj[part + "_nodes"], obj[part + "_parts"] = root, nodes, handles
        except Exception as e:
            print("gimbal %s twin failed" % part, e)
            obj[part + "_head"], obj[part + "_nodes"], obj[part + "_parts"] = None, [], []
        obj.pop("_shown_" + part, None)
        obj.pop("_vis_" + part, None)
    obj["_model_used"] = model
    obj["_model_sig"] = _model_sig(obj)
    _set_visible(obj, "real", False)


def _model_sig(obj):
    cfg = ((obj or {}).get("gimbal") or {}).get("cfg") or {}
    _, _, mtime = gmodel.load(cfg.get("model"))
    return (gmodel.path_of(cfg.get("model")), mtime)


def twin_camera_world(obj, part="sim"):
    """World (pos, wxyz) of a twin's camera frame: z lens, x right, y image
    down (a Camera object, which looks along -z, sits on it turned 180 deg about x)."""
    model = obj.get("_model_used") or model_of(obj)[0]
    g = obj.get("gimbal") or {}
    angles = g.get("target") if part == "sim" else (_rt.get(str(obj.get("id"))) or {}).get("read")
    q, p = gmodel.camera_pose(model, gmodel.joint_values(model, angles or g.get("target")))
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


def drive_from_child(child):
    """A Camera (or other object) on a Sim gimbal turned in the world: solve
    J1-J3 so the gimbal turns it there (its local offset stays). True if handled."""
    pid = str((child or {}).get("parent_id") or "")
    base_id, part = split_part(pid)
    obj = _find(base_id) if part == "sim" else None
    if obj is None or _aiming(obj):
        return False
    lq = np.asarray(child.get("local_wxyz", [1, 0, 0, 0]), dtype=float)
    want = gmodel.qmul(np.asarray(child.get("wxyz", [1, 0, 0, 0]), dtype=float), gmodel.qinv(lq / (np.linalg.norm(lq) or 1.0)))
    solve_world(obj, want)
    _follow_now(obj)
    return True


def _follow_now(obj):
    """Show the new target at once (the gimbal thread would on its next tick)."""
    _set_twin(obj, "sim", (obj.get("gimbal") or {}).get("target"))
    _follow_sim(obj)


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
            go_to_initial_pose()  # back to the start for the next play
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
            on = not _solo["paused"]
            return _solo["t"], on and _solo["kind"] == "real", on and _solo["kind"] == "sim"
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


SPIN_LOOKAHEAD_S = 0.3
SPIN_MAX_DPS = 180.0


def _spin_end(rt):
    sp = rt.pop("spin", None)
    if sp is None:
        return
    try:
        rt["mod"].spin_end()
    except Exception as e:
        rt["cmd_err"] = "spin end failed: %s" % e
    rt["cmd_end"] = -1.0  # the next frame's move goes out at once


def _drive_spin(oid, rt, a, b, t):
    """Inside a segment with spin turns: turn J1 with speed commands (resent
    every tick), closing on where the track says J1 should be."""
    mod = rt["mod"]
    if not all(callable(getattr(mod, f, None)) for f in ("spin_begin", "spin_speed", "spin_end")):
        rt["cmd_err"] = "the driver script can't spin J1 (needs spin_begin / spin_speed / spin_end)"
        return
    j1 = rt["read"][0] if rt["read_ok"] and rt["read"] else None
    sp = rt.get("spin")
    if sp is None or sp.get("seg") != (a["t"], b["t"]):
        _spin_end(rt)
        try:
            mod.spin_begin()
        except Exception as e:
            rt["cmd_err"] = "spin start failed: %s" % e
            return
        start = a["yaw"] if j1 is None else j1
        sp = rt["spin"] = {"seg": (a["t"], b["t"]), "start": start, "last": start, "total": 0.0,
                           "sign": rt.get("spin_sign", 1.0), "cmd": 0.0,
                           "flipped": "spin_sign" in rt, "settle": time.time() + 0.3}
    if j1 is not None:
        sp["total"] += ((j1 - sp["last"] + 180.0) % 360.0) - 180.0
        sp["last"] = j1
        # J1 turning against the command: the command sign is the other way.
        # Once per spin, and not on readings from before a change (they lag a tick).
        if (not sp.get("flipped") and time.time() >= sp.get("settle", 0.0) and abs(sp["total"]) > 6
                and abs(sp["cmd"]) > 5 and (sp["total"] > 0) != (sp["cmd"] > 0)):
            sp["sign"] = -sp["sign"]
            sp["flipped"] = True
            rt["spin_sign"] = sp["sign"]
            sp["start"], sp["total"] = j1, 0.0
            sp["settle"] = time.time() + 0.5
    span = max(1e-3, b["t"] - a["t"])
    total = _seg_yaw(a, b)
    u = max(0.0, min(1.0, (t + SPIN_LOOKAHEAD_S - a["t"]) / span))
    want = a["yaw"] + total * u                    # unwrapped J1 the track wants soon
    have = sp["start"] + sp["total"]
    rate = max(-SPIN_MAX_DPS, min(SPIN_MAX_DPS, (want - have) / SPIN_LOOKAHEAD_S))
    try:
        mod.spin_speed(rate * sp["sign"])
        sp["cmd"] = rate
        rt["cmd_err"] = ""
    except Exception as e:
        rt["cmd_err"] = "spin speed failed: %s" % e


def _drive_run(oid, rt, keys, t):
    """During a real run: command the next frame, timed to arrive on it
    (spin segments turn J1 with speed commands instead)."""
    seg = spin_segment(keys, t) if keys else None
    if seg is not None:
        _drive_spin(oid, rt, seg[0], seg[1], t)
        return
    _spin_end(rt)
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
    # Target follows the track while it plays or when the playhead moves.
    moved = rt["last_play"] is None or abs(t - rt["last_play"]) > 1e-6
    if keys and (real_on or sim_on or moved or rt["follow"]):
        g["target"] = clamp_angles(angles_at(keys, t))
    rt["follow"] = False
    rt["last_play"] = t
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
    if calib == "cancel":
        rt.pop("calib_rec", None)
        rt["calib"] = {"ok": None, "time": time.time(), "text": "Recalibration cancelled."}
        calib = None
    if _ensure_connected(oid, cfg, rt):
        if calib == "start":
            _calib_start(rt)
        elif calib == "finish":
            _calib_finish(obj, cfg, rt)
        if cfg.get("drive", True):
            for kind, angles, dur in reqs:
                _move(oid, rt, angles, dur)
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
        if "calib_rec" in rt:
            _calib_sample(rt)
    elif calib or "calib_rec" in rt:
        rt.pop("calib_rec", None)
        rt["calib"] = {"ok": False, "time": time.time(),
                       "text": "Recalibrate failed: the real gimbal is not connected (%s)" % (rt.get("conn_err") or "no reply")}
    if not real_on and "spin" in rt and rt.get("mod") is not None:
        _spin_end(rt)  # run stopped or paused mid-spin
    rt["running"] = real_on
    _set_twin(obj, "sim", g.get("target"))
    ok = real_ok(oid)
    if ok:
        _set_twin(obj, "real", rt["read"])
    _set_visible(obj, "real", ok)


CALIB_MAX = 3000  # samples (5 min at 10 Hz)


def _calib_start(rt):
    """Recalibrate, step 1: record (joints, camera world angles) every tick
    while the handle is held still, hand end up, and the camera is moved."""
    if not callable(getattr(rt.get("mod"), "get_world", None)):
        rt["calib"] = {"ok": False, "time": time.time(),
                       "text": "Recalibrate failed: the driver script has no get_world() (camera world angles)"}
        return
    rt["calib_rec"] = {"samples": [], "start": time.time(), "err": ""}
    rt["calib"] = {"ok": None, "recording": True, "n": 0, "time": time.time(),
                   "text": "Recording: hold the handle still, hand end up, and move the camera around "
                           "with the joystick (pan all the way round, tilt up and down) for 10 s or more, then press Finish."}


def _calib_sample(rt):
    rec = rt["calib_rec"]
    if not rt["read_ok"]:
        return
    # Joints, then world, then joints again: the world reading is paired with
    # the mean of the two, and dropped if the camera moved much in between.
    try:
        world = [float(v) for v in list(rt["mod"].get_world())[:3]]
        after = [float(v) for v in list(rt["mod"].get_angles())[:3]]
    except Exception as e:
        rec["err"] = str(e) or type(e).__name__
        return
    if len(world) < 3 or not all(math.isfinite(v) for v in world + after):
        return
    before = list(rt["read"])
    step = [gmodel._wrap(b - a) for a, b in zip(before, after)]
    if max(abs(v) for v in step) > 3.0:
        rec["skipped"] = rec.get("skipped", 0) + 1
        return
    read = [a + d / 2 for a, d in zip(before, step)]
    rec.setdefault("raw", []).append({"t": round(time.time() - rec["start"], 3), "joints": before,
                                      "world": world, "joints_after": after})
    S = rec["samples"]
    # Keep a sample when the joints moved (a still gimbal adds nothing).
    if S and max(abs(gmodel._wrap(a - b)) for a, b in zip(read, S[-1][0])) < 0.5:
        return
    if len(S) < CALIB_MAX:
        S.append((read, world))
    J = np.array([x[0] for x in S])
    span = (J.max(axis=0) - J.min(axis=0)) if len(S) else [0, 0, 0]
    rt["calib"] = {"ok": None, "recording": True, "n": len(S), "time": time.time(),
                   "text": "Recording: %d samples, J1 span %.0f deg, J3 span %.0f deg%s. Keep the handle still; "
                           "press Finish when done (aim for J1 > 180 and J3 > 90)." % (
                               len(S), span[0], span[2], " (world read failed: %s)" % rec["err"] if rec["err"] else "")}


def _calib_finish(obj, cfg, rt):
    """Recalibrate, step 2: fit J2's axis and the J2 / J3 directions and zero
    offsets to the samples (gimbal_model.fit_calibration, on its own thread:
    it takes up to a minute) and write them into the model file."""
    rec = rt.pop("calib_rec", None)
    if rec is None:
        return
    samples = list(rec["samples"])
    model_file = cfg.get("model")
    # Every recording is kept, for checking the fit by hand.
    log_path = None
    try:
        import json
        d = gmodel.models_dir() / "calibrations"
        d.mkdir(parents=True, exist_ok=True)
        stem = os.path.splitext(os.path.basename(gmodel.path_of(model_file)))[0]
        log_path = str(d / ("%s_%s.json" % (stem, time.strftime("%Y%m%d-%H%M%S"))))
        rec_doc = {"model": os.path.basename(gmodel.path_of(model_file)), "samples": samples,
                   "raw": rec.get("raw", []), "skipped_moving": rec.get("skipped", 0)}
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(rec_doc, f)
    except Exception as e:
        print("gimbal calibrate: could not save the recording:", e)
        log_path = None
    rt["calib"] = {"ok": None, "fitting": True, "n": len(samples), "time": time.time(),
                   "text": "Fitting the model to %d samples (about a minute)…" % len(samples)}

    def work():
        import json
        try:
            path = gmodel.path_of(model_file)
            with open(path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            was = gmodel.parse(raw)["joints"]
            # Start from the model as it was before any recalibration, so a bad
            # result doesn't steer the next one. A file never recalibrated (new,
            # or restored / updated by hand) becomes the new .orig.
            import shutil
            if not any(j.get("calibrated") for j in raw.get("joints") or []) or not os.path.exists(path + ".orig"):
                shutil.copyfile(path, path + ".orig")
            with open(path + ".orig", "r", encoding="utf-8") as f:
                ojs = json.load(f).get("joints") or raw["joints"]
            # J3's direction isn't fitted: it is always the current file's (edited by hand).
            cur3 = next((j.get("sign", 1) for j in raw["joints"] if j.get("sdk") == "pitch"), 1)
            raw["joints"] = [dict(j, sign=cur3) if j.get("sdk") == "pitch" else j for j in ojs]
            fit = gmodel.fit_calibration(raw, samples)
            if log_path:
                with open(log_path, "r", encoding="utf-8") as f:
                    rec_doc = json.load(f)
                rec_doc["fit"] = fit
                with open(log_path, "w", encoding="utf-8") as f:
                    json.dump(rec_doc, f)
            where = " Recording: %s." % log_path if log_path else ""
            conv = "; best rms by attitude sign (pitch, roll): %s" % ", ".join(
                "%s %.2f" % kv for kv in sorted(fit["conventions"].items()))
            if fit["rms"] >= 4:
                rt["calib"] = {"ok": False, "time": time.time(), **fit, "text": (
                    "Recalibration not saved: the model doesn't fit the %d samples (rms %.2f deg, worst %.2f deg%s). "
                    "Was the handle held still and vertical, hand end up?%s" % (fit["samples"], fit["rms"], fit["worst"], conv, where))}
                print("gimbal calibrate:", rt["calib"]["text"])
                return
            note = ("%s: %d samples, hand end up; rms %.2f deg, worst %.2f deg; handle %.1f / %.1f deg off vertical" % (
                time.strftime("%Y-%m-%d %H:%M"), fit["samples"], fit["rms"], fit["worst"],
                fit["handle_tilt"][0], fit["handle_tilt"][1]))
            gmodel.write_joints(model_file, {
                "roll": {"axis": fit["axis"], "sign": fit["j2_sign"], "offset": fit["j2_offset"]},
                "pitch": {"sign": fit["j3_sign"], "offset": fit["j3_offset"]}}, note)
            quality = "good" if fit["rms"] < 1.5 else "fair"
            text = ("Recalibrated from %d samples: fit %s (rms %.2f deg, worst %.2f deg). J2 axis %s (tilt %.1f deg, side %.1f deg; "
                    "was %s), J2 %s offset %.1f (was %s %.1f), J3 %s offset %.1f (was %s %.1f); handle held %.1f / %.1f deg off "
                    "vertical. Saved to %s (previous in .bak, the original in .orig)%s.%s" % (
                        fit["samples"], quality, fit["rms"], fit["worst"], fit["axis"], fit["tilt"], fit["side"],
                        [round(float(v), 3) for v in was[1]["axis"]],
                        "+" if fit["j2_sign"] > 0 else "reversed", fit["j2_offset"],
                        "+" if was[1]["sign"] > 0 else "reversed", was[1]["offset"],
                        "+" if fit["j3_sign"] > 0 else "reversed", fit["j3_offset"],
                        "+" if was[2]["sign"] > 0 else "reversed", was[2]["offset"],
                        fit["handle_tilt"][0], fit["handle_tilt"][1], os.path.basename(path), conv, where))
            rt["calib"] = {"ok": True, "time": time.time(), **fit, "file": os.path.basename(path), "text": text}
            print("gimbal calibrate:", text)
        except Exception as e:
            rt["calib"] = {"ok": False, "time": time.time(), "text": "Recalibrate failed: %s%s" % (
                e, " (recording: %s)" % log_path if log_path else "")}
            print("gimbal calibrate failed:", e)

    threading.Thread(target=work, name="gimbal-calibrate", daemon=True).start()


def request_calibrate(oid, step="start"):
    """step: start (record samples), finish (fit and save) or cancel."""
    if _find(oid) is None or step not in ("start", "finish", "cancel"):
        return False
    rt = _rt_for(oid)
    if step == "start" and (rt.get("calib") or {}).get("fitting"):
        return False
    rt["calib_req"] = step
    if step == "start":
        rt["calib"] = {"ok": None, "recording": True, "n": 0, "time": time.time(), "text": "Starting…"}
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
            "joints": [{"name": j["name"], "sdk": j["sdk"], "range": j["range"]} for j in model["joints"]],
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
        "keys": keys_for(oid),
        "track_id": track_id(oid),
        "limits": LIMITS,
    }

