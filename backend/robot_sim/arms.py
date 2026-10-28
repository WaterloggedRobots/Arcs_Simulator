"""Several Aubo Arms in one scene.

Each arm has its own robots (real, sim, hidden IK model), base pose, cursors,
saved points, waypoints and bake. The rest of the code keeps that state in
module globals (robot_runtime, waypoints, path_bake, simulatorview), written
for one arm. Rather than rewrite all of it, every arm keeps its own copy of
those globals (PER_ARM below) and the copy of the arm being worked on is
swapped into the modules:

- activate(aid): the arm the UI edits. Selecting one of its cursors, its base,
  its waypoints or its saved points, or picking it in Robot Settings, makes
  it active. The waypoint and saved point lists show the active arm.
- using(aid): run some code as that arm for a moment (ticks, playback, file
  save/load, viewer clicks), then swap back.

Everything that reads or writes arm state holds LOCK, so two threads never
see a half-swapped arm. Scene objects, saved paths, selection of objects and
timeline extras stay shared by all arms (world coordinates). Saved points
are shared too, but relative to the base: each arm draws its own copy.

The default arm keeps the old ids ("sim_cursor", "sim_bot", "real_cursor")
and scene node names, so files and code written for one arm still work.
Other arms add "@<arm id>" to those ids and "_<arm id>" to node names.
"""
import contextlib
import copy
import sys
import threading

LOCK = threading.RLock()
DEFAULT_ID = "arm0"
DEFAULT_NAME = "Aubo Arm"
BUILTIN_IDS = ("sim_cursor", "sim_bot", "real_cursor")

PER_ARM = {
    "robot_runtime": [
        "robot_actual", "robot_sim", "robot_ik", "robot",
        "_MODEL_ID", "URDF_PATH", "URDF_SIM_PATH", "_robot_addr",
        "AUBO_I5_MAX_VEL_DEG", "AUBO_I5_MAX_ACCEL_DEG",
        "_base_pos", "_base_wxyz",
        "_cursors_ready", "_cursor_sim", "_arrow_sim", "_arrow_sim_side",
        "_cursor_real", "_arrow_real", "_arrow_real_side",
        "_sim_gizmo", "_sim_rot0", "_sim_selected", "_sim_wxyz", "_last_good_q",
        "_sim_ik_pending", "_track", "_hold_sim", "_start_snap_done",
        "_hide_real_robot", "sim_mode", "sim_path",
        "_hold_move", "_hold_target_ik", "_hold_target_q", "_hold_arrived", "_hold_mode",
        "_pb", "_rp", "_rp_gen", "_last_vis", "_motion", "_speed_log",
        "_burn_pts", "_burn_line", "_real_link",
    ],
    "waypoints": [
        "waypoints", "_pose_markers",
        "sim_camera", "sim_offset", "sim_collider", "sim_comp_order", "look_bars",
        "path_handles", "selected_idx", "selected_ctrl", "selected_wp_idxs",
        "selected_pose_ids", "_dotted_handles", "_pending_rebuild_idx",
        "_gizmo_dirty", "_drag_checkpointed", "undo_stack", "redo_stack",
        "_path_window", "_path_win_t",
    ],
    "path_bake": [
        "segments", "sample_handles", "baked_hops", "baked_points", "_speed_valid",
        "custom_curves", "_tl_cache", "_dots_high", "selected_sample", "import_raw_json",
    ],
    "simulatorview": ["_sim_bot_viz", "_sim_bot_giz", "_arm_move"],
}

_defaults = {}
_arms = {DEFAULT_ID: {"id": DEFAULT_ID, "name": DEFAULT_NAME, "state": None}}
_order = [DEFAULT_ID]
_active = DEFAULT_ID
_loaded = DEFAULT_ID
_seq = 0
# using() nesting: an activate() inside it keeps the new arm loaded at exit.
_depth = 0
_sticky = False


def _mod(name):
    return sys.modules.get(name)


def capture_defaults():
    """Fresh-arm values, read before the first arm builds its robots."""
    if _defaults:
        return
    for mname, names in PER_ARM.items():
        m = _mod(mname)
        if m is None:
            continue
        row = {}
        for n in names:
            if not hasattr(m, n):
                continue
            try:
                row[n] = copy.deepcopy(getattr(m, n))
            except Exception:
                row[n] = None
        _defaults[mname] = row


def _fresh_state():
    return {m: {n: copy.deepcopy(v) for n, v in row.items()} for m, row in _defaults.items()}


def _store(aid):
    state = {}
    for mname, names in PER_ARM.items():
        m = _mod(mname)
        if m is None:
            continue
        state[mname] = {n: getattr(m, n) for n in names if hasattr(m, n)}
    _arms[aid]["state"] = state


def _restore(aid):
    state = _arms[aid].get("state") or {}
    for mname, row in state.items():
        m = _mod(mname)
        if m is None:
            continue
        for n, v in row.items():
            setattr(m, n, v)


def _swap(aid):
    global _loaded
    if aid == _loaded or aid not in _arms:
        return
    _store(_loaded)
    _restore(aid)
    _loaded = aid


@contextlib.contextmanager
def using(aid):
    """Run as arm `aid`, then go back to the arm that was loaded before."""
    global _depth, _sticky
    with LOCK:
        if aid is None or aid not in _arms:
            aid = _loaded
        prev = _loaded
        _depth += 1
        _swap(aid)
        try:
            yield aid
        finally:
            _depth -= 1
            if _sticky:
                if _depth == 0:
                    _sticky = False
            elif prev in _arms:
                _swap(prev)


def each():
    """Arm ids in list order, for code that runs as every arm in turn."""
    with LOCK:
        return list(_order)


def current():
    return _loaded


def active():
    return _active


def is_default():
    return _loaded == DEFAULT_ID


def exists(aid):
    return aid in _arms


def name_of(aid=None):
    rec = _arms.get(aid or _loaded)
    return rec["name"] if rec else ""


def rename(aid, name):
    with LOCK:
        if aid in _arms and str(name or "").strip():
            _arms[aid]["name"] = str(name).strip()
            return True
    return False


def ns(aid=None):
    """Suffix for scene node names: '' for the default arm."""
    aid = aid or _loaded
    return "" if aid == DEFAULT_ID else "_" + aid


def cid(base, aid=None):
    """Scene id of an arm's built-in item (sim_cursor, sim_bot, real_cursor)."""
    aid = aid or _loaded
    return base if aid == DEFAULT_ID else "%s@%s" % (base, aid)


def split_id(oid):
    """'sim_cursor@arm2' -> ('sim_cursor', 'arm2'); 'sim_cursor' -> ('sim_cursor', 'arm0').
    Anything that is not an arm's built-in item -> (oid, None)."""
    s = str(oid or "")
    base, _, aid = s.partition("@")
    if base not in BUILTIN_IDS:
        return s, None
    if not aid:
        return base, DEFAULT_ID
    return base, (aid if aid in _arms else None)


def is_builtin(oid):
    return split_id(oid)[1] is not None


def _leave_active():
    """Before another arm takes over: drop this arm's gizmos and selection."""
    import waypoints as wp
    try:
        wp.deselect_all()
    except Exception as e:
        print("arms: deselect failed", e)
    try:
        import robot_runtime as rr
        rr.hide_sim_gizmo()
    except Exception:
        pass
    try:
        wp.hide_sim_bot_gizmo()
    except Exception:
        pass


def activate(aid):
    """Make `aid` the arm the UI edits. Returns True when it changed."""
    global _active, _sticky
    with LOCK:
        if aid not in _arms or aid == _active:
            if aid in _arms and _loaded != aid:
                _swap(aid)
                if _depth:
                    _sticky = True
            return False
        _swap(_active)
        _leave_active()
        _swap(aid)
        _active = aid
        if _depth:
            _sticky = True
        print("arms: active", aid, _arms[aid]["name"])
    return True


def ensure_active_loaded():
    with LOCK:
        _swap(_active)


def next_name():
    used = {r["name"] for r in _arms.values()}
    n = 1
    while "%s %d" % (DEFAULT_NAME, n) in used:
        n += 1
    return "%s %d" % (DEFAULT_NAME, n)


def add_arm(name=None, aid=None, pos=None, wxyz=None, model=None, ip=None, port=None, activate_it=True):
    """New Aubo Arm with its own robots and cursors. Sim only until it has an IP."""
    global _seq
    import robot_runtime as rr
    with LOCK:
        if aid is None or aid in _arms:
            while True:
                _seq += 1
                aid = "arm%d" % _seq
                if aid not in _arms:
                    break
        else:
            try:
                _seq = max(_seq, int(str(aid).replace("arm", "")))
            except Exception:
                pass
        rec = {"id": aid, "name": str(name or next_name()), "state": _fresh_state()}
        _arms[aid] = rec
        _order.append(aid)
        if pos is None:
            pos = _free_spot()
        with using(aid):
            try:
                rr.setup_extra_arm(pos=pos, wxyz=wxyz, model=model, ip=ip, port=port)
            except Exception as e:
                import traceback
                traceback.print_exc()
                print("arms: add failed", e)
        print("arms: added", aid, rec["name"], "at", pos)
    if activate_it:
        activate(aid)
    return aid


def _free_spot():
    """Next to the last arm, so a new one does not stand inside another."""
    import numpy as np
    import robot_runtime as rr
    xs = []
    for aid in list(_order):
        if aid == _loaded:
            p, _ = rr.arm_base_pose()
        else:
            st = (_arms[aid].get("state") or {}).get("robot_runtime") or {}
            p = st.get("_base_pos")
        if p is not None:
            xs.append(np.asarray(p, dtype=float).reshape(3))
    if not xs:
        return [0.0, 1.0, 0.0]
    y = max(float(p[1]) for p in xs) + 1.0
    return [0.0, y, 0.0]


def remove_arm(aid):
    """Delete an extra arm and everything it drew. The default arm stays."""
    import robot_runtime as rr
    if aid == DEFAULT_ID or aid not in _arms:
        return False
    with LOCK:
        if _active == aid:
            activate(DEFAULT_ID)
        with using(aid):
            try:
                rr.teardown_arm()
            except Exception as e:
                print("arms: teardown failed", e)
        # Children of this arm's cursors and base stay, now in the world.
        try:
            import waypoints as wp
            ids = {cid(b, aid) for b in BUILTIN_IDS}
            for obj in list(getattr(wp, "scene_objects", None) or []):
                if str(obj.get("parent_id") or "") in ids:
                    wp.reparent_object(obj.get("id"), None)
        except Exception as e:
            print("arms: unparent failed", e)
        del _arms[aid]
        _order.remove(aid)
    return True


def remove_extra_arms():
    for aid in [a for a in _order if a != DEFAULT_ID]:
        remove_arm(aid)


def listing():
    """For /api/status: every arm, the active one flagged."""
    import robot_runtime as rr
    out = []
    with LOCK:
        for aid in list(_order):
            with using(aid):
                try:
                    connected = bool(rr.robot_connected())
                except Exception:
                    connected = False
                out.append({
                    "id": aid,
                    "name": _arms[aid]["name"],
                    "active": aid == _active,
                    "default": aid == DEFAULT_ID,
                    "sim_id": cid("sim_cursor", aid),
                    "bot_id": cid("sim_bot", aid),
                    "real_id": cid("real_cursor", aid),
                    "real_connected": connected,
                })
    return out


# Viser runs click and gizmo callbacks on its own threads. Each callback runs
# under LOCK, as the arm that made the node: clicking another arm's cursor,
# base, waypoint or saved point makes that arm active first. Scene objects
# (shared by all arms) and camera / measure nodes run as the active arm.
_SHARED_NODES = ("/obj_", "/measure", "/cam_track", "/grid")


def _wrap_callback(handle, func):
    if getattr(func, "_arm_wrapped", False):
        return func
    owner = _loaded
    name = str(getattr(handle, "name", "") or "")
    owned = not name.startswith(_SHARED_NODES)

    def run(*a, **k):
        with LOCK:
            if owner not in _arms:
                return None
            ensure_active_loaded()
            if owned and owner != _active:
                activate(owner)
            return func(*a, **k)
    run._arm_wrapped = True
    run.__name__ = getattr(func, "__name__", "callback")
    return run


def _install_viser_hooks():
    try:
        from viser import _scene_handles as sh
    except Exception as e:
        print("arms: viser hooks not installed", e)
        return

    def patch(cls, meth):
        orig = getattr(cls, meth, None)
        if orig is None or getattr(orig, "_arm_patched", False):
            return

        def wrapped(self, func=None, *a, **k):
            if func is None or not callable(func):
                # Decorator with options: @h.on_click(modifier=...)
                inner = orig(self, func, *a, **k) if func is not None else orig(self, *a, **k)
                if callable(inner):
                    return lambda f: inner(_wrap_callback(self, f))
                return inner
            return orig(self, _wrap_callback(self, func), *a, **k)
        wrapped._arm_patched = True
        setattr(cls, meth, wrapped)

    patch(sh._RaycastSupportedSceneNodeHandle, "on_click")
    for meth in ("on_update", "on_drag_start", "on_drag_end"):
        patch(sh.TransformControlsHandle, meth)


_install_viser_hooks()
