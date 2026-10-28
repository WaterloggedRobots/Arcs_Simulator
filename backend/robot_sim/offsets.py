"""Offset component: a yellow, waypoint-like frame fixed to a scene object or
the sim cursor, stored as a local transform relative to its owner's origin.

The marker is drawn only while its owner is selected. It never joins the
waypoint list or the path. Other components (Look At) can aim from it.

Data lives on the owner so undo snapshots and saves carry it:
  scene object -> obj["offset"]      sim cursor -> wp.sim_offset
  {"enabled": bool, "pos": [x, y, z], "wxyz": [w, x, y, z]}  (local frame)
"""
import functools
import threading

import numpy as np

import waypoints as wp

# Offset component switched off for now (user, 2026-10-27): no panel, no
# markers, can't be added. Records already in files are kept untouched.
OFFSETS_ENABLED = False
OFFSET_RGB = (255, 210, 40)
DEFAULT_POS = (0.0, 0.0, 0.1)

_viz = {}          # owner -> {"sphere", "arrow", "side", "gizmo", "world", "rot0"}
_edit_owner = None  # owner whose offset gizmo is showing (yellow sphere clicked)
_viz_seq = 0
# Request threads, the viewer tick and Viser callbacks all touch _viz; without
# one lock two threads could build the same marker and use a removed handle.
_lock = threading.RLock()


def _locked(fn):
    @functools.wraps(fn)
    def inner(*a, **k):
        with _lock:
            return fn(*a, **k)
    return inner


def _sv():
    import simulatorview
    return simulatorview


def _norm_owner(owner):
    o = str(owner or "")
    return "sim" if o in ("sim", "sim_cursor") else o


def default_offset():
    return {"enabled": True, "pos": list(DEFAULT_POS), "wxyz": [1.0, 0.0, 0.0, 0.0]}


def clean_offset(rec):
    """JSON-safe copy of an offset record, or None."""
    if not isinstance(rec, dict):
        return None
    pos = list(np.asarray(rec.get("pos") or [0, 0, 0], dtype=float).reshape(-1)[:3])
    q = np.asarray(rec.get("wxyz") or [1, 0, 0, 0], dtype=float).reshape(-1)[:4]
    n = float(np.linalg.norm(q)) or 1.0
    return {
        "enabled": bool(rec.get("enabled", True)),
        "pos": [float(v) for v in pos],
        "wxyz": [float(v) for v in (q / n)],
    }


def get_offset(owner):
    owner = _norm_owner(owner)
    if owner == "sim":
        return getattr(wp, "sim_offset", None)
    obj = _sv()._find_object(owner)
    return None if obj is None else obj.get("offset")


def _set_record(owner, rec):
    owner = _norm_owner(owner)
    if owner == "sim":
        wp.sim_offset = rec
        return True
    obj = _sv()._find_object(owner)
    if obj is None:
        return False
    obj["offset"] = rec
    return True


def offset_status(owner):
    """Props-panel payload: added / enabled / local pos / local wxyz."""
    rec = get_offset(owner)
    if not rec or not OFFSETS_ENABLED:
        return {"added": False, "enabled": False}
    c = clean_offset(rec)
    c["added"] = True
    return c


def _owner_pose(owner):
    owner = _norm_owner(owner)
    if owner == "sim":
        try:
            import robot_runtime as rr
            p, q = rr.get_sim_cursor_pose()
            if p is None or q is None:
                return None
            return np.asarray(p, dtype=float).reshape(3), np.asarray(q, dtype=float).reshape(4)
        except Exception:
            return None
    obj = _sv()._find_object(owner)
    if obj is None:
        return None
    return (np.asarray(obj.get("pos"), dtype=float).reshape(3),
            np.asarray(obj.get("wxyz", [1, 0, 0, 0]), dtype=float).reshape(4))


def compose(owner_pos, owner_wxyz, rec):
    """World pose of a local offset record on an owner pose."""
    sv = _sv()
    lp = np.asarray(rec.get("pos") or [0, 0, 0], dtype=float).reshape(3)
    lq = np.asarray(rec.get("wxyz") or [1, 0, 0, 0], dtype=float).reshape(4)
    oq = np.asarray(owner_wxyz, dtype=float).reshape(4)
    pos = np.asarray(owner_pos, dtype=float).reshape(3) + sv._qrot(oq, lp)
    q = sv._qmul(oq, lq)
    return pos, q / (float(np.linalg.norm(q)) or 1.0)


def offset_world(owner, require_enabled=True):
    if not OFFSETS_ENABLED:
        return None
    rec = get_offset(owner)
    if not rec or (require_enabled and not rec.get("enabled", True)):
        return None
    pose = _owner_pose(owner)
    if pose is None:
        return None
    return compose(pose[0], pose[1], rec)


@_locked
def add_offset_component(owner):
    owner = _norm_owner(owner)
    if not OFFSETS_ENABLED:
        return offset_status(owner)
    if get_offset(owner):
        return offset_status(owner)
    if owner != "sim":
        wp._push_history()
    if not _set_record(owner, default_offset()):
        return None
    _sv().note_component(owner, "offset")
    refresh_offset_viz()
    return offset_status(owner)


@_locked
def update_offset(owner, enabled=None, pos=None, wxyz=None, remove=False):
    owner = _norm_owner(owner)
    rec = get_offset(owner)
    if remove:
        if rec and owner != "sim":
            wp._push_history()
        _set_record(owner, None)
        _drop_viz(owner)
        refresh_offset_viz()
        return offset_status(owner)
    if not rec:
        return offset_status(owner)
    if owner != "sim":
        wp._push_history()
    rec = dict(rec)
    if enabled is not None:
        rec["enabled"] = enabled is True or str(enabled).lower() in ("1", "true", "on")
    if pos is not None:
        rec["pos"] = [float(v) for v in np.asarray(pos, dtype=float).reshape(-1)[:3]]
    if wxyz is not None:
        q = np.asarray(wxyz, dtype=float).reshape(-1)[:4]
        n = float(np.linalg.norm(q)) or 1.0
        rec["wxyz"] = [float(v) for v in q / n]
    _set_record(owner, clean_offset(rec))
    v = _viz.get(owner)
    if v is not None:
        v["world"] = None
    refresh_offset_viz()
    return offset_status(owner)


# ---------------------------------------------------------------------------
# Viser marker: yellow sphere + arrows + transform gizmo
# ---------------------------------------------------------------------------
def _node(owner):
    return "/offset_%s" % ("sim" if owner == "sim" else owner)


def _drop_viz(owner):
    global _edit_owner
    v = _viz.pop(owner, None)
    if _edit_owner == owner:
        _edit_owner = None
        _show_owner_gizmo(owner, True)
    if not v:
        return
    for key in ("gizmo", "sphere", "arrow", "side"):
        h = v.get(key)
        if h is not None:
            try:
                h.remove()
            except Exception:
                pass


@_locked
def clear_offset_viz():
    for owner in list(_viz.keys()):
        _drop_viz(owner)


def _selected_owners():
    out = []
    for kind, key in list(getattr(wp, "_selection", []) or []):
        if kind == "sim":
            out.append("sim")
        elif kind == "obj":
            out.append(str(key))
    return out


def _show_owner_gizmo(owner, on):
    try:
        if owner == "sim":
            import robot_runtime as rr
            g = getattr(rr, "_sim_gizmo", None)
        else:
            obj = _sv()._find_object(owner)
            g = None if obj is None else obj.get("gizmo")
        if g is not None:
            g.visible = bool(on)
    except Exception:
        pass


def _make_viz(owner, pos, q):
    global _viz_seq
    srv = wp.server
    if srv is None:
        return None
    # A fresh node name each time, so a new marker never replaces one that
    # is still being removed.
    _viz_seq += 1
    name = "%s_%d" % (_node(owner), _viz_seq)
    sphere = srv.scene.add_icosphere(
        name=name + "/sphere",
        radius=0.016,
        color=OFFSET_RGB,
        position=tuple(float(v) for v in pos),
        wxyz=tuple(float(v) for v in q),
    )
    arrow = wp.add_forward_arrow(name, colors=OFFSET_RGB, pos=pos, wxyz=q)
    side = wp.add_side_arrow(name, colors=OFFSET_RGB, pos=pos, wxyz=q)
    v = {"sphere": sphere, "arrow": arrow, "side": side, "gizmo": None, "world": None,
         "rot0": np.array(q, dtype=float), "node": name}
    _viz[owner] = v

    @sphere.on_click
    def _(_e=None):
        wp.note_scene_pick()
        begin_offset_edit(owner)

    return v


def _bind_gizmo(owner, v, pos, q):
    old = v.get("gizmo")
    if old is not None:
        try:
            old.remove()
        except Exception:
            pass
    giz = wp._new_transform_gizmo(
        "%s/gizmo_%s" % (v.get("node") or _node(owner), wp.gizmo_space), pos, wp._gizmo_display_wxyz(q), 0.18, False, True)
    v["gizmo"] = giz
    v["rot0"] = np.array(q, dtype=float)

    @giz.on_update
    @_locked
    def _(_):
        if wp._suspend_gizmo_cb:
            return
        rec = get_offset(owner)
        pose = _owner_pose(owner)
        if not rec or pose is None:
            return
        cur = v.get("world")
        old_q = np.asarray(cur[1] if cur else q, dtype=float)
        new_pos = np.array(giz.position, dtype=float).reshape(3)
        new_q = wp._keep_heading(old_q, wp._orientation_from_gizmo(giz, old_q, v.get("rot0", old_q)))
        if owner != "sim" and not wp._drag_checkpointed and not wp._suspend_history:
            wp._push_history()
            wp._drag_checkpointed = True
        sv = _sv()
        inv = sv._qinv(pose[1])
        lp = sv._qrot(inv, new_pos - pose[0])
        lq = sv._qmul(inv, new_q)
        lq = lq / (float(np.linalg.norm(lq)) or 1.0)
        rec["pos"] = [float(x) for x in lp]
        rec["wxyz"] = [float(x) for x in lq]
        v["world"] = (new_pos, new_q)
        _place(v, new_pos, new_q, gizmo=False)
        wp._arm_drag_release(giz)

    return giz


def _place(v, pos, q, gizmo=True):
    p = tuple(float(x) for x in np.asarray(pos).reshape(3))
    w = tuple(float(x) for x in np.asarray(q).reshape(4))
    for key in ("sphere", "arrow", "side"):
        h = v.get(key)
        if h is None:
            continue
        try:
            h.position = p
            h.wxyz = w
        except Exception:
            pass
    giz = v.get("gizmo")
    if gizmo and giz is not None:
        wp._suspend_gizmo_cb = True
        try:
            giz.position = p
            giz.wxyz = wp._gizmo_display_wxyz(w)
            v["rot0"] = np.array(w, dtype=float)
        except Exception:
            pass
        finally:
            wp._suspend_gizmo_cb = False


@_locked
def begin_offset_edit(owner):
    """Yellow sphere clicked: show its gizmo, park the owner's."""
    global _edit_owner
    owner = _norm_owner(owner)
    v = _viz.get(owner)
    if v is None:
        return False
    if _edit_owner and _edit_owner != owner:
        end_offset_edit()
    _edit_owner = owner
    pose = offset_world(owner)
    if pose is None:
        return False
    _bind_gizmo(owner, v, pose[0], pose[1])
    _show_owner_gizmo(owner, False)
    return True


@_locked
def end_offset_edit():
    global _edit_owner
    owner = _edit_owner
    _edit_owner = None
    if owner is None:
        return
    v = _viz.get(owner)
    if v and v.get("gizmo") is not None:
        try:
            v["gizmo"].remove()
        except Exception:
            pass
        v["gizmo"] = None
    if owner in _selected_owners():
        _show_owner_gizmo(owner, True)


@_locked
def on_selection_changed():
    end_offset_edit()
    refresh_offset_viz()


@_locked
def rebuild_offset_gizmos():
    """Gizmo space changed (local / global)."""
    if _edit_owner and _edit_owner in _viz:
        owner = _edit_owner
        pose = offset_world(owner)
        if pose is not None:
            _bind_gizmo(owner, _viz[owner], pose[0], pose[1])
            _show_owner_gizmo(owner, False)


@_locked
def refresh_offset_viz():
    """Draw markers for selected owners with an enabled offset; drop the rest."""
    if wp.server is None:
        return
    want = {}
    for owner in _selected_owners():
        pose = offset_world(owner)
        if pose is not None:
            want[owner] = pose
    for owner in list(_viz.keys()):
        if owner not in want:
            _drop_viz(owner)
    for owner, (pos, q) in want.items():
        v = _viz.get(owner)
        if v is None:
            v = _make_viz(owner, pos, q)
            if v is None:
                continue
        last = v.get("world")
        if last is not None and np.allclose(last[0], pos, atol=1e-6) and np.allclose(last[1], q, atol=1e-6):
            continue
        v["world"] = (np.array(pos, dtype=float), np.array(q, dtype=float))
        _place(v, pos, q)


@_locked
def tick_offsets():
    """Viewer-loop tick: keep markers on their owners as those move."""
    if not _viz and not _selected_owners():
        return
    refresh_offset_viz()
