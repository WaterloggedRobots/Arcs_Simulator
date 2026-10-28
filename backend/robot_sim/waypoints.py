# version: beta.0904
import copy
import json
import time
import threading
import numpy as np

import arms


def _pb_mod():
    import path_bake as m
    return m


def _bake_call(name, *args, **kwargs):
    fn = getattr(_pb_mod(), name, None)
    if callable(fn):
        return fn(*args, **kwargs)
    return None

server = None

waypoints = []
scene_objects = []
selected_obj_id = None
selected_obj_ids = []
selected_wp_idxs = []
selected_pose_ids = []
selection_kind = None
grid_spacing = 0.01
_select_ctrl = False
_select_shift = False
_selection = []  # ("sim", None) | ("obj", id) | ("wp", idx)
_measure_line = None
_obj_seq = 0
extra_tracks = []
sim_poses = []
_sim_pose_seq = 0
saved_paths = []
_saved_path_seq = 0
selected_path_id = None
_pose_markers = {}
show_pose_markers = True
sim_camera = None
sim_offset = None  # sim cursor Offset component (see offsets.py)
sim_collider = None  # sim cursor Collider component (see colliders.py)
sim_comp_order = []  # sim cursor components, oldest first
look_bars = []  # Look At timeline bars [{t0, t1, name}], rebuilt on bake
list_vis = {"obj": True, "wp": True, "pose": True, "track": True}
snap_to_grid = False
point_order = []
point_sort = "time"
BUILTIN_SIM_POSES = (
    {"id": "real", "name": "Real pose", "builtin": True, "store": "global", "added_at": 0.0},
    {"id": "home", "name": "Home pose", "builtin": True, "store": "global", "added_at": 0.0,
     "q_deg": [0.0, 0.0, 90.0, 0.0, 90.0, 0.0]},
)
path_handles = []
selected_idx = None
selected_ctrl = None
current_mode = "arc"
_line_id = 0
_path_gen = 0

undo_stack = []
redo_stack = []
_MAX = 80
_suspend_history = False
_drag_checkpointed = False
_gizmo_last_move = 0.0
_gizmo_dirty = False
_pending_rebuild_idx = None

CURVE_CTRL_MAX = 0.12
OFFSET_FROM_LAST = np.array([0.08, 0.0, 0.0], dtype=np.float32)

# True: several waypoints may share the same keyframe time.
# False: adding at an existing time moves that waypoint instead.
allow_multiple_waypoints_per_keyframe = False

# "local" = gizmo axes follow the object. "global" = world XYZ.
gizmo_space = "local"
WORLD_WXYZ = (1.0, 0.0, 0.0, 0.0)
_suspend_gizmo_cb = False


def init(viser_server):
    global server
    server = viser_server
    _bind_empty_click(viser_server)


# Click on empty space in the world view deselects everything. Viser fires
# the scene-wide click after any per-node on_click, so a node pick inside
# the same click window means the click hit something.
_last_pick_ts = 0.0
EMPTY_CLICK_WINDOW_S = 0.35


def note_scene_pick():
    global _last_pick_ts
    _last_pick_ts = time.time()


def _bind_empty_click(srv):
    scene = getattr(srv, "scene", None)
    on_click = getattr(scene, "on_click", None)
    if not callable(on_click):
        return  # older viser: no scene-wide click event

    def _on_scene_click(_event=None):
        t_click = time.time()

        def _later():
            time.sleep(EMPTY_CLICK_WINDOW_S)
            if _last_pick_ts >= t_click - EMPTY_CLICK_WINDOW_S:
                return
            if _select_ctrl or _select_shift:
                return
            try:
                deselect_all()
            except Exception as e:
                print("empty-click deselect failed:", e)

        threading.Thread(target=_later, daemon=True).start()

    try:
        on_click()(_on_scene_click)
    except Exception as e:
        print("scene click bind failed:", e)


def _try_set_space(handle, space):
    if handle is None:
        return
    space = "local" if space == "local" else "world"
    for attr in ("space", "control_space", "origin"):
        if hasattr(handle, attr):
            try:
                setattr(handle, attr, space)
            except Exception:
                pass
    if space == "world":
        try:
            handle.wxyz = WORLD_WXYZ
        except Exception:
            pass


def _quat_mul_wxyz(a, b):
    a = np.asarray(a, dtype=float).reshape(4)
    b = np.asarray(b, dtype=float).reshape(4)
    wa, xa, ya, za = a
    wb, xb, yb, zb = b
    return np.array([
        wa * wb - xa * xb - ya * yb - za * zb,
        wa * xb + xa * wb + ya * zb - za * yb,
        wa * yb - xa * zb + ya * wb + za * xb,
        wa * zb + xa * yb - ya * xb + za * wb,
    ], dtype=float)


def _quat_almost_I(q, eps=1e-3):
    q = np.asarray(q, dtype=float).reshape(4)
    if q[0] < 0:
        q = -q
    return abs(q[0] - 1.0) < eps and np.linalg.norm(q[1:]) < eps


def _gizmo_display_wxyz(obj_wxyz):
    if gizmo_space == "local":
        return tuple(float(v) for v in np.asarray(obj_wxyz, dtype=float).reshape(4))
    return WORLD_WXYZ


def _keep_heading(old_q, new_q):
    """Never replace a real heading with identity (gizmo snap)."""
    new_q = np.asarray(new_q, dtype=float).reshape(4)
    old_q = np.asarray(old_q, dtype=float).reshape(4)
    n = float(np.linalg.norm(new_q)) or 1.0
    new_q = new_q / n
    if _quat_almost_I(new_q) and not _quat_almost_I(old_q):
        return old_q
    return new_q


def _orientation_from_gizmo(gizmo, obj_wxyz, start_wxyz):
    """Local: gizmo pose is the object. Global: gizmo rotation is a world-frame delta."""
    gw = np.asarray(gizmo.wxyz, dtype=float).reshape(4)
    if gizmo_space == "local":
        return gw
    if _quat_almost_I(gw, eps=2e-4):
        return np.asarray(obj_wxyz, dtype=float).reshape(4)
    return _quat_mul_wxyz(gw, start_wxyz)


def _new_transform_gizmo(name, pos, display_wxyz, scale, disable_rotations, visible):
    pos = tuple(float(v) for v in np.asarray(pos, dtype=float).reshape(3))
    kwargs = dict(
        name=name,
        position=pos,
        wxyz=display_wxyz,
        scale=scale,
        active_axes=(True, True, True),
        disable_rotations=disable_rotations,
        visible=visible,
    )
    try:
        h = server.scene.add_transform_controls(
            **kwargs, space=("local" if gizmo_space == "local" else "world")
        )
    except TypeError:
        h = server.scene.add_transform_controls(**kwargs)
    _try_set_space(h, "local" if gizmo_space == "local" else "world")
    try:
        h.wxyz = display_wxyz
    except Exception:
        pass
    return h


def _bind_waypoint_gizmo(idx):
    wp = waypoints[idx]
    old = wp.get("gizmo")
    vis = False
    if old is not None:
        vis = bool(getattr(old, "visible", False))
        try:
            old.remove()
        except Exception:
            pass
    display = _gizmo_display_wxyz(wp.get("wxyz", WORLD_WXYZ))
    name = f"{_wp_node(wp)}/gizmo_{gizmo_space}"
    gizmo = _new_transform_gizmo(name, wp["pos"], display, 0.18, False, vis)
    wp["gizmo"] = gizmo

    @gizmo.on_update
    def _(_):
        global _drag_checkpointed
        if _suspend_gizmo_cb:
            return
        item = wp
        i = _index_of(item)
        if i is None:
            return
        new_pos = np.array(gizmo.position)
        start = item.get("_rot0", item["wxyz"])
        new_wxyz = _keep_heading(
            item["wxyz"],
            _orientation_from_gizmo(gizmo, item["wxyz"], start),
        )
        if np.allclose(new_pos, item["pos"], atol=1e-8) and np.allclose(
            np.asarray(new_wxyz, dtype=float),
            np.asarray(item["wxyz"], dtype=float),
            atol=1e-8,
        ):
            return
        if not _suspend_history and not _drag_checkpointed:
            _push_history()
            _drag_checkpointed = True
            item["_rot0"] = np.array(item["wxyz"], dtype=float)
        delta = new_pos - item["pos"]
        item["pos"] = new_pos
        item["wxyz"] = new_wxyz
        item["q_dirty"] = True
        for key in ("sphere", "arrow", "side", "halo"):
            h = item.get(key)
            if h is None:
                continue
            h.position = new_pos
            if hasattr(h, "wxyz"):
                h.wxyz = new_wxyz
        for which in ("out", "in"):
            key = f"{which}_ctrl"
            key_pos = f"{which}_ctrl_pos"
            if item.get(key) is not None:
                item[key_pos] = item[key_pos] + delta
                item[key].position = item[key_pos]
                cg = getattr(item[key], "_gizmo", None)
                if cg is not None:
                    cg.position = item[key_pos]
        draw_path()
        if selected_idx is not None:
            show_dotted_for_selection(selected_idx)
        _queue_path_rebuild(i)
        _arm_drag_release(gizmo)

    return gizmo


def _drop_ctrl_gizmo(ctrl):
    if ctrl is None:
        return
    if getattr(ctrl, "_drag_hold", False):
        return
    old = getattr(ctrl, "_gizmo", None)
    if old is None:
        return
    try:
        old.remove()
    except Exception:
        pass
    ctrl._gizmo = None


def _bind_ctrl_gizmo(idx, which):
    wp = waypoints[idx]
    ctrl = wp.get(f"{which}_ctrl")
    if ctrl is None:
        return
    _drop_ctrl_gizmo(ctrl)
    pos = wp.get(f"{which}_ctrl_pos", wp["pos"])
    name = f"{_wp_node(wp)}/{which}_ctrl/gizmo_{gizmo_space}"
    gizmo = _new_transform_gizmo(name, pos, WORLD_WXYZ, 0.10, True, True)

    @gizmo.on_update
    def _(_):
        global _drag_checkpointed
        if _suspend_gizmo_cb:
            return
        ctrl._drag_hold = True
        _mark_gizmo_moved()
        if not _suspend_history and not _drag_checkpointed:
            _push_history()
            _drag_checkpointed = True
        item = wp
        i = _index_of(item)
        if i is None:
            return
        new_pos = np.array(gizmo.position)
        old_pos = np.asarray(item.get(f"{which}_ctrl_pos", new_pos), dtype=float)
        if np.allclose(new_pos, old_pos, atol=1e-8):
            return
        item[f"{which}_ctrl_pos"] = new_pos
        ctrl.position = gizmo.position
        draw_path()
        if selected_idx is not None:
            show_dotted_for_selection(selected_idx)
        _mark_gizmo_moved()
        global _pending_rebuild_idx
        if _pending_rebuild_idx is None:
            _pending_rebuild_idx = i

    ctrl._gizmo = gizmo


def _mark_gizmo_moved():
    global _gizmo_last_move, _gizmo_dirty
    import time as _t
    _gizmo_last_move = _t.time()
    _gizmo_dirty = True


def _queue_path_rebuild(idx):
    """Rebuild bake after the gizmo pointer is released, not on every tick."""
    global _pending_rebuild_idx
    if idx is None:
        return
    try:
        idx = int(idx)
    except Exception:
        return
    if _pending_rebuild_idx is None:
        _pending_rebuild_idx = idx
    else:
        _pending_rebuild_idx = min(_pending_rebuild_idx, idx)
    _mark_gizmo_moved()


def flush_idle_path_rebuild(delay=0.28):
    global _pending_rebuild_idx
    if _pending_rebuild_idx is None:
        return
    if selected_ctrl is not None:
        return
    import time as _t
    if _t.time() - _gizmo_last_move < delay:
        return
    idx = _pending_rebuild_idx
    _pending_rebuild_idx = None
    _bake_call("on_waypoint_moved", idx)


def snap_global_gizmos_idle(delay=0.85):
    """World axes snap only after the pointer is released, not on a mid-drag pause."""
    return


def _arm_drag_release(gizmo):
    if getattr(gizmo, "_release_armed", False):
        return
    gizmo._release_armed = True
    for name in ("on_drag_end", "on_pointer_up", "on_mouse_up"):
        fn = getattr(gizmo, name, None)
        if not callable(fn):
            continue
        try:
            fn(lambda _e=None: _release_gizmo_drag())
            return
        except Exception:
            pass


def _recent_gizmo_drag(window=0.6):
    import time as _t
    if selected_ctrl is not None:
        item = waypoints[selected_ctrl[0]] if 0 <= selected_ctrl[0] < len(waypoints) else None
        ctrl = None if item is None else item.get("%s_ctrl" % selected_ctrl[1])
        if ctrl is not None and getattr(ctrl, "_drag_hold", False):
            return True
    return (_t.time() - float(_gizmo_last_move or 0)) < window


def _release_gizmo_drag():
    """Pointer-up: park world-mode axes, keep the Bézier handle selected."""
    global _gizmo_dirty, _drag_checkpointed, _suspend_gizmo_cb, _pending_rebuild_idx
    _drag_checkpointed = False
    try:
        simulatorview.park_sim_bot_gizmo()
    except Exception:
        pass
    if selected_ctrl is not None and 0 <= selected_ctrl[0] < len(waypoints):
        ctrl = waypoints[selected_ctrl[0]].get("%s_ctrl" % selected_ctrl[1])
        if ctrl is not None:
            ctrl._drag_hold = False
    if gizmo_space == "global" and _gizmo_dirty:
        _gizmo_dirty = False
        _suspend_gizmo_cb = True
        try:
            _snap_global_gizmos_apply()
        finally:
            _suspend_gizmo_cb = False
    if _pending_rebuild_idx is not None and selected_ctrl is None:
        idx = _pending_rebuild_idx
        _pending_rebuild_idx = None
        _bake_call("on_waypoint_moved", idx)


def _snap_global_gizmos_apply():
    for wp in waypoints:
        g = wp.get("gizmo")
        if g is None:
            continue
        try:
            if not getattr(g, "visible", False):
                continue
            g.wxyz = WORLD_WXYZ
        except Exception:
            pass
        wp["_rot0"] = np.array(wp.get("wxyz", WORLD_WXYZ), dtype=float)
    try:
        import robot_runtime as rr
        giz = getattr(rr, "_sim_gizmo", None)
        if giz is not None:
            giz.wxyz = WORLD_WXYZ
            rr._sim_rot0 = np.array(rr._sim_wxyz, dtype=float)
    except Exception:
        pass


def apply_gizmo_space(space=None):
    """Rebuild gizmos so axes match local object frame or world XYZ."""
    global gizmo_space, _suspend_gizmo_cb
    if space is not None:
        gizmo_space = "local" if str(space).lower().startswith("loc") else "global"
    _suspend_gizmo_cb = True
    try:
        for i in range(len(waypoints)):
            _bind_waypoint_gizmo(i)
            for which in ("out", "in"):
                if waypoints[i].get(f"{which}_ctrl") is not None:
                    if selected_ctrl == (i, which):
                        _bind_ctrl_gizmo(i, which)
                    else:
                        _drop_ctrl_gizmo(waypoints[i][f"{which}_ctrl"])
        for obj in scene_objects:
            vis = selected_obj_id == obj.get("id")
            simulatorview._bind_object_gizmo(obj, visible=vis)
        simulatorview.drop_sim_bot_gizmo()
        if any(k == "bot" for k, _ in _selection):
            simulatorview.show_sim_bot_gizmo()
        try:
            import robot_runtime as rr
            if hasattr(rr, "apply_gizmo_space"):
                rr.apply_gizmo_space(gizmo_space)
        except Exception:
            pass
        offsets.rebuild_offset_gizmos()
    finally:
        _suspend_gizmo_cb = False
    print("gizmo space applied", gizmo_space)
    return gizmo_space


def set_mode(mode):
    global current_mode
    current_mode = mode
    print(f"Next segment → {mode.upper()}")


def get_mode():
    return current_mode


# ---------------------------------------------------------------------------
# History: serializable document only (no Viser handles)
# ---------------------------------------------------------------------------
def _snapshot():
    snaps = []
    for wp in waypoints:
        snaps.append({
            "pos": np.array(wp["pos"], copy=True),
            "wxyz": np.array(wp.get("wxyz", [1, 0, 0, 0]), copy=True),
            "mode": wp.get("mode", "arc"),
            "out_ctrl_pos": None if wp.get("out_ctrl_pos") is None else np.array(wp["out_ctrl_pos"], copy=True),
            "in_ctrl_pos": None if wp.get("in_ctrl_pos") is None else np.array(wp["in_ctrl_pos"], copy=True),
            "t": float(wp["t"]) if wp.get("t") is not None else None,
            "q": None if wp.get("q") is None else np.array(wp["q"], copy=True),
            "q_dirty": bool(wp.get("q_dirty")),
            "force_fk": bool(wp.get("force_fk")),
            "added_from": wp.get("added_from"),
            "track_object": None if not wp.get("track_object") else dict(wp.get("track_object")),
            "ik_stale": bool(wp.get("ik_stale")),
        })
    objs = []
    for o in scene_objects:
        if not isinstance(o, dict):
            continue
        objs.append({
            "id": o.get("id"),
            "kind": o.get("kind") or "empty",
            "name": o.get("name") or "",
            "pos": np.array(o.get("pos", [0, 0, 0]), copy=True),
            "wxyz": np.array(o.get("wxyz", [1, 0, 0, 0]), copy=True),
            "scale": np.array(o.get("scale", [1, 1, 1]), copy=True),
            "parent_id": o.get("parent_id"),
            "space": o.get("space") or "world",
            "local_pos": np.array(o.get("local_pos", o.get("pos", [0, 0, 0])), copy=True),
            "local_wxyz": np.array(o.get("local_wxyz", o.get("wxyz", [1, 0, 0, 0])), copy=True),
            "camera": None if not o.get("camera") else dict(o.get("camera")),
            "offset": offsets.clean_offset(o.get("offset")),
            "collider": colliders.clean_collider(o.get("collider")),
            "comp_order": list(o.get("comp_order") or []),
            "mesh_path": o.get("mesh_path"),
        })
    try:
        import robot_runtime as rr
        bpos, bq = rr.arm_base_pose()
        arm_base = {"pos": bpos, "wxyz": bq}
    except Exception:
        arm_base = None
    return {
        "waypoints": snaps,
        "selected_idx": selected_idx,
        "objects": objs,
        "selected_obj_id": selected_obj_id,
        "arm_base": arm_base,
    }


def _push_history():
    if _suspend_history:
        return
    undo_stack.append(_snapshot())
    if len(undo_stack) > _MAX:
        undo_stack.pop(0)
    redo_stack.clear()


def checkpoint():
    _push_history()


def _apply_snap_fields(wp, s):
    wp["pos"] = np.asarray(s["pos"], dtype=float).reshape(3)
    wp["wxyz"] = np.asarray(s.get("wxyz", [1, 0, 0, 0]), dtype=float).reshape(4)
    wp["mode"] = s.get("mode", "arc")
    wp["t"] = None if s.get("t") is None else float(s.get("t"))
    if s.get("q") is None:
        wp["q"] = None
    else:
        wp["q"] = np.asarray(s["q"], dtype=float).reshape(-1)[:6]
    wp["q_dirty"] = bool(s.get("q_dirty"))
    wp["force_fk"] = bool(s.get("force_fk"))
    wp["added_from"] = s.get("added_from")
    wp["track_object"] = None if not s.get("track_object") else dict(s.get("track_object"))
    set_wp_stale(wp, bool(s.get("ik_stale")))
    pos = wp["pos"]
    wxyz = wp["wxyz"]
    for key in ("sphere", "arrow", "side", "halo"):
        h = wp.get(key)
        if h is None:
            continue
        try:
            h.position = pos
            if hasattr(h, "wxyz"):
                h.wxyz = wxyz
        except Exception:
            pass
    giz = wp.get("gizmo")
    if giz is not None:
        try:
            giz.position = pos
            giz.wxyz = _gizmo_display_wxyz(wxyz)
        except Exception:
            pass


def _restore_objects_snapshot(doc):
    global selected_obj_id
    if not isinstance(doc, dict) or "objects" not in doc:
        return
    want = list(doc.get("objects") or [])
    have = [str(o.get("id")) for o in scene_objects]
    want_ids = [str(o.get("id")) for o in want]
    if have == want_ids and len(have) == len(want):
        selected_obj_id = doc.get("selected_obj_id")
        for o, rec in zip(scene_objects, want):
            o["offset"] = offsets.clean_offset(rec.get("offset"))
            o["collider"] = colliders.clean_collider(rec.get("collider"))
            o["comp_order"] = list(rec.get("comp_order") or [])
        offsets.refresh_offset_viz()
        colliders.on_selection_changed()
        return
    try:
        simulatorview.clear_scene_objects()
    except Exception:
        scene_objects[:] = []
    for rec in want:
        try:
            simulatorview.restore_scene_object(rec)
        except Exception as e:
            print("undo restore object failed", e)
    offsets.refresh_offset_viz()
    colliders.on_selection_changed()
    sel = doc.get("selected_obj_id")
    if sel and simulatorview._find_object(sel) is not None:
        selected_obj_id = sel
    else:
        selected_obj_id = None
    try:
        simulatorview.sync_camera_tracks()
    except Exception:
        pass


def _rebuild_from_snapshot(doc):
    global _suspend_history, selected_idx, _drag_checkpointed
    _suspend_history = True
    _drag_checkpointed = False
    snaps = doc["waypoints"] if isinstance(doc, dict) else doc
    want_sel = doc.get("selected_idx") if isinstance(doc, dict) else None
    base = doc.get("arm_base") if isinstance(doc, dict) else None
    if base:
        try:
            simulatorview.set_sim_bot_pose(base["pos"], base["wxyz"])
        except Exception as e:
            print("undo arm base failed", e)

    if len(snaps) == len(waypoints):
        for i, s in enumerate(snaps):
            _apply_snap_fields(waypoints[i], s)
            if s.get("out_ctrl_pos") is not None:
                ensure_ctrl(i, "out")
                waypoints[i]["out_ctrl_pos"] = np.asarray(s["out_ctrl_pos"], dtype=np.float32)
                if waypoints[i].get("out_ctrl") is not None:
                    waypoints[i]["out_ctrl"].position = waypoints[i]["out_ctrl_pos"]
                    if hasattr(waypoints[i]["out_ctrl"], "_gizmo"):
                        waypoints[i]["out_ctrl"]._gizmo.position = waypoints[i]["out_ctrl_pos"]
            if s.get("in_ctrl_pos") is not None:
                ensure_ctrl(i, "in")
                waypoints[i]["in_ctrl_pos"] = np.asarray(s["in_ctrl_pos"], dtype=np.float32)
                if waypoints[i].get("in_ctrl") is not None:
                    waypoints[i]["in_ctrl"].position = waypoints[i]["in_ctrl_pos"]
                    if hasattr(waypoints[i]["in_ctrl"], "_gizmo"):
                        waypoints[i]["in_ctrl"]._gizmo.position = waypoints[i]["in_ctrl_pos"]
        draw_path()
        _bake_call("pin_key_times")
        _bake_call("bump_timeline_cache")
        if want_sel is not None and 0 <= int(want_sel) < len(waypoints):
            try:
                select_waypoint(int(want_sel))
            except Exception:
                pass
        hide_unselected_handles()
        _restore_objects_snapshot(doc if isinstance(doc, dict) else {})
        _suspend_history = False
        return

    _clear_scene_only()
    for s in snaps:
        add_waypoint(
            s["pos"],
            mode=s.get("mode", "arc"),
            wxyz=s.get("wxyz"),
            t=s.get("t"),
            q=s.get("q"),
            skip_ctrls=True,
        )
        wp = waypoints[-1]
        wp["q_dirty"] = bool(s.get("q_dirty"))
        wp["force_fk"] = bool(s.get("force_fk"))
        wp["added_from"] = s.get("added_from")
        wp["track_object"] = None if not s.get("track_object") else dict(s.get("track_object"))
        set_wp_stale(wp, bool(s.get("ik_stale")))
        idx = len(waypoints) - 1
        if s.get("out_ctrl_pos") is not None:
            ensure_ctrl(idx, "out")
            wp["out_ctrl_pos"] = np.asarray(s["out_ctrl_pos"], dtype=np.float32)
            wp["out_ctrl"].position = wp["out_ctrl_pos"]
            if hasattr(wp["out_ctrl"], "_gizmo"):
                wp["out_ctrl"]._gizmo.position = wp["out_ctrl_pos"]
        if s.get("in_ctrl_pos") is not None:
            ensure_ctrl(idx, "in")
            wp["in_ctrl_pos"] = np.asarray(s["in_ctrl_pos"], dtype=np.float32)
            wp["in_ctrl"].position = wp["in_ctrl_pos"]
            if hasattr(wp["in_ctrl"], "_gizmo"):
                wp["in_ctrl"]._gizmo.position = wp["in_ctrl_pos"]

    draw_path()
    _bake_call("invalidate_from", 0)
    deselect_all()
    hide_unselected_handles()
    _restore_objects_snapshot(doc if isinstance(doc, dict) else {})
    _suspend_history = False


def undo():
    if not undo_stack:
        print("Nothing to undo")
        return
    redo_stack.append(_snapshot())
    _rebuild_from_snapshot(undo_stack.pop())
    print("Undo")


def redo():
    if not redo_stack:
        print("Nothing to redo")
        return
    undo_stack.append(_snapshot())
    _rebuild_from_snapshot(redo_stack.pop())
    print("Redo")


# ---------------------------------------------------------------------------
# Scene helpers
# ---------------------------------------------------------------------------
def add_forward_arrow(parent_name, length=0.08, colors=(80, 180, 255), pos=None, wxyz=None):
    points = np.array([[[0.0, 0.0, 0.0], [0.0, 0.0, float(length)]]], dtype=np.float32)
    rgb = np.asarray(colors, dtype=np.uint8).reshape(3)
    kw = dict(
        name=f"{parent_name}/arrow",
        points=points,
        colors=rgb,
        shaft_radius=0.004,
        head_radius=0.010,
        head_length=0.018,
        visible=True,
    )
    if pos is not None:
        kw["position"] = np.asarray(pos, dtype=float).reshape(3)
    if wxyz is not None:
        kw["wxyz"] = np.asarray(wxyz, dtype=float).reshape(4)
    return server.scene.add_arrows(**kw)


def add_side_arrow(parent_name, length=0.036, colors=(80, 180, 255), pos=None, wxyz=None):
    points = np.array([[[0.0, 0.0, 0.0], [float(length), 0.0, 0.0]]], dtype=np.float32)
    rgb = np.asarray(colors, dtype=np.uint8).reshape(3)
    kw = dict(
        name=f"{parent_name}/arrow_side",
        points=points,
        colors=rgb,
        shaft_radius=0.003,
        head_radius=0.007,
        head_length=0.012,
        visible=True,
    )
    if pos is not None:
        kw["position"] = np.asarray(pos, dtype=float).reshape(3)
    if wxyz is not None:
        kw["wxyz"] = np.asarray(wxyz, dtype=float).reshape(4)
    return server.scene.add_arrows(**kw)


def make_arc_with_controls(p1, p2, c1, c2, num_points=24):
    p1 = np.asarray(p1, dtype=np.float32)
    p2 = np.asarray(p2, dtype=np.float32)
    c1 = np.asarray(c1, dtype=np.float32)
    c2 = np.asarray(c2, dtype=np.float32)
    t = np.linspace(0, 1, num_points).reshape(-1, 1)
    points = (
        (1 - t) ** 3 * p1
        + 3 * (1 - t) ** 2 * t * c1
        + 3 * (1 - t) * t ** 2 * c2
        + t ** 3 * p2
    )
    return points.astype(np.float32)


_dotted_handles = []


def _dotted_points(p1, p2, segments=12):
    p1 = np.asarray(p1, dtype=np.float32).reshape(3)
    p2 = np.asarray(p2, dtype=np.float32).reshape(3)
    t = np.linspace(0, 1, segments + 1)
    pts = (1 - t)[:, None] * p1 + t[:, None] * p2
    line_pts = [[pts[i], pts[i + 1]] for i in range(0, segments, 2)]
    if not line_pts:
        return None
    return np.array(line_pts, dtype=np.float32)


def make_dotted_line(p1, p2, name=None, color=(200, 200, 200), segments=12):
    global _line_id, _dotted_handles
    pts = _dotted_points(p1, p2, segments)
    if pts is None or server is None:
        return None
    _line_id += 1
    name = "/dot_line/%d" % _line_id
    try:
        h = server.scene.add_line_segments(
            name=name,
            points=pts,
            colors=color,
            line_width=2.0,
        )
    except Exception:
        return None
    _dotted_handles.append(h)
    return h


def clear_dotted_lines():
    global _dotted_handles
    for h in list(_dotted_handles):
        try:
            h.visible = False
        except Exception:
            pass
        _safe_remove(h)
    _dotted_handles = []
    for wp in waypoints:
        wp["dotted_out"] = None
        wp["dotted_in"] = None
        for which in ("out", "in"):
            ctrl = wp.get(f"{which}_ctrl")
            if ctrl is None:
                continue
            try:
                ctrl.visible = False
            except Exception:
                pass
            _drop_ctrl_gizmo(ctrl)


def _safe_remove(h):
    if h is None:
        return
    for fn in ("remove", "delete"):
        if hasattr(h, fn):
            try:
                getattr(h, fn)()
            except Exception:
                pass
            return


def clear_path():
    global path_handles
    for h in list(path_handles):
        _safe_remove(h)
    path_handles = []


def _segment_polyline(i, num_points=24):
    a = waypoints[i]
    b = waypoints[i + 1]
    p1 = np.asarray(a["pos"], dtype=np.float32).reshape(3)
    p2 = np.asarray(b["pos"], dtype=np.float32).reshape(3)
    if float(np.linalg.norm(p2 - p1)) < 1e-4:
        return np.stack([p1, p1], axis=0)
    if a.get("force_fk"):
        return np.stack([p1, p2], axis=0)
    if str(a.get("clip_id") or "") != str(b.get("clip_id") or ""):
        return np.stack([p1, p2], axis=0)
    c1 = a.get("out_ctrl_pos")
    c2 = b.get("in_ctrl_pos")
    if c1 is None or c2 is None:
        return np.stack([p1, p2], axis=0)
    return make_arc_with_controls(p1, p2, c1, c2, num_points=num_points)


_path_window = None
_path_win_on = False
_path_win_r = 1.0
_path_win_t = None


def in_path_window(t):
    """True when time t falls inside the path window, i.e. is hidden in 3D."""
    if _path_window is None or t is None:
        return False
    try:
        t = float(t)
    except Exception:
        return False
    return _path_window[0] <= t <= _path_window[1]


def _wp_mesh_shown(item):
    return item.get("visible", True) is not False and not in_path_window(item.get("t"))


def _apply_path_window_wps():
    for item in waypoints:
        shown = _wp_mesh_shown(item)
        for key in ("sphere", "arrow", "side"):
            h = item.get(key)
            if h is None:
                continue
            try:
                h.visible = shown
            except Exception:
                pass


def set_path_window(t0=None, t1=None):
    """Visual only: hide path, samples and waypoints within the window around
    the playhead. Bake, timeline and the path itself are untouched."""
    global _path_window
    if t0 is None or t1 is None:
        _path_window = None
    else:
        _path_window = (float(min(t0, t1)), float(max(t0, t1)))
    draw_path()
    _apply_path_window_wps()
    try:
        import path_bake as bake
        bake._draw_sample_dots()
    except Exception:
        pass


def configure_path_window(on, radius=1.0, t=None):
    """Remember clip settings; apply around t (playhead / run time)."""
    global _path_win_on, _path_win_r, _path_win_t
    _path_win_on = bool(on)
    try:
        _path_win_r = max(0.0, float(radius))
    except Exception:
        _path_win_r = 1.0
    if not _path_win_on:
        _path_win_t = None
        set_path_window(None, None)
        return
    if t is not None:
        follow_path_window(t, force=True)
    else:
        follow_path_window(_path_win_t if _path_win_t is not None else 0.0, force=True)


def follow_path_window(t, force=False, min_dt=0.04):
    """Slide the clip window with the playhead. Skip tiny steps to avoid redraw spam."""
    global _path_win_t
    if not _path_win_on:
        return False
    try:
        t = float(t)
    except Exception:
        return False
    if not force and _path_win_t is not None and abs(t - _path_win_t) < min_dt:
        return False
    _path_win_t = t
    r = float(_path_win_r)
    set_path_window(t - r, t + r)
    return True


def draw_path():
    global path_handles, _path_gen
    if server is None or len(waypoints) < 2:
        clear_path()
        return

    _path_gen += 1
    segs = []
    cols = []
    for i in range(len(waypoints) - 1):
        # Path window hides a segment once both its ends are inside it.
        if in_path_window(waypoints[i].get("t")) and in_path_window(waypoints[i + 1].get("t")):
            continue
        poly = _segment_polyline(i)
        color = _bake_call("segment_color", i) or (255, 160, 40)
        for k in range(len(poly) - 1):
            segs.append([poly[k], poly[k + 1]])
            cols.append([color, color])

    if not segs:
        clear_path()
        return

    points = np.asarray(segs, dtype=np.float32)
    colors = np.asarray(cols, dtype=np.uint8)
    if path_handles:
        try:
            path_handles[0].points = points
            path_handles[0].colors = colors
            return
        except Exception:
            clear_path()
    h = server.scene.add_line_segments(
        name="/path%s/current" % arms.ns(),
        points=points,
        colors=colors,
        line_width=3.0,
    )
    path_handles = [h]


def show_dotted_for_selection(idx):
    clear_dotted_lines()
    if idx is None or idx < 0 or idx >= len(waypoints):
        return
    wp = waypoints[idx]

    hide_out = bool(wp.get("force_fk"))
    hide_in = bool(idx > 0 and waypoints[idx - 1].get("force_fk"))
    if wp.get("out_ctrl") is not None:
        try:
            wp["out_ctrl"].visible = not hide_out
            cg = getattr(wp["out_ctrl"], "_gizmo", None)
            if cg is not None:
                cg.visible = not hide_out
        except Exception:
            pass
    if wp.get("in_ctrl") is not None:
        try:
            wp["in_ctrl"].visible = not hide_in
            cg = getattr(wp["in_ctrl"], "_gizmo", None)
            if cg is not None:
                cg.visible = not hide_in
        except Exception:
            pass
    if not hide_out and wp.get("out_ctrl") is not None and wp.get("out_ctrl_pos") is not None:
        wp["dotted_out"] = make_dotted_line(
            wp["pos"], wp["out_ctrl_pos"], name=f"/dot{arms.ns()}/{idx}/out"
        )
    if not hide_in and wp.get("in_ctrl") is not None and wp.get("in_ctrl_pos") is not None:
        wp["dotted_in"] = make_dotted_line(
            wp["pos"], wp["in_ctrl_pos"], name=f"/dot{arms.ns()}/{idx}/in"
        )

    if idx > 0 and waypoints[idx]["mode"] == "arc" and not waypoints[idx - 1].get("force_fk"):
        prev = waypoints[idx - 1]
        if prev.get("out_ctrl") is not None and prev.get("out_ctrl_pos") is not None:
            prev["dotted_out"] = make_dotted_line(
                prev["pos"], prev["out_ctrl_pos"], name=f"/dot{arms.ns()}/{idx - 1}/out"
            )

    if idx < len(waypoints) - 1 and waypoints[idx + 1]["mode"] == "arc" and not wp.get("force_fk"):
        nxt = waypoints[idx + 1]
        if nxt.get("in_ctrl") is not None and nxt.get("in_ctrl_pos") is not None:
            nxt["dotted_in"] = make_dotted_line(
                nxt["pos"], nxt["in_ctrl_pos"], name=f"/dot{arms.ns()}/{idx + 1}/in"
            )
        try:
            if nxt.get("in_ctrl") is not None:
                nxt["in_ctrl"].visible = True
        except Exception:
            pass
    elif idx < len(waypoints) - 1 and wp.get("force_fk"):
        nxt = waypoints[idx + 1]
        try:
            if nxt.get("in_ctrl") is not None:
                nxt["in_ctrl"].visible = False
                cg = getattr(nxt["in_ctrl"], "_gizmo", None)
                if cg is not None:
                    cg.visible = False
        except Exception:
            pass


def _straight_ctrl_distance(p1, p2):
    p1 = np.asarray(p1, dtype=np.float32).reshape(3)
    p2 = np.asarray(p2, dtype=np.float32).reshape(3)
    delta = p2 - p1
    L = float(np.linalg.norm(delta))
    if L < 1e-8:
        return 0.0, np.array([0.0, 0.0, 1.0], dtype=np.float32)
    return L / 3.0, delta / L


def _set_ctrl_world(idx, which, pos):
    ensure_ctrl(idx, which)
    wp = waypoints[idx]
    key = f"{which}_ctrl"
    key_pos = f"{which}_ctrl_pos"
    pos = np.asarray(pos, dtype=np.float32).reshape(3)
    wp[key_pos] = pos
    h = wp.get(key)
    if h is not None:
        try:
            h.position = pos
        except Exception:
            pass
        g = getattr(h, "_gizmo", None)
        if g is not None:
            try:
                g.position = pos
            except Exception:
                pass


def _clip_boundary(i_prev, i_new):
    a = waypoints[i_prev] if 0 <= i_prev < len(waypoints) else {}
    b = waypoints[i_new] if 0 <= i_new < len(waypoints) else {}
    ca = str(a.get("clip_id") or "")
    cb = str(b.get("clip_id") or "")
    return ca != cb


def place_segment_ctrls(i_prev, i_new, force=False):
    prev = waypoints[i_prev]
    nxt = waypoints[i_new]
    p1 = np.asarray(prev["pos"], dtype=np.float32).reshape(3)
    p2 = np.asarray(nxt["pos"], dtype=np.float32).reshape(3)
    if _clip_boundary(i_prev, i_new):
        prev["out_ctrl_pos"] = p1.copy()
        nxt["in_ctrl_pos"] = p2.copy()
        prev["transition"] = prev.get("transition") or "linear"
        return
    ca = str(prev.get("clip_id") or "")
    cb = str(nxt.get("clip_id") or "")
    if ca and ca == cb and prev.get("clip_locked") and not force:
        return
    out_ok = prev.get("out_ctrl_pos") is not None and not _ctrl_is_stub(p1, prev.get("out_ctrl_pos"))
    in_ok = nxt.get("in_ctrl_pos") is not None and not _ctrl_is_stub(p2, nxt.get("in_ctrl_pos"))
    d, u = _straight_ctrl_distance(p1, p2)
    out_pos = p1 + d * u
    in_pos = p2 - d * u
    if out_ok and not force:
        _set_ctrl_world(i_prev, "out", prev["out_ctrl_pos"])
    else:
        _set_ctrl_world(i_prev, "out", out_pos)
    if in_ok and not force:
        _set_ctrl_world(i_new, "in", nxt["in_ctrl_pos"])
    else:
        _set_ctrl_world(i_new, "in", in_pos)


def _ctrl_is_stub(anchor, ctrl_pos):
    """Default import handles sit on world-Z only (ensure_ctrl ±0.08). Not a real curve."""
    if ctrl_pos is None or anchor is None:
        return True
    p = np.asarray(anchor, dtype=float).reshape(3)
    c = np.asarray(ctrl_pos, dtype=float).reshape(3)
    d = c - p
    if float(np.linalg.norm(d)) < 1e-6:
        return True
    xy = float(np.hypot(d[0], d[1]))
    return xy < 1e-4


def _real_ctrl(row, which, anchor=None):
    pos = None if not row else row.get("%s_ctrl_pos" % which)
    if pos is None:
        return None
    base = anchor
    if base is None and row is not None:
        base = row.get("pos")
    if _ctrl_is_stub(base, pos):
        return None
    return pos


def ensure_ctrl(idx, which):
    wp = waypoints[idx]
    key = f"{which}_ctrl"
    key_pos = f"{which}_ctrl_pos"
    if wp.get(key) is not None:
        return

    name = f"{_wp_node(wp)}/{which}_ctrl"
    pos = np.asarray(wp["pos"], dtype=np.float32).reshape(3)
    if wp.get(key_pos) is not None and not _ctrl_is_stub(pos, wp.get(key_pos)):
        ctrl_pos = np.asarray(wp[key_pos], dtype=np.float32).reshape(3)
    else:
        offset = np.array([0.0, 0.0, 0.08 if which == "out" else -0.08], dtype=np.float32)
        ctrl_pos = pos + offset

    ctrl = server.scene.add_icosphere(
        name=f"{name}/sphere",
        radius=0.016,
        color=(255, 180, 50),
        position=ctrl_pos,
        visible=False,
    )
    wp[key] = ctrl
    wp[key_pos] = ctrl_pos

    _bind_ctrl_click(wp, which, ctrl)


def _ctrl_index(ctrl, which):
    if ctrl is None:
        return None
    for i, item in enumerate(waypoints):
        if item.get(f"{which}_ctrl") is ctrl:
            return i
    return None


def _bind_ctrl_click(item, which, ctrl=None):
    ctrl = ctrl or (item.get(f"{which}_ctrl") if item else None)
    if ctrl is None:
        return

    def _pick(_event=None, _item=item, _which=which, _ctrl=ctrl):
        i = _index_of(_item)
        if i is None:
            i = _ctrl_index(_ctrl, _which)
        if i is not None:
            select_control(i, _which)

    try:
        ctrl.on_click(_pick)
    except Exception:
        @ctrl.on_click
        def _(_event=None, _item=item, _which=which, _ctrl=ctrl):
            i = _index_of(_item)
            if i is None:
                i = _ctrl_index(_ctrl, _which)
            if i is not None:
                select_control(i, _which)


def next_waypoint_offset(last_pos):
    return np.asarray(last_pos, dtype=np.float32) + OFFSET_FROM_LAST


def _times_equal(a, b, eps=0.005):
    if a is None or b is None:
        return False
    return abs(float(a) - float(b)) <= eps


def ensure_builtin_sim_poses():
    """Keep Real pose and Home pose first; they cannot be removed."""
    global sim_poses
    customs = []
    seen = set()
    for p in list(sim_poses or []):
        if not isinstance(p, dict):
            continue
        pid = str(p.get("id") or "")
        if pid in ("real", "home") or p.get("builtin"):
            continue
        if pid in seen:
            continue
        seen.add(pid)
        customs.append(p)
    sim_poses = [dict(x) for x in BUILTIN_SIM_POSES] + customs
    return sim_poses


def list_sim_poses():
    return ensure_builtin_sim_poses()


def next_stored_name():
    n = 1
    have = set()
    for p in sim_poses or []:
        nm = str((p or {}).get("name") or "")
        have.add(nm.lower())
        if nm.lower().startswith("saved point ") or nm.lower().startswith("stored point "):
            try:
                n = max(n, int(nm.split()[-1]) + 1)
            except Exception:
                pass
    while ("saved point %03d" % n).lower() in have:
        n += 1
    return "saved point %03d" % n


def add_sim_pose(name=None, ik=None, q=None, q_deg=None, source=None, wxyz=None):
    global _sim_pose_seq, sim_poses
    ensure_builtin_sim_poses()
    _sim_pose_seq += 1
    pid = "pose_%d" % _sim_pose_seq
    src = str(source or "sim").lower()
    pos = None
    try:
        import robot_runtime as rr
        import path_bake as bake
        if src.startswith("real"):
            if hasattr(rr, "get_ee_pose"):
                pos, wxyz = rr.get_ee_pose("real")
            else:
                pos, wxyz = rr._cursor_pose_from_ee(rr.robot_actual, "real")
            q = rr.current_joint_q("real") if hasattr(rr, "current_joint_q") else q
        else:
            pos, wxyz = rr.get_sim_cursor_pose()
            q = rr.current_joint_q("sim") if hasattr(rr, "current_joint_q") else q
        if pos is not None and wxyz is not None:
            ik = bake._ik_from_pos_wxyz(pos, wxyz) or ik
    except Exception as e:
        print("add_sim_pose capture failed", src, e)
    ik = list(ik or [])
    rec = {
        "id": pid,
        "name": str(name or "").strip() or next_stored_name(),
        "builtin": False,
        "store": "local",
        "added_at": time.time(),
        "ik": [float(v) for v in (ik or [0, 0, 0, 0, 0, 0])[:6]],
        "q": [float(v) for v in (q or [0, 0, 0, 0, 0, 0])[:6]],
        "pos": None if pos is None else [float(v) for v in np.asarray(pos, dtype=float).reshape(3)],
        "wxyz": None if wxyz is None else [float(v) for v in np.asarray(wxyz, dtype=float).reshape(4)],
        "robot_model": None,
    }
    try:
        rec["robot_model"] = str(rr._MODEL_ID or "")
    except Exception:
        rec["robot_model"] = None
    if q_deg is not None:
        rec["q_deg"] = [float(v) for v in list(q_deg)[:6]]
    sim_poses.append(rec)
    refresh_pose_markers()
    return rec


def delete_sim_pose(pid):
    global sim_poses
    pid = str(pid or "")
    if pid in ("real", "home"):
        ensure_builtin_sim_poses()
        return False
    rec = next((p for p in (sim_poses or []) if str(p.get("id")) == pid), None)
    if rec is not None and str(rec.get("store") or "local") == "global":
        print("refuse delete global saved point", pid)
        return False
    sim_poses = [p for p in (sim_poses or []) if str(p.get("id")) != pid]
    ensure_builtin_sim_poses()
    refresh_pose_markers()
    return True


def rename_sim_pose(pid, name):
    pid = str(pid or "")
    name = str(name or "").strip()
    if not name:
        return None
    for p in sim_poses or []:
        if str(p.get("id")) == pid:
            p["name"] = name
            return p
    return None


def set_pose_store(pid, mode):
    pid = str(pid or "")
    mode = "global" if str(mode or "").lower().startswith("glob") else "local"
    rec = None
    for p in sim_poses or []:
        if str(p.get("id")) == pid:
            if p.get("builtin") or pid in ("real", "home"):
                p["store"] = "global"
                return p
            p["store"] = mode
            rec = p
            break
    try:
        persist_stored_points()
    except Exception:
        pass
    return rec


def _dump_wp_rec(w, index=None):
    rec = {
        "name": str((w or {}).get("name") or ""),
        "t": None if (w or {}).get("t") is None else float(w.get("t")),
        "mode": (w or {}).get("mode") or "arc",
        "force_fk": bool((w or {}).get("force_fk")),
        "q_dirty": bool((w or {}).get("q_dirty")),
        "added_from": (w or {}).get("added_from"),
        "transition": (w or {}).get("transition"),
        "lin_speed": (w or {}).get("lin_speed"),
    }
    try:
        import path_bake as bake
        if index is not None and 0 <= int(index) < len(getattr(bake, "segments", None) or []):
            seg = bake.segments[int(index)]
            rec["transition"] = seg.get("transition") or rec.get("transition")
            rec["lin_speed"] = seg.get("lin_speed") or rec.get("lin_speed")
            curve = None
            if hasattr(bake, "get_custom_curve"):
                curve = bake.get_custom_curve(int(index))
            if curve and rec.get("transition") == "custom":
                rec["trans_c1"] = list(curve.get("c1") or [])
                rec["trans_c2"] = list(curve.get("c2") or [])
    except Exception:
        pass
    if (w or {}).get("trans_c1") is not None:
        rec["trans_c1"] = [float(v) for v in list(w.get("trans_c1"))[:2]]
    if (w or {}).get("trans_c2") is not None:
        rec["trans_c2"] = [float(v) for v in list(w.get("trans_c2"))[:2]]
    if (w or {}).get("pos") is not None:
        rec["pos"] = [float(v) for v in np.asarray(w.get("pos"), dtype=float).reshape(3)]
    if (w or {}).get("wxyz") is not None:
        rec["wxyz"] = [float(v) for v in np.asarray(w.get("wxyz"), dtype=float).reshape(4)]
    if (w or {}).get("q") is not None:
        rec["q"] = [float(v) for v in np.asarray(w.get("q"), dtype=float).reshape(-1)[:6]]
    pos = None
    if (w or {}).get("pos") is not None:
        pos = np.asarray(w.get("pos"), dtype=float).reshape(3)
    if (w or {}).get("out_ctrl_pos") is not None and not _ctrl_is_stub(w.get("pos"), w.get("out_ctrl_pos")):
        outp = np.asarray(w.get("out_ctrl_pos"), dtype=float).reshape(3)
        rec["out_ctrl_pos"] = [float(v) for v in outp]
        if pos is not None:
            rec["out_ctrl_rel"] = [float(v) for v in (outp - pos)]
    if (w or {}).get("in_ctrl_pos") is not None and not _ctrl_is_stub(w.get("pos"), w.get("in_ctrl_pos")):
        inp = np.asarray(w.get("in_ctrl_pos"), dtype=float).reshape(3)
        rec["in_ctrl_pos"] = [float(v) for v in inp]
        if pos is not None:
            rec["in_ctrl_rel"] = [float(v) for v in (inp - pos)]
    return rec


def next_saved_path_name():
    n = 1
    have = set()
    for p in saved_paths or []:
        nm = str((p or {}).get("name") or "")
        have.add(nm.lower())
        if nm.lower().startswith("saved path "):
            try:
                n = max(n, int(nm.split()[-1]) + 1)
            except Exception:
                pass
    while ("saved path %03d" % n).lower() in have:
        n += 1
    return "saved path %03d" % n


def capture_saved_path(name=None):
    """Snapshot current timeline waypoints as a read-only saved path."""
    global _saved_path_seq, saved_paths
    src = list(waypoints or [])
    if not src:
        return None
    rows = [_dump_wp_rec(w, i) for i, w in enumerate(src)]
    times = [r["t"] for r in rows if r.get("t") is not None]
    if not times:
        for i, r in enumerate(rows):
            r["t"] = float(i)
        times = [r["t"] for r in rows]
    tmin = min(times)
    if tmin > 1e-6 or all(abs(float(r.get("t") or 0.0)) > 1e-6 for r in rows):
        head = dict(rows[0])
        head["t"] = 0.0
        rows = [head] + rows
    t0 = 0.0
    tmax = max(float(r.get("t") or 0.0) for r in rows)
    for r in rows:
        r["t"] = max(0.0, float(r.get("t") or 0.0) - t0)
    _saved_path_seq += 1
    rec = {
        "id": "path_%d" % _saved_path_seq,
        "name": str(name or "").strip() or next_saved_path_name(),
        "store": "local",
        "added_at": time.time(),
        "dur": float(tmax - t0),
        "waypoints": rows,
    }
    saved_paths.append(rec)
    if rec.get("store") == "global":
        persist_saved_paths()
    print("captured saved path", rec["id"], "n", len(rows), "dur", rec["dur"])
    return rec


def delete_saved_path(pid):
    global saved_paths, selected_path_id
    pid = str(pid or "")
    rec = next((p for p in (saved_paths or []) if str(p.get("id")) == pid), None)
    if rec is not None and str(rec.get("store") or "local") == "global":
        print("refuse delete global saved path", pid)
        return False
    saved_paths = [p for p in (saved_paths or []) if str(p.get("id")) != pid]
    if selected_path_id == pid:
        selected_path_id = None
    return True


def rename_saved_path(pid, name):
    pid = str(pid or "")
    name = str(name or "").strip()
    if not name:
        return None
    for p in saved_paths or []:
        if str(p.get("id")) == pid:
            p["name"] = name
            return p
    return None


def set_path_store(pid, mode):
    pid = str(pid or "")
    mode = "global" if str(mode or "").lower().startswith("glob") else "local"
    rec = None
    for p in saved_paths or []:
        if str(p.get("id")) == pid:
            p["store"] = mode
            rec = p
            break
    try:
        persist_saved_paths()
    except Exception:
        pass
    return rec


def persist_saved_paths():
    try:
        import webui
        rows = []
        for p in saved_paths or []:
            if str(p.get("store") or "local") != "global":
                continue
            rows.append(json.loads(json.dumps(p, default=lambda o: list(o) if hasattr(o, "__iter__") else str(o))))
        webui.save_settings({"stored_paths": rows})
    except Exception as e:
        print("persist saved_paths failed", e)


def load_global_saved_paths(rows):
    global saved_paths, _saved_path_seq
    have = {str(p.get("id")) for p in saved_paths or []}
    for rec in rows or []:
        if not isinstance(rec, dict):
            continue
        pid = str(rec.get("id") or "")
        if not pid or pid in have:
            continue
        rec = dict(rec)
        rec["store"] = "global"
        saved_paths.append(rec)
        have.add(pid)
        if pid.startswith("path_"):
            try:
                _saved_path_seq = max(_saved_path_seq, int(pid.split("_")[-1]))
            except Exception:
                pass


def place_saved_path(pid, t0=0.0, confirm=False):
    """Drop a saved path onto the timeline starting at t0. Waypoints are locked."""
    global _clip_seq
    pid = str(pid or "")
    rec = next((p for p in (saved_paths or []) if str(p.get("id")) == pid), None)
    if rec is None:
        return None
    rows = list(rec.get("waypoints") or [])
    if not rows:
        return None
    t0 = max(0.0, float(t0 or 0.0))
    rel = [float(r.get("t") or 0.0) for r in rows]
    dur = max(rel) - min(rel) if rel else float(rec.get("dur") or 0.0)
    t1 = t0 + max(0.0, dur)
    hits = []
    for i, w in enumerate(waypoints):
        tw = w.get("t")
        if tw is None:
            continue
        if t0 - 1e-4 <= float(tw) <= t1 + 1e-4:
            hits.append(i)
    if hits and not confirm:
        return {"overlap": True, "t0": t0, "t1": t1, "count": len(hits), "path_id": pid}
    if hits and confirm:
        for i in sorted(hits, reverse=True):
            _remove_wp_handles(waypoints[i])
            waypoints.pop(i)
    if not hasattr(place_saved_path, "_clip_seq"):
        place_saved_path._clip_seq = 0
    place_saved_path._clip_seq += 1
    clip_id = "clip_%d" % place_saved_path._clip_seq
    _push_history()
    placed = []
    for i, raw in enumerate(rows):
        pos = raw.get("pos")
        if pos is None:
            continue
        tw = t0 + float(raw.get("t") or 0.0) - min(rel or [0.0])
        before = {id(x) for x in waypoints}
        add_waypoint(
            pos,
            mode=raw.get("mode") or "arc",
            wxyz=raw.get("wxyz"),
            t=tw,
            q=raw.get("q"),
            skip_ctrls=True,
            replace=False,
        )
        w = next((x for x in waypoints if id(x) not in before), waypoints[-1])
        w["clip_id"] = clip_id
        w["clip_path"] = pid
        w["clip_name"] = rec.get("name")
        w["clip_locked"] = True
        pos_a = np.asarray(w["pos"], dtype=float).reshape(3)
        if raw.get("in_ctrl_rel") is not None:
            w["in_ctrl_pos"] = pos_a + np.asarray(raw.get("in_ctrl_rel"), dtype=float).reshape(3)
        elif raw.get("in_ctrl_pos") is not None:
            w["in_ctrl_pos"] = np.asarray(raw.get("in_ctrl_pos"), dtype=float).reshape(3)
        if raw.get("out_ctrl_rel") is not None:
            w["out_ctrl_pos"] = pos_a + np.asarray(raw.get("out_ctrl_rel"), dtype=float).reshape(3)
        elif raw.get("out_ctrl_pos") is not None:
            w["out_ctrl_pos"] = np.asarray(raw.get("out_ctrl_pos"), dtype=float).reshape(3)
        placed.append(w)
        if raw.get("transition"):
            w["transition"] = raw.get("transition")
        if raw.get("lin_speed") is not None:
            w["lin_speed"] = raw.get("lin_speed")
        if raw.get("trans_c1") is not None:
            w["trans_c1"] = list(raw.get("trans_c1"))
        if raw.get("trans_c2") is not None:
            w["trans_c2"] = list(raw.get("trans_c2"))
        if i == 0:
            w["clip_head"] = True
    try:
        import path_bake as bake
        bake.invalidate_from(0)
        for i, w in enumerate(waypoints):
            if str(w.get("clip_id") or "") != clip_id:
                continue
            if w.get("out_ctrl_pos") is not None:
                set_ctrl_pos(i, "out", w.get("out_ctrl_pos"))
            if w.get("in_ctrl_pos") is not None:
                set_ctrl_pos(i, "in", w.get("in_ctrl_pos"))
            if i + 1 < len(waypoints) and _clip_boundary(i, i + 1):
                enable_force_fk(i)
            if i > 0 and _clip_boundary(i - 1, i):
                enable_force_fk(i - 1)
            mode = w.get("transition")
            if mode and hasattr(bake, "apply_transition"):
                bake.apply_transition(i, mode, c1=w.get("trans_c1"), c2=w.get("trans_c2"))
        for seg in getattr(bake, "segments", []) or []:
            seg["status"] = "pending"
            seg["timing"] = None
            seg["micro"] = []
    except Exception as e:
        print("place path timing apply failed", e)
    draw_path()
    try:
        import path_bake as bake
        if hasattr(bake, "bump_timeline_cache"):
            bake.bump_timeline_cache()
    except Exception:
        pass
    print("placed saved path", pid, "clip", clip_id, "at", t0)
    return {"clip_id": clip_id, "path_id": pid, "t": t0, "dur": float(rec.get("dur") or 0.0)}


def recover_wrapped_clips():
    """Re-tag clip groups after load."""
    for w in waypoints or []:
        if w.get("clip_path") and not w.get("clip_id"):
            w["clip_id"] = "clip_" + str(w.get("clip_path"))
            w["clip_locked"] = True
        if w.get("clip_id"):
            if w.get("clip_locked") is None:
                w["clip_locked"] = True
    used = {id(w) for w in waypoints if w.get("clip_id")}
    seq = int(getattr(place_saved_path, "_clip_seq", 0) or 0)
    for rec in saved_paths or []:
        rows = list(rec.get("waypoints") or [])
        if len(rows) < 2:
            continue
        targets = []
        for r in rows:
            try:
                targets.append(np.asarray(r.get("pos"), dtype=float).reshape(3))
            except Exception:
                targets.append(None)
        if any(t is None for t in targets):
            continue
        n = len(waypoints)
        m = len(targets)
        found = None
        for i in range(0, n - m + 1):
            chunk = waypoints[i:i + m]
            if any(id(w) in used for w in chunk):
                continue
            ok = True
            for j, w in enumerate(chunk):
                try:
                    if float(np.linalg.norm(np.asarray(w.get("pos"), dtype=float).reshape(3) - targets[j])) > 1.5e-3:
                        ok = False
                        break
                except Exception:
                    ok = False
                    break
            if ok:
                found = chunk
                break
        if not found:
            continue
        seq += 1
        cid = "clip_%d" % seq
        for w in found:
            w["clip_id"] = cid
            w["clip_path"] = rec.get("id")
            w["clip_name"] = rec.get("name")
            w["clip_locked"] = True
            used.add(id(w))
        found[0]["clip_head"] = True
    for w in waypoints or []:
        cid = str(w.get("clip_id") or "")
        if cid.startswith("clip_"):
            tail = cid.split("_")[-1]
            if tail.isdigit():
                seq = max(seq, int(tail))
    place_saved_path._clip_seq = seq


def unwrap_clip(clip_id):
    clip_id = str(clip_id or "")
    n = 0
    for w in waypoints or []:
        if str(w.get("clip_id") or "") == clip_id:
            w.pop("clip_id", None)
            w.pop("clip_path", None)
            w.pop("clip_name", None)
            w.pop("clip_locked", None)
            w.pop("clip_head", None)
            n += 1
    draw_path()
    try:
        import path_bake as bake
        if hasattr(bake, "bump_timeline_cache"):
            bake.bump_timeline_cache()
    except Exception:
        pass
    return n


def persist_stored_points():
    """Write global stored points into settings.json."""
    try:
        import webui
        rows = []
        for p in sim_poses or []:
            if not isinstance(p, dict) or p.get("builtin") or str(p.get("id")) in ("real", "home"):
                continue
            if str(p.get("store") or "local") != "global":
                continue
            row = {
                "id": str(p.get("id")),
                "name": str(p.get("name") or ""),
                "store": "global",
                "added_at": float(p.get("added_at") or 0.0),
                "ik": [float(v) for v in (p.get("ik") or [0] * 6)[:6]],
                "q": [float(v) for v in (p.get("q") or [0] * 6)[:6]],
            }
            if p.get("pos") is not None:
                row["pos"] = [float(v) for v in np.asarray(p.get("pos"), dtype=float).reshape(3)]
            if p.get("wxyz") is not None:
                row["wxyz"] = [float(v) for v in np.asarray(p.get("wxyz"), dtype=float).reshape(4)]
            rows.append(row)
        webui.save_settings({"stored_points": rows})
    except Exception as e:
        print("persist stored_points failed", e)


def load_global_stored_points(rows):
    global sim_poses, _sim_pose_seq
    ensure_builtin_sim_poses()
    have = {str(p.get("id")) for p in sim_poses or []}
    for rec in rows or []:
        if not isinstance(rec, dict):
            continue
        pid = str(rec.get("id") or "")
        if not pid or pid in have or pid in ("real", "home"):
            continue
        rec = dict(rec)
        rec["store"] = "global"
        rec["builtin"] = False
        sim_poses.append(rec)
        have.add(pid)
        try:
            if pid.startswith("pose_"):
                _sim_pose_seq = max(_sim_pose_seq, int(pid.split("_")[-1]))
        except Exception:
            pass
    refresh_pose_markers()


def select_pose(pid, mode="replace"):
    note_scene_pick()
    return select_item("pose", pid, mode=mode)


def custom_stored_poses():
    out = []
    for p in sim_poses or []:
        if not isinstance(p, dict):
            continue
        pid = str(p.get("id") or "")
        if p.get("builtin") or pid in ("real", "home"):
            continue
        out.append(p)
    return out


def list_merged_points():
    """Waypoints + custom stored points for the shared hierarchy list."""
    rows = []
    for i, w in enumerate(waypoints):
        has_q = w.get("q") is not None and not bool(w.get("q_dirty"))
        rows.append({
            "kind": "wp",
            "id": "wp:%d" % i,
            "index": i,
            "name": str(w.get("name") or ("Waypoint %d" % i)),
            "added_at": float(w.get("added_at") or 0.0),
            "t": None if w.get("t") is None else float(w.get("t")),
            "can_go": bool(has_q),
        })
    seen_pose = set()
    for p in sim_poses or []:
        if not isinstance(p, dict):
            continue
        pid = str(p.get("id") or "")
        if not pid or pid in seen_pose:
            continue
        seen_pose.add(pid)
        locked = bool(p.get("builtin") or pid in ("real", "home"))
        rows.append({
            "kind": "pose",
            "id": "pose:" + pid,
            "pid": pid,
            "name": str(p.get("name") or pid),
            "added_at": float(p.get("added_at") or (0.0 if locked else 0.0)),
            "store": "global" if locked else str(p.get("store") or "local"),
            "can_go": True,
            "locked_store": locked,
        })
    order = list(point_order or [])
    if point_sort == "name":
        rows.sort(key=lambda r: (r.get("name") or "").lower())
    elif point_sort == "time":
        rows.sort(key=lambda r: float(r.get("added_at") or 0.0))
    elif point_sort == "custom" and order:
        rank = {k: i for i, k in enumerate(order)}
        rows.sort(key=lambda r: rank.get(r["id"], 10000 + (0 if r["kind"] == "wp" else 1000)))
    else:
        rows.sort(key=lambda r: (0 if r["kind"] == "wp" else 1, float(r.get("t") or r.get("added_at") or 0.0)))
    return rows


def go_to_point(data=None):
    """Move the sim cursor/arm to a waypoint or saved point."""
    data = data or {}
    kind = str(data.get("kind") or "")
    import robot_runtime as rr
    if kind == "wp" or data.get("index") is not None and kind != "pose":
        try:
            idx = int(data.get("index"))
        except Exception:
            return {"ok": False, "error": "no index"}
        if idx < 0 or idx >= len(waypoints):
            return {"ok": False, "error": "bad index"}
        w = waypoints[idx]
        q = w.get("q")
        if q is not None and not w.get("q_dirty") and hasattr(rr, "reset_sim_to_q"):
            rr.reset_sim_to_q(q)
            return {"ok": True, "kind": "wp", "index": idx}
        pos, wxyz = w.get("pos"), w.get("wxyz")
        if pos is not None and hasattr(rr, "sim_ik"):
            qq = rr.sim_ik(pos, wxyz, seed=q)
            if qq is not None and hasattr(rr, "reset_sim_to_q"):
                rr.reset_sim_to_q(qq)
                return {"ok": True, "kind": "wp", "index": idx}
        return {"ok": False, "error": "no pose"}
    pid = str(data.get("id") or data.get("pid") or "")
    if pid == "real" and hasattr(rr, "reset_sim_to_real"):
        rr.reset_sim_to_real()
        return {"ok": True, "kind": "pose", "id": pid}
    if pid == "home" and hasattr(rr, "reset_sim_to_home"):
        rr.reset_sim_to_home()
        return {"ok": True, "kind": "pose", "id": pid}
    rec = None
    for p in sim_poses or []:
        if str(p.get("id")) == pid:
            rec = p
            break
    if rec is None:
        return {"ok": False, "error": "no pose"}
    q = rec.get("q")
    # Relative to this arm's base, like its marker.
    pos, wxyz = _stored_point_world(rec)
    if pos is None:
        pos, wxyz = rec.get("pos"), rec.get("wxyz")
    cur_model = str(getattr(rr, "_MODEL_ID", "") or "")
    tagged = str(rec.get("robot_model") or "")
    model_mismatch = bool(tagged) and tagged != cur_model
    pose_q_mismatch = False
    if pos is not None and q is not None:
        try:
            model = getattr(rr, "robot_ik", None) or getattr(rr, "robot_sim", None)
            if model is not None and hasattr(rr, "_set_q"):
                rr._set_q(model, q)
                fk, _ = rr._ee_frame(model)
                err = float(np.linalg.norm(np.asarray(fk).reshape(3) - np.asarray(pos).reshape(3)))
                pose_q_mismatch = err > 0.03
            else:
                pose_q_mismatch = True
        except Exception:
            pose_q_mismatch = True
    use_xyz = bool(pos is not None and (model_mismatch or pose_q_mismatch or not q))
    if use_xyz and hasattr(rr, "sim_ik"):
        qq = rr.sim_ik(pos, wxyz, seed=q)
        if qq is not None and hasattr(rr, "reset_sim_to_q"):
            rr.reset_sim_to_q(qq)
            return {"ok": True, "kind": "pose", "id": pid, "via": "xyz"}
    if rec.get("q_deg") is not None and (q is None or not any(q)):
        if hasattr(rr, "reset_sim_to_q"):
            rr.reset_sim_to_q(rec.get("q_deg"), units="deg")
            return {"ok": True, "kind": "pose", "id": pid, "via": "q_deg"}
    if q is not None and hasattr(rr, "reset_sim_to_q"):
        rr.reset_sim_to_q(q)
        return {"ok": True, "kind": "pose", "id": pid, "via": "q"}
    return {"ok": False, "error": "no q"}


def set_point_order(ids, sort_mode=None):
    global point_order, point_sort
    if sort_mode:
        point_sort = str(sort_mode)
    if ids:
        point_order = [str(x) for x in ids]
        point_sort = "custom"
    return list_merged_points()


def rename_waypoint(idx, name):
    if idx is None or idx < 0 or idx >= len(waypoints):
        return None
    name = str(name or "").strip()
    if not name:
        return None
    waypoints[idx]["name"] = name
    return name


def _stored_point_world(p):
    """A saved point's world pose for the arm being run as. Saved points are
    shared by every arm and kept relative to the base (their IK), so each arm
    finds its own copy in front of it."""
    pid = str((p or {}).get("id") or "")
    ik = (p or {}).get("ik") or []
    if pid in ("home", "real") or (p or {}).get("builtin") or len(ik) < 6:
        return None, None
    try:
        pos, wxyz = _pb_mod()._pos_wxyz_from_ik(ik)
    except Exception:
        return None, None
    if pos is None:
        return None, None
    return np.asarray(pos, dtype=float).reshape(3), (None if wxyz is None else np.asarray(wxyz, dtype=float).reshape(4))


def _stored_point_xyz(p):
    pos, _w = _stored_point_world(p)
    if pos is not None:
        return pos
    raw = (p or {}).get("pos")
    if raw is not None and str((p or {}).get("id") or "") not in ("home", "real"):
        try:
            return np.asarray(raw, dtype=float).reshape(3)
        except Exception:
            pass
    pid = str((p or {}).get("id") or "")
    if pid == "home":
        qdeg = (p or {}).get("q_deg") or [0.0, 0.0, 90.0, 0.0, 90.0, 0.0]
        try:
            import robot_runtime as rr
            q = np.deg2rad(np.asarray(qdeg, dtype=float).reshape(-1)[:6])
            if rr.robot_sim is not None:
                hold = np.array(rr.robot_sim.angle_vector(), dtype=float)
                try:
                    rr.robot_sim.angle_vector(q)
                    pos, _w = rr._cursor_pose_from_ee(rr.robot_sim, "sim")
                finally:
                    rr.robot_sim.angle_vector(hold)
                if pos is not None:
                    return np.asarray(pos, dtype=float).reshape(3)
        except Exception:
            pass
    if pid == "real":
        try:
            import robot_runtime as rr
            if getattr(rr, "robot_actual", None) is not None:
                pos, _w = rr._cursor_pose_from_ee(rr.robot_actual, "real")
                if pos is not None:
                    return np.asarray(pos, dtype=float).reshape(3)
        except Exception:
            return None
        return None
    ik = (p or {}).get("ik") or [0, 0, 0]
    return np.array([float(ik[0]), float(ik[1]), float(ik[2])], dtype=float)


def _stored_point_wxyz(p):
    _p, w = _stored_point_world(p)
    if w is not None:
        return w
    pid = str((p or {}).get("id") or "")
    if pid == "home":
        qdeg = (p or {}).get("q_deg") or [0.0, 0.0, 90.0, 0.0, 90.0, 0.0]
        try:
            import robot_runtime as rr
            q = np.deg2rad(np.asarray(qdeg, dtype=float).reshape(-1)[:6])
            if rr.robot_sim is not None:
                hold = np.array(rr.robot_sim.angle_vector(), dtype=float)
                try:
                    rr.robot_sim.angle_vector(q)
                    _pos, wxyz = rr._cursor_pose_from_ee(rr.robot_sim, "sim")
                finally:
                    rr.robot_sim.angle_vector(hold)
                if wxyz is not None:
                    return np.asarray(wxyz, dtype=float).reshape(4)
        except Exception:
            pass
    if pid == "real":
        try:
            import robot_runtime as rr
            if getattr(rr, "robot_actual", None) is not None:
                _pos, wxyz = rr._cursor_pose_from_ee(rr.robot_actual, "real")
                if wxyz is not None:
                    return np.asarray(wxyz, dtype=float).reshape(4)
        except Exception:
            pass
    raw = (p or {}).get("wxyz")
    if raw is not None:
        try:
            return np.asarray(raw, dtype=float).reshape(4)
        except Exception:
            pass
    ik = (p or {}).get("ik") or []
    if len(ik) >= 6:
        try:
            import path_bake as bake
            _pos, wxyz = bake._pos_wxyz_from_ik(ik)
            if wxyz is not None:
                return np.asarray(wxyz, dtype=float).reshape(4)
        except Exception:
            pass
    return np.array([1.0, 0.0, 0.0, 0.0], dtype=float)


def refresh_pose_markers(aid=None):
    """Draw the shared saved points once per arm (or for arm `aid` only)."""
    if server is None:
        return
    ensure_builtin_sim_poses()
    for a in ([aid] if aid else arms.each()):
        with arms.using(a):
            _refresh_pose_markers_one()


def _refresh_pose_markers_one():
    global _pose_markers
    if server is None:
        return
    keep = set()
    shown = bool(show_pose_markers and list_vis.get("pose", False))
    grey = (244, 160, 200)
    for p in sim_poses or []:
        pid = str(p.get("id") or "")
        if not pid:
            continue
        real_ok = True
        if pid == "real":
            try:
                import robot_runtime as _rr
                real_ok = bool(_rr.robot_connected())
            except Exception:
                real_ok = False
        if pid == "real" and not real_ok:
            pack = _pose_markers.get(pid)
            if pack:
                for h in pack.values() if isinstance(pack, dict) else [pack]:
                    try:
                        h.visible = False
                    except Exception:
                        pass
            continue
        if not shown or item_vis.get(_vis_key("pose", pid), True) is False:
            pack = _pose_markers.get(pid)
            if pack:
                for h in pack.values() if isinstance(pack, dict) else [pack]:
                    try:
                        h.visible = False
                    except Exception:
                        pass
            continue
        xyz = _stored_point_xyz(p)
        if xyz is None:
            continue
        pos = np.asarray(xyz, dtype=float).reshape(3)
        wxyz = _stored_point_wxyz(p)
        keep.add(pid)
        pack = _pose_markers.get(pid)
        name = "/stored%s/%s" % (arms.ns(), pid)
        if not isinstance(pack, dict) or pack.get("sphere") is None:
            if pack is not None and not isinstance(pack, dict):
                try:
                    pack.remove()
                except Exception:
                    pass
            try:
                sph = server.scene.add_icosphere(
                    name=name + "/sphere", radius=0.02, color=grey,
                    position=tuple(pos), wxyz=tuple(wxyz),
                )
                arr = add_forward_arrow(name, colors=grey, pos=pos, wxyz=wxyz)
                side = add_side_arrow(name, colors=grey, pos=pos, wxyz=wxyz)

                def _pick(_e=None, _pid=pid):
                    try:
                        select_pose(_pid)
                    except Exception as ex:
                        print("select stored failed", ex)

                try:
                    sph.on_click(_pick)
                except Exception:
                    pass
                _pose_markers[pid] = {"sphere": sph, "arrow": arr, "side": side}
            except Exception as e:
                print("stored mark add failed", pid, e)
                continue
        else:
            try:
                pack["sphere"].position = pos
                pack["sphere"].wxyz = wxyz
                pack["sphere"].visible = True
                pack["arrow"].position = pos
                pack["arrow"].wxyz = wxyz
                pack["arrow"].visible = True
                pack["side"].position = pos
                pack["side"].wxyz = wxyz
                pack["side"].visible = True
            except Exception:
                pass
    for pid, pack in list(_pose_markers.items()):
        if pid in keep:
            continue
        hs = pack.values() if isinstance(pack, dict) else [pack]
        for h in hs:
            try:
                h.visible = False
                h.remove()
            except Exception:
                pass
        _pose_markers.pop(pid, None)


def set_show_pose_markers(on):
    global show_pose_markers
    show_pose_markers = bool(on)
    list_vis["pose"] = show_pose_markers
    refresh_pose_markers()
    return show_pose_markers


def set_list_visible(kind, on=None):
    kind = str(kind or "")
    if kind not in list_vis:
        list_vis[kind] = True
    on = (not list_vis[kind]) if on is None else bool(on)
    if kind in ("wp", "pose"):
        # Every arm's waypoints / saved points, not only the listed arm's.
        for aid in arms.each():
            with arms.using(aid):
                _set_list_visible(kind, on)
        return list_vis[kind]
    return _set_list_visible(kind, on)


def _set_list_visible(kind, on):
    list_vis[kind] = bool(on)
    if kind == "pose":
        set_show_pose_markers(list_vis["pose"])
    elif kind == "wp":
        for i, item in enumerate(waypoints):
            shown = list_vis["wp"] and item_vis.get(_vis_key("wp", i), True)
            item["visible"] = shown
            for key in ("sphere", "arrow", "side", "halo", "gizmo"):
                h = item.get(key)
                if h is None:
                    continue
                try:
                    h.visible = shown if key != "gizmo" else False
                except Exception:
                    pass
            for which in ("out_ctrl", "in_ctrl"):
                h = item.get(which)
                if h is None:
                    continue
                try:
                    h.visible = False
                except Exception:
                    pass
        try:
            draw_path()
        except Exception:
            pass
    elif kind == "obj":
        shown_list = list_vis["obj"]
        for aid in arms.each():
            with arms.using(aid):
                try:
                    import robot_runtime as rr
                    if rr._cursor_sim is not None:
                        rr._cursor_sim.visible = shown_list and item_vis.get("obj:%s" % arms.cid("sim_cursor"), True)
                except Exception:
                    pass
        for obj in scene_objects:
            shown = shown_list and item_vis.get("obj:%s" % obj.get("id"), True)
            h = obj.get("handle")
            if h is not None:
                try:
                    h.visible = shown
                except Exception:
                    pass
    return list_vis[kind]


def find_waypoint_at_time(t):
    if t is None:
        return None
    for i, wp in enumerate(waypoints):
        if _times_equal(wp.get("t"), t):
            return i
    return None


def _index_of(item):
    for i, w in enumerate(waypoints):
        if w is item:
            return i
    return None


_wp_node_seq = 0


def _new_wp_node():
    global _wp_node_seq
    _wp_node_seq += 1
    return f"/wp_n{_wp_node_seq}"


STALE_RGB = (230, 50, 50)


def set_wp_stale(item, on):
    """Red waypoint: its joints were cleared (arm moved, waypoint kept) and the
    next bake re-solves them. Drawn as a red shell on the white sphere, as a
    child node so it follows the sphere."""
    item["ik_stale"] = bool(on)
    old = item.get("stale_mark")
    if not on:
        item["stale_mark"] = None
        if old is not None:
            try:
                old.remove()
            except Exception:
                pass
        return
    if old is not None or server is None or item.get("sphere") is None:
        return
    try:
        mark = server.scene.add_icosphere(
            name=f"{_wp_node(item)}/sphere/stale", radius=0.023, color=STALE_RGB)
    except Exception as e:
        print("stale mark failed", e)
        return
    item["stale_mark"] = mark

    @mark.on_click
    def _(_e=None, _item=item):
        try:
            j = _index_of(_item)
            mode = "add" if _select_ctrl else ("range" if _select_shift else "replace")
            select_waypoint(j, mode=mode)
        except Exception as e:
            print("select stale waypoint failed", e)


def refresh_stale_marks():
    """After a bake: waypoints that got joints back turn white again."""
    for item in waypoints:
        if item.get("ik_stale") and item.get("q") is not None and not item.get("q_dirty"):
            set_wp_stale(item, False)


def _wp_node(item):
    """Viser scene path for one waypoint; stable across reorder/delete."""
    node = item.get("node")
    if not node:
        node = _new_wp_node()
        item["node"] = node
    return node


def _next_wp_label():
    used = {str(w.get("name") or "") for w in waypoints}
    n = len(waypoints)
    while ("Waypoint %d" % n) in used:
        n += 1
    return "Waypoint %d" % n


def _snap_new_t(t):
    """New keyframe time: timeline grid when Snap is on, else 0.01 s.

    Undo/redo and file load restore exact times (_suspend_history), so only
    a user add is snapped to the grid.
    """
    t = float(t)
    if snap_to_grid and not _suspend_history:
        try:
            g = max(0.001, float(grid_spacing))
        except Exception:
            g = 0.01
        t = round(t / g) * g
    return max(0.0, round(t / 0.01) * 0.01)


def _snap_t(t):
    return max(0.0, round(float(t) / 0.01) * 0.01)


def _dedupe_times():
    used = set()
    for w in waypoints:
        t = _snap_t(w.get("t") or 0.0)
        while t in used:
            t = _snap_t(t + 0.01)
        used.add(t)
        w["t"] = t


def sort_waypoints_by_time():
    """Sort editor list by t. Returns True if order changed."""
    if len(waypoints) < 2:
        _dedupe_times()
        return False
    before = [id(w) for w in waypoints]
    _dedupe_times()
    waypoints.sort(key=lambda w: float(w.get("t") or 0.0))
    return [id(w) for w in waypoints] != before


def _rebind_after_reorder():
    for i in range(len(waypoints)):
        item = waypoints[i]
        sphere = item.get("sphere")
        if sphere is not None:
            try:
                @sphere.on_click
                def _(_e=None, _item=item):
                    j = _index_of(_item)
                    if j is not None:
                        select_waypoint(j)
            except Exception:
                pass
        try:
            _bind_waypoint_gizmo(i)
            giz = item.get("gizmo")
            if giz is not None:
                giz.visible = False
        except Exception:
            pass
        for which in ("in", "out"):
            if item.get(f"{which}_ctrl") is not None:
                try:
                    _bind_ctrl_click(item, which)
                except Exception:
                    pass
                try:
                    _drop_ctrl_gizmo(item.get(f"{which}_ctrl"))
                    item[f"{which}_ctrl"].visible = False
                except Exception:
                    pass
    prune_end_handles()


def _dirty_start_from_moves(old_index_by_id, moved_ids):
    dirty = len(waypoints)
    for i, w in enumerate(waypoints):
        old = old_index_by_id.get(id(w), i)
        if old != i or id(w) in moved_ids:
            dirty = min(dirty, old, i)
    if dirty == len(waypoints):
        dirty = 0
    return max(0, dirty)


def move_waypoint(idx, pos, wxyz=None, q=None):
    wp = waypoints[idx]
    pos = np.asarray(pos, dtype=np.float32).reshape(-1)[:3]
    if wxyz is None:
        wxyz = wp.get("wxyz", np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32))
    wxyz = np.asarray(wxyz, dtype=np.float32).reshape(-1)[:4]
    delta = pos - np.asarray(wp["pos"], dtype=np.float32).reshape(3)
    wp["pos"] = pos
    wp["wxyz"] = wxyz
    if q is not None:
        wp["q"] = np.asarray(q, dtype=float).reshape(-1)[:6]
    for key in ("sphere", "arrow", "side", "halo"):
        h = wp.get(key)
        if h is None:
            continue
        try:
            h.position = pos
            if hasattr(h, "wxyz"):
                h.wxyz = wxyz
        except Exception:
            pass
    giz = wp.get("gizmo")
    if giz is not None:
        try:
            giz.position = pos
            giz.wxyz = _gizmo_display_wxyz(wxyz)
        except Exception:
            pass
    for which in ("out", "in"):
        key = f"{which}_ctrl"
        key_pos = f"{which}_ctrl_pos"
        if wp.get(key) is not None and wp.get(key_pos) is not None:
            wp[key_pos] = np.asarray(wp[key_pos], dtype=np.float32) + delta
            wp[key].position = wp[key_pos]
            if hasattr(wp[key], "_gizmo"):
                wp[key]._gizmo.position = wp[key_pos]
    draw_path()
    _bake_call("on_waypoint_moved", idx)
    print(f"Moved waypoint #{idx} to t={wp.get('t')}")


def add_waypoint(pos, mode=None, wxyz=None, t=None, q=None, replace=None, skip_ctrls=False):
    global _drag_checkpointed

    if mode is None:
        mode = current_mode or "arc"
    mode = "arc"

    if wxyz is None and isinstance(pos, (tuple, list)) and len(pos) == 2:
        pos, wxyz = pos

    pos = np.asarray(pos, dtype=np.float32).reshape(-1)[:3]
    if t is None:
        t = 0.0 if not waypoints else None
    if t is not None:
        t = _snap_new_t(t)
        hit = None
        if not _suspend_history and not allow_multiple_waypoints_per_keyframe:
            hit = find_waypoint_at_time(t)
        if hit is not None:
            if replace is False:
                return {"needs_confirm": True, "index": hit, "t": t}
            if not _suspend_history:
                _push_history()
            move_waypoint(hit, pos, wxyz=wxyz, q=q)
            return {"replaced": True, "index": hit}

    if not _suspend_history:
        _push_history()
        _quit_sim()
    _drag_checkpointed = False
    if wxyz is None:
        if waypoints:
            wxyz = waypoints[-1].get("wxyz", np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32))
        else:
            wxyz = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    wxyz = np.asarray(wxyz, dtype=np.float32).reshape(-1)[:4]

    if np.allclose(pos, 0.0, atol=1e-6) and waypoints:
        pos = next_waypoint_offset(waypoints[-1]["pos"])

    idx = len(waypoints)
    # Scene node and label must be unique, not the list index: after a delete
    # or an out-of-order insert, len(waypoints) can equal an existing
    # waypoint's old index, and Viser would replace that waypoint's sphere.
    name = _new_wp_node()
    label = _next_wp_label()

    sphere = server.scene.add_icosphere(
        name=f"{name}/sphere",
        radius=0.02,
        color=(240, 240, 240),
        position=pos,
        wxyz=wxyz,
    )

    @sphere.on_click
    def _(_e=None, _idx_at_create=idx):
        item = None
        if 0 <= _idx_at_create < len(waypoints):
            item = waypoints[_idx_at_create] if waypoints[_idx_at_create].get("sphere") is sphere else None
        if item is None:
            for w in waypoints:
                if w.get("sphere") is sphere:
                    item = w
                    break
        j = _index_of(item) if item is not None else _idx_at_create
        print("clicked wp", j)
        try:
            mode = "add" if _select_ctrl else ("range" if _select_shift else "replace")
            select_waypoint(j, mode=mode)
        except Exception as e:
            print("select_waypoint failed:", e)
            import traceback
            traceback.print_exc()

    arrow = add_forward_arrow(name, pos=pos, wxyz=wxyz)
    side = add_side_arrow(name, pos=pos, wxyz=wxyz)

    if t is None:
        if waypoints:
            prev_t = waypoints[-1].get("t")
            t = (0.0 if prev_t is None else float(prev_t)) + 1.0
        else:
            t = 0.0
    t = _snap_new_t(t)
    if q is None:
        q = _capture_current_q()
    if q is not None:
        q = np.asarray(q, dtype=float).reshape(-1)[:6]

    waypoints.append({
        "pos": pos,
        "wxyz": wxyz,
        "mode": mode,
        "name": label,
        "node": name,
        "added_at": time.time(),
        "t": t,
        "q": q,
        "q_dirty": False,
        "force_fk": False,
        "added_from": None,
        "track_object": None,
        "sphere": sphere,
        "gizmo": None,
        "arrow": arrow,
        "side": side,
        "out_ctrl": None,
        "out_ctrl_pos": None,
        "in_ctrl": None,
        "in_ctrl_pos": None,
        "dotted_out": None,
        "dotted_in": None,
        "halo": None,
    })

    _bind_waypoint_gizmo(idx)

    if mode == "arc" and idx > 0 and len(waypoints) >= 2 and not skip_ctrls:
        place_segment_ctrls(idx - 1, idx)

    item = waypoints[-1]
    old_ids = {id(w): i for i, w in enumerate(waypoints)}
    reordered = sort_waypoints_by_time()
    new_idx = _index_of(item)
    if new_idx is None:
        new_idx = idx
    if reordered:
        _rebind_after_reorder()
        if not skip_ctrls:
            if new_idx > 0:
                place_segment_ctrls(new_idx - 1, new_idx)
            if new_idx + 1 < len(waypoints):
                place_segment_ctrls(new_idx, new_idx + 1)
    prune_end_handles()
    draw_path()
    if not _suspend_history:
        try:
            select_waypoint(new_idx)
        except Exception:
            pass
    dirty = max(0, new_idx - 1)
    if reordered:
        dirty = _dirty_start_from_moves(old_ids, {id(item)})
        dirty = max(0, dirty - 1) if dirty > 0 else 0
    if len(waypoints) < 2:
        _bake_call("clear_sample_dots")
    else:
        _bake_call("invalidate_from", dirty)
    print(f"Added waypoint #{new_idx}  mode={mode}  t={t:.2f}")
    return {"ok": True, "index": new_idx, "t": t}


def _remove_ctrl(idx, which):
    if idx < 0 or idx >= len(waypoints):
        return
    wp = waypoints[idx]
    ctrl = wp.get(f"{which}_ctrl")
    _drop_ctrl_gizmo(ctrl)
    if ctrl is not None:
        if getattr(ctrl, "_removed", False):
            wp[f"{which}_ctrl"] = None
            return
        try:
            ctrl._removed = True
            ctrl.remove()
        except Exception:
            pass
    wp[f"{which}_ctrl"] = None
    wp[f"{which}_ctrl_pos"] = None


def prune_end_handles():
    """First waypoint has no incoming handle; last has no outgoing handle."""
    n = len(waypoints)
    if n == 0:
        return
    _remove_ctrl(0, "in")
    _remove_ctrl(n - 1, "out")


def set_ctrl_pos(idx, which, pos):
    if idx < 0 or idx >= len(waypoints) or pos is None:
        return
    ensure_ctrl(idx, which)
    key = f"{which}_ctrl"
    key_pos = f"{which}_ctrl_pos"
    p = np.asarray(pos, dtype=np.float32).reshape(3)
    waypoints[idx][key_pos] = p
    h = waypoints[idx].get(key)
    if h is not None:
        try:
            h.position = p
        except Exception:
            pass
        if hasattr(h, "_gizmo") and h._gizmo is not None:
            try:
                h._gizmo.position = p
            except Exception:
                pass


def poses_equal(a, b, eps=1e-4):
    pa = np.asarray(a.get("pos"), dtype=float).reshape(-1)[:3]
    pb = np.asarray(b.get("pos"), dtype=float).reshape(-1)[:3]
    if np.linalg.norm(pa - pb) > eps:
        return False
    qa = np.asarray(a.get("wxyz", [1, 0, 0, 0]), dtype=float).reshape(-1)[:4]
    qb = np.asarray(b.get("wxyz", [1, 0, 0, 0]), dtype=float).reshape(-1)[:4]
    if qa[0] < 0:
        qa = -qa
    if qb[0] < 0:
        qb = -qb
    return float(np.linalg.norm(qa - qb)) <= eps


def _capture_current_q():
    try:
        import robot_runtime as rr
        if rr.robot_sim is not None:
            return np.asarray(rr.robot_sim.angle_vector(), dtype=float).reshape(-1)[:6]
        if rr.robot_actual is not None:
            return np.asarray(rr.robot_actual.angle_vector(), dtype=float).reshape(-1)[:6]
    except Exception:
        return None
    return None


def enable_force_fk(idx):
    """Turn Force FK on via toggle_force_fk. No-op if that span is already on."""
    try:
        idx = int(idx)
    except Exception:
        return False
    if idx < 0 or idx >= len(waypoints):
        return False
    if idx >= len(waypoints) - 1:
        idx = len(waypoints) - 2
    if idx < 0:
        return False
    if waypoints[idx].get("force_fk"):
        return True
    return bool(toggle_force_fk(idx))


def toggle_force_fk(idx):
    if idx is None:
        return False
    try:
        idx = int(idx)
    except Exception:
        return False
    n = len(waypoints)
    if n < 2 or idx < 0:
        return False
    if idx >= n - 1:
        idx = n - 2
    waypoints[idx]["force_fk"] = not bool(waypoints[idx].get("force_fk"))
    on = bool(waypoints[idx]["force_fk"])
    try:
        import path_bake as bake
        bake._sync_segment_count()
        if 0 <= idx < len(bake.segments):
            bake.segments[idx] = bake._build_segment(idx)
            seed = None
            try:
                if bake.rr.robot_sim is not None:
                    seed = np.array(bake.rr.robot_sim.angle_vector(), dtype=float).reshape(-1)[:6]
            except Exception:
                seed = None
            if on:
                bake._solve_force_fk_segment(idx, bake.segments[idx], seed)
            try:
                bake.evaluate_timing(force=True)
            except Exception:
                pass
        bake._tl_cache = None
        bake._bake_generation = int(getattr(bake, "_bake_generation", 0)) + 1
    except Exception as e:
        print("force_fk bake rebuild failed", e)
    draw_path()
    show_dotted_for_selection(idx)
    print("force_fk", idx, on)
    return on


def merge_keyframes(i, j):
    if i is None or j is None:
        return {"ok": False, "error": "need two keyframes"}
    i, j = int(i), int(j)
    if i > j:
        i, j = j, i
    if j != i + 1 or i < 0 or j >= len(waypoints):
        return {"ok": False, "error": "select two consecutive keyframes"}
    if not poses_equal(waypoints[i], waypoints[j]):
        return {"ok": False, "error": "poses differ"}
    gap = float(waypoints[j].get("t") or 0.0) - float(waypoints[i].get("t") or 0.0)
    out_pos = waypoints[j].get("out_ctrl_pos")
    in_pos = waypoints[i].get("in_ctrl_pos")
    selected_idx_hold = i
    global selected_idx
    selected_idx = j
    delete_selected()
    if i >= len(waypoints):
        return {"ok": False, "error": "delete failed"}
    if in_pos is not None:
        set_ctrl_pos(i, "in", in_pos)
    if out_pos is not None:
        set_ctrl_pos(i, "out", out_pos)
    if gap > 1e-6:
        for k in range(i + 1, len(waypoints)):
            tk = waypoints[k].get("t")
            if tk is None:
                continue
            waypoints[k]["t"] = max(0.0, round((float(tk) - gap) / 0.01) * 0.01)
    selected_idx = i
    draw_path()
    _bake_call("invalidate_from", i)
    return {"ok": True, "index": i, "gap": gap}


def _flat_list(x, n):
    if x is None:
        return None
    try:
        arr = np.asarray(x, dtype=float).reshape(-1)
    except Exception:
        return None
    return [float(v) for v in arr[:n]]


def _jsonable_snapshot(doc=None):
    doc = _snapshot() if doc is None else doc
    out = {"selected_idx": doc.get("selected_idx"), "waypoints": []}
    for s in doc.get("waypoints") or []:
        out["waypoints"].append({
            "pos": _flat_list(s.get("pos"), 3) or [0, 0, 0],
            "wxyz": _flat_list(s.get("wxyz"), 4) or [1, 0, 0, 0],
            "mode": s.get("mode", "arc"),
            "out_ctrl_pos": _flat_list(s.get("out_ctrl_pos"), 3),
            "in_ctrl_pos": _flat_list(s.get("in_ctrl_pos"), 3),
            "t": None if s.get("t") is None else float(s["t"]),
            "q": _flat_list(s.get("q"), 6),
        })
    return out


def export_document():
    return {
        "format": "aubo-path",
        "version": "0.6.0",
        "document": _jsonable_snapshot(),
    }


def load_document(data):
    if not isinstance(data, dict):
        raise ValueError("invalid document")
    doc = data.get("document", data)
    wps = doc.get("waypoints", data.get("waypoints"))
    if wps is None:
        raise ValueError("no waypoints in file")
    _quit_sim()
    if not _suspend_history:
        _push_history()
    _rebuild_from_snapshot({
        "waypoints": wps,
        "selected_idx": doc.get("selected_idx"),
    })
    return len(waypoints)


def export_keyframes():
    out = []
    for i, wp in enumerate(waypoints):
        out.append({
            "index": i,
            "track": "arm",
            "t": None if wp.get("t") is None else float(wp["t"]),
            "pos": _flat_list(wp.get("pos"), 3),
            "wxyz": _flat_list(wp.get("wxyz"), 4),
            "q": _flat_list(wp.get("q"), 6),
            "mode": wp.get("mode", "arc"),
        })
    return out


def _hide_wp_halo(item):
    h = item.get("halo") if item else None
    if h is None:
        return
    try:
        h.visible = False
    except Exception:
        pass


def _show_wp_halo(idx):
    if idx < 0 or idx >= len(waypoints) or server is None:
        return
    item = waypoints[idx]
    pos = item["pos"]
    h = item.get("halo")
    if h is None:
        try:
            h = server.scene.add_icosphere(
                name=f"{_wp_node(item)}/halo",
                radius=0.034,
                color=(255, 140, 30),
                position=pos,
            )
            try:
                h.opacity = 0.4
            except Exception:
                pass
            item["halo"] = h
        except Exception as e:
            print("halo failed", e)
            return
    try:
        h.position = pos
        h.visible = True
    except Exception:
        pass
    giz = item.get("gizmo")
    if giz is not None:
        try:
            giz.visible = False
        except Exception:
            pass


def select_waypoint(idx, mode="replace"):
    global selected_idx, selected_ctrl, _drag_checkpointed, selected_obj_id, selected_obj_ids, selection_kind, _pending_rebuild_idx
    note_scene_pick()
    if _recent_gizmo_drag() and selected_ctrl is not None:
        _release_gizmo_drag()
        if _pending_rebuild_idx is not None:
            idx = _pending_rebuild_idx
            _pending_rebuild_idx = None
            _bake_call("on_waypoint_moved", idx)
        return
    _drag_checkpointed = False
    if idx is None or idx < 0 or idx >= len(waypoints):
        return
    waypoints[idx].pop("_rot0", None)
    try:
        import path_bake as bake
        bake.selected_sample = None
    except Exception:
        pass
    select_item("wp", idx, mode=mode)
    selected_ctrl = None
    wp = waypoints[idx]

    _hide_wp_halo(wp)
    for key in ("sphere", "arrow", "side"):
        h = wp.get(key)
        if h is not None:
            try:
                h.visible = True
            except Exception:
                pass
    if wp.get("gizmo") is not None:
        wp["gizmo"].visible = True
    if wp.get("arrow") is not None:
        wp["arrow"].visible = True

    def _show_handle(i, which):
        if i < 0 or i >= len(waypoints):
            return
        if which == "in" and i == 0:
            return
        if which == "out" and i == len(waypoints) - 1:
            return
        ensure_ctrl(i, which)
        ctrl = waypoints[i].get(f"{which}_ctrl")
        if ctrl is None:
            return
        try:
            ctrl.visible = True
        except Exception:
            pass
        giz = getattr(ctrl, "_gizmo", None)
        if giz is not None:
            try:
                giz.visible = False
            except Exception:
                pass

    prune_end_handles()
    if idx > 0:
        _show_handle(idx - 1, "out")
        _show_handle(idx, "in")
    if idx < len(waypoints) - 1:
        _show_handle(idx, "out")
        _show_handle(idx + 1, "in")

    show_dotted_for_selection(idx)


def deselect():
    deselect_all()


def hide_all_ctrl_gizmos():
    for item in waypoints:
        for which in ("out", "in"):
            _drop_ctrl_gizmo(item.get(f"{which}_ctrl"))


def hide_unselected_handles():
    """Bezier dots + dotted lines only for the current selection."""
    clear_dotted_lines()
    sel = set(selected_wp_idxs or [])
    if selected_idx is not None:
        sel.add(selected_idx)
    for i, item in enumerate(waypoints):
        show_mesh = _wp_mesh_shown(item)
        for key in ("sphere", "arrow", "side"):
            h = item.get(key)
            if h is None:
                continue
            try:
                h.visible = show_mesh
            except Exception:
                pass
        for which in ("out", "in"):
            ctrl = item.get(f"{which}_ctrl")
            if ctrl is None:
                continue
            try:
                ctrl.visible = False
            except Exception:
                pass
            _drop_ctrl_gizmo(ctrl)
    if selected_ctrl is not None:
        idx, which = selected_ctrl
        if 0 <= idx < len(waypoints):
            _bind_ctrl_gizmo(idx, which)
    if selected_idx is not None:
        show_dotted_for_selection(selected_idx)


def select_control(idx, which):
    global selected_ctrl, selected_idx
    note_scene_pick()
    if idx < 0 or idx >= len(waypoints):
        return
    if selected_idx != idx:
        select_waypoint(idx)
    hide_all_ctrl_gizmos()
    ctrl = waypoints[idx].get(f"{which}_ctrl")
    if ctrl is None:
        return
    selected_ctrl = (idx, which)
    _show_wp_halo(idx)
    _bind_ctrl_gizmo(idx, which)
    print("selected bezier", idx, which)


def deselect_all():
    global selected_idx, selected_ctrl, selected_obj_id, selected_obj_ids, selected_wp_idxs, _selection
    if _recent_gizmo_drag() and (selected_ctrl is not None or selected_idx is not None):
        _release_gizmo_drag()
        return
    selected_obj_id = None
    selected_obj_ids = []
    selected_wp_idxs = []
    _selection = []
    # Also clears selection_kind, so the panel falls back to the actual robot.
    _sync_selection_globals()
    simulatorview._clear_measure_line()
    clear_dotted_lines()
    simulatorview._hide_all_object_gizmos()
    _hide_all_wp_gizmos()
    try:
        import robot_runtime as rr
        rr.hide_sim_gizmo()
    except Exception:
        pass
    for item in waypoints:
        giz = item.get("gizmo")
        if giz is not None:
            try:
                giz.visible = False
            except Exception:
                pass
        if item.get("arrow") is not None:
            try:
                item["arrow"].visible = True
            except Exception:
                pass
        _hide_wp_halo(item)
        for which in ("out", "in"):
            ctrl = item.get(f"{which}_ctrl")
            if ctrl is not None:
                try:
                    ctrl.visible = False
                except Exception:
                    pass
                _drop_ctrl_gizmo(ctrl)
    selected_idx = None
    selected_ctrl = None
    try:
        import robot_runtime as rr
        if hasattr(rr, "hide_sim_gizmo"):
            rr.hide_sim_gizmo()
    except Exception:
        pass
    offsets.on_selection_changed()
    colliders.on_selection_changed()


def _remove_wp_handles(wp):
    for key in ("gizmo", "sphere", "arrow", "side", "halo"):
        h = wp.get(key)
        if h is not None:
            try:
                h.remove()
            except Exception:
                pass
    for which in ("out", "in"):
        ctrl = wp.get(f"{which}_ctrl")
        if ctrl is not None and not getattr(ctrl, "_removed", False):
            try:
                ctrl._removed = True
                ctrl.remove()
            except Exception:
                pass
            if hasattr(ctrl, "_gizmo"):
                try:
                    ctrl._gizmo.remove()
                except Exception:
                    pass
    # Node paths are unique per waypoint now, so drop the empty parent frame
    # too instead of leaving one behind per deleted waypoint.
    node = wp.get("node")
    remove_by_name = getattr(getattr(server, "scene", None), "remove_by_name", None)
    if node and callable(remove_by_name):
        try:
            remove_by_name(node)
        except Exception:
            pass


def _clear_scene_only():
    """Remove all waypoint scene objects without touching history."""
    global selected_idx
    deselect_all()
    for wp in waypoints:
        _remove_wp_handles(wp)
    waypoints.clear()
    clear_path()


def _quit_sim():
    try:
        import robot_runtime as rr
        if hasattr(rr, "exit_simulation_mode"):
            rr.exit_simulation_mode()
    except Exception:
        pass


def clear_waypoints():
    if not _suspend_history:
        _push_history()
    _quit_sim()
    _clear_scene_only()
    _bake_call("invalidate_from", 0, drop_all=True)
    print("Path cleared")


def reset_document():
    """New file: empty path, objects, tracks, custom poses, bake, cameras."""
    global extra_tracks, sim_poses, undo_stack, redo_stack, selected_obj_id
    _quit_sim()
    try:
        simulatorview.stop_all_camera_records()
    except Exception:
        pass
    try:
        clear_dotted_lines()
        hide_unselected_handles()
    except Exception:
        pass
    _clear_scene_only()
    try:
        clear_dotted_lines()
        clear_path()
    except Exception:
        pass
    try:
        simulatorview.clear_scene_objects()
    except Exception:
        pass
    extra_tracks = []
    try:
        simulatorview.sync_camera_tracks()
    except Exception:
        extra_tracks = []
    sim_poses = []
    try:
        ensure_builtin_sim_poses()
    except Exception:
        pass
    try:
        import webui
        st = webui.load_settings() or {}
        load_global_stored_points(st.get("stored_points") or [])
        load_global_saved_paths(st.get("stored_paths") or [])
    except Exception:
        pass
    selected_obj_id = None
    try:
        simulatorview._cam_frame_jpeg.clear()
    except Exception:
        pass
    _bake_call("invalidate_from", 0, drop_all=True)
    try:
        import path_bake as bake
        if hasattr(bake, "clear_bake"):
            bake.clear_bake()
        if hasattr(bake, "custom_curves"):
            bake.custom_curves = {}
        if hasattr(bake, "clipboard"):
            bake.clipboard = None
    except Exception:
        pass
    try:
        import robot_runtime as rr
        if hasattr(rr, "drop_uploaded_buffer"):
            rr.drop_uploaded_buffer()
    except Exception:
        pass
    undo_stack = []
    try:
        redo_stack = []
    except Exception:
        pass
    print("Document reset")


def _clip_indexes(idx):
    if idx is None or idx < 0 or idx >= len(waypoints):
        return []
    cid = str(waypoints[idx].get("clip_id") or "")
    if not cid:
        return [idx]
    return [i for i, w in enumerate(waypoints) if str(w.get("clip_id") or "") == cid]


def delete_selected():
    global selected_idx, _drag_checkpointed
    pose_ids = [k for knd, k in list(_selection) if knd == "pose"]
    if pose_ids:
        for pid in pose_ids:
            if str(pid) not in ("real", "home"):
                delete_sim_pose(pid)
        deselect_all()
        return
    if selected_obj_id is not None:
        _push_history()
        simulatorview.delete_scene_object(selected_obj_id)
        return
    if selected_idx is None or not waypoints:
        print("No waypoint selected")
        return
    _quit_sim()
    _push_history()
    _drag_checkpointed = False
    idxs = _clip_indexes(selected_idx)
    for idx in sorted(idxs, reverse=True):
        _remove_wp_handles(waypoints[idx])
        waypoints.pop(idx)
    deselect_all()
    draw_path()
    _bake_call("invalidate_from", 0, drop_all=True)
    print("Deleted waypoints", idxs)


def _pose_from_q(q):
    try:
        import robot_runtime as rr
        if rr.robot_sim is None or q is None:
            return None, None
        rr.robot_sim.angle_vector(np.asarray(q, dtype=float).reshape(-1)[:6])
        return rr._cursor_pose_from_ee(rr.robot_sim, "sim")
    except Exception:
        return None, None


def load_editor_waypoints(rows):
    global _suspend_history
    _push_history()
    _quit_sim()
    _suspend_history = True
    try:
        _clear_scene_only()
        stored = []
        for row in rows or []:
            pos = row.get("pos")
            wxyz = row.get("wxyz")
            q = row.get("q") if row.get("q") is not None else row.get("joints")
            if pos is None and q is not None:
                pos, wxyz = _pose_from_q(q)
            if pos is None:
                continue
            before = {id(x) for x in waypoints}
            add_waypoint(
                pos,
                mode=row.get("mode", "arc"),
                wxyz=wxyz,
                t=row.get("t"),
                q=q,
                skip_ctrls=True,
            )
            item = next((x for x in waypoints if id(x) not in before), waypoints[-1])
            item["force_fk"] = bool(row.get("force_fk"))
            item["q_dirty"] = bool(row.get("q_dirty"))
            if row.get("ik_stale"):
                set_wp_stale(item, True)
            item["added_from"] = row.get("added_from")
            if row.get("name"):
                item["name"] = str(row.get("name"))
            if row.get("added_at") is not None:
                try:
                    item["added_at"] = float(row.get("added_at"))
                except Exception:
                    pass
            item["track_object"] = None if not row.get("track_object") else dict(row.get("track_object"))
            if row.get("clip_id"):
                item["clip_id"] = row.get("clip_id")
                item["clip_path"] = row.get("clip_path")
                item["clip_name"] = row.get("clip_name")
                item["clip_locked"] = bool(row.get("clip_locked", True))
                item["clip_head"] = bool(row.get("clip_head"))
            if row.get("transition"):
                item["transition"] = row.get("transition")
            if row.get("lin_speed") is not None:
                item["lin_speed"] = row.get("lin_speed")
            stored.append(row)
        n = len(waypoints)
        for idx, row in enumerate(stored):
            if idx >= n:
                break
            out_p = _real_ctrl(row, "out", waypoints[idx].get("pos"))
            in_p = _real_ctrl(row, "in", waypoints[idx].get("pos"))
            if idx < n - 1 and out_p is not None:
                set_ctrl_pos(idx, "out", out_p)
            if idx > 0 and in_p is not None:
                set_ctrl_pos(idx, "in", in_p)
        for i in range(max(0, n - 1)):
            a = waypoints[i]
            b = waypoints[i + 1]
            out_ok = a.get("out_ctrl_pos") is not None and not _ctrl_is_stub(a.get("pos"), a.get("out_ctrl_pos"))
            in_ok = b.get("in_ctrl_pos") is not None and not _ctrl_is_stub(b.get("pos"), b.get("in_ctrl_pos"))
            if out_ok and in_ok:
                continue
            if str(a.get("clip_id") or "") and str(a.get("clip_id")) == str(b.get("clip_id") or ""):
                continue
            place_segment_ctrls(i, i + 1, force=False)
        prune_end_handles()
        seq = int(getattr(place_saved_path, "_clip_seq", 0) or 0)
        for w in waypoints:
            cid = str(w.get("clip_id") or "")
            if cid.startswith("clip_"):
                try:
                    seq = max(seq, int(cid.split("_")[-1]))
                except Exception:
                    pass
        place_saved_path._clip_seq = seq
        draw_path()
        deselect_all()
        hide_unselected_handles()
    finally:
        _suspend_history = False


def load_from_motion_points(wp_rows, sample_rows):
    """Rebuild waypoints from motion points and fit Bézier handles to samples."""
    global _suspend_history
    _push_history()
    _quit_sim()
    _suspend_history = True
    try:
        _clear_scene_only()
        built = []
        for row in wp_rows or []:
            q = row.get("q") if row.get("q") is not None else row.get("joints")
            pos = row.get("pos")
            wxyz = row.get("wxyz")
            if pos is None and q is not None:
                pos, wxyz = _pose_from_q(q)
            if pos is None:
                continue
            add_waypoint(pos, mode="arc", wxyz=wxyz, t=row.get("t"), q=q, skip_ctrls=True)
            waypoints[-1]["force_fk"] = bool(row.get("force_fk"))
            waypoints[-1]["q_dirty"] = bool(row.get("q_dirty"))
            if row.get("ik_stale"):
                set_wp_stale(waypoints[-1], True)
            waypoints[-1]["added_from"] = row.get("added_from")
            if row.get("name"):
                waypoints[-1]["name"] = str(row.get("name"))
            if row.get("added_at") is not None:
                try:
                    waypoints[-1]["added_at"] = float(row.get("added_at"))
                except Exception:
                    pass

            waypoints[-1]["track_object"] = None if not row.get("track_object") else dict(row.get("track_object"))
            built.append(waypoints[-1])
        samples = []
        for row in sample_rows or []:
            q = row.get("q") if row.get("q") is not None else row.get("joints")
            pos = row.get("pos")
            if pos is None and q is not None:
                pos, _ = _pose_from_q(q)
            if pos is None:
                continue
            samples.append({
                "t": float(row.get("t") or 0.0),
                "pos": np.asarray(pos, dtype=float).reshape(3),
            })
        fit = _bake_call
        # prefer saved handles; otherwise fit to samples
        for i in range(len(waypoints) - 1):
            a = wp_rows[i] if i < len(wp_rows or []) else {}
            b = wp_rows[i + 1] if (i + 1) < len(wp_rows or []) else {}
            out_p = _real_ctrl(a, "out", waypoints[i].get("pos"))
            in_p = _real_ctrl(b, "in", waypoints[i + 1].get("pos"))
            if out_p is not None or in_p is not None:
                if out_p is not None:
                    set_ctrl_pos(i, "out", out_p)
                if in_p is not None:
                    set_ctrl_pos(i + 1, "in", in_p)
                continue
            t0 = float(waypoints[i].get("t") or 0.0)
            t1 = float(waypoints[i + 1].get("t") or t0)
            span = max(1e-6, t1 - t0)
            mids = []
            for sm in samples:
                if t0 < sm["t"] < t1:
                    mids.append(((sm["t"] - t0) / span, sm["pos"]))
            if not mids and samples and abs(t1 - t0) < 1e-6:
                n = len(samples)
                mids = [((k + 1) / (n + 1), sm["pos"]) for k, sm in enumerate(samples)]
            place_segment_ctrls(i, i + 1, force=False)
        prune_end_handles()
        draw_path()
        deselect_all()
        hide_unselected_handles()
    finally:
        _suspend_history = False
        print("loaded waypoints", len(waypoints))


def set_select_mods(ctrl=None, shift=None):
    global _select_ctrl, _select_shift
    if ctrl is not None:
        _select_ctrl = bool(ctrl)
    if shift is not None:
        _select_shift = bool(shift)


def _hide_all_wp_gizmos():
    for item in waypoints:
        giz = item.get("gizmo")
        if giz is None:
            continue
        try:
            giz.visible = False
        except Exception:
            pass


def _sync_selection_globals():
    global selected_obj_id, selected_obj_ids, selected_wp_idxs, selected_idx, selection_kind, selected_pose_ids
    selected_obj_ids = []
    selected_wp_idxs = []
    selected_pose_ids = []
    selected_obj_id = None
    selected_idx = None
    selection_kind = None
    for kind, key in _selection:
        if kind == "sim":
            selected_obj_ids.append(arms.cid("sim_cursor"))
            selection_kind = "sim"
        elif kind == "bot":
            selected_obj_ids.append(arms.cid("sim_bot"))
            selection_kind = "bot"
        elif kind == "real":
            selected_obj_ids.append(arms.cid("real_cursor"))
            selection_kind = "real"
        elif kind == "obj":
            selected_obj_ids.append(key)
            selected_obj_id = key
            selection_kind = "object"
        elif kind == "wp":
            selected_wp_idxs.append(int(key))
            selected_idx = int(key)
            selection_kind = "waypoint"
        elif kind == "pose":
            selected_pose_ids.append(key)
            selection_kind = "pose"


def _apply_selection_viz():
    simulatorview._hide_all_object_gizmos()
    _hide_all_wp_gizmos()
    try:
        import robot_runtime as rr
        if not any(k == "sim" for k, _ in _selection):
            rr.hide_sim_gizmo()
    except Exception:
        pass
    for kind, key in _selection:
        if kind == "obj":
            obj = simulatorview._find_object(key)
            if obj is None:
                continue
            giz = obj.get("gizmo")
            if giz is None:
                giz = simulatorview._bind_object_gizmo(obj, visible=True)
            try:
                giz.visible = True
                giz.position = tuple(float(v) for v in np.asarray(obj["pos"]).reshape(3))
                giz.wxyz = _gizmo_display_wxyz(obj.get("wxyz"))
            except Exception:
                pass
        elif kind == "wp":
            try:
                i = int(key)
            except Exception:
                continue
            if 0 <= i < len(waypoints):
                giz = waypoints[i].get("gizmo")
                if giz is not None:
                    try:
                        giz.visible = True
                    except Exception:
                        pass
        elif kind == "bot":
            try:
                import robot_runtime as rr
                if not rr.edits_locked():
                    simulatorview.show_sim_bot_gizmo()
            except Exception:
                pass
        elif kind == "sim":
            try:
                import robot_runtime as rr
                # No cursor gizmo while a sim plays; playback owns the cursor.
                if rr._sim_gizmo is None and wp_server_ready() and not rr._pb["active"]:
                    p, w = rr.get_sim_cursor_pose()
                    rr._make_sim_gizmo(server, p, w)
            except Exception:
                pass
    simulatorview._update_measure_line()
    offsets.on_selection_changed()
    colliders.on_selection_changed()


def wp_server_ready():
    return server is not None


def select_item(kind, key=None, mode="replace"):
    global _selection
    note_scene_pick()
    mode = str(mode or "replace")
    token = (str(kind), key if kind != "wp" else int(key))
    if mode == "add":
        if token in _selection:
            _selection = [t for t in _selection if t != token]
        else:
            _selection.append(token)
    elif mode == "range" and kind == "wp":
        idxs = [k for knd, k in _selection if knd == "wp"]
        anchor = idxs[0] if idxs else int(key)
        lo, hi = (anchor, int(key)) if anchor <= int(key) else (int(key), anchor)
        others = [t for t in _selection if t[0] != "wp"]
        _selection = others + [("wp", i) for i in range(lo, hi + 1)]
    elif mode == "range" and kind == "pose":
        order = [str(p.get("id")) for p in (sim_poses or [])]
        ids = [str(k) for knd, k in _selection if knd == "pose"]
        anchor = ids[0] if ids else str(key)
        key = str(key)
        if anchor in order and key in order:
            i0, i1 = order.index(anchor), order.index(key)
            if i0 > i1:
                i0, i1 = i1, i0
            others = [t for t in _selection if t[0] != "pose"]
            _selection = others + [("pose", oid) for oid in order[i0:i1 + 1]]
        else:
            _selection = [token]
    elif mode == "range" and kind == "obj":
        order = [o.get("id") for o in scene_objects]
        ids = [k for knd, k in _selection if knd == "obj"]
        anchor = ids[0] if ids else key
        if anchor in order and key in order:
            i0, i1 = order.index(anchor), order.index(key)
            if i0 > i1:
                i0, i1 = i1, i0
            others = [t for t in _selection if t[0] != "obj"]
            _selection = others + [("obj", oid) for oid in order[i0:i1 + 1]]
        else:
            _selection = [token]
    else:
        _selection = [token]
    _sync_selection_globals()
    _apply_selection_viz()
    if kind == "wp" and mode == "replace" and selected_idx is not None:
        try:
            show_dotted_for_selection(selected_idx)
        except Exception:
            pass
    return True


_obj_select_hold = 0.0
item_vis = {"pose:real": False}


def set_item_vis(kind, oid, on=None):
    return set_item_visible(kind, oid, on)


def _vis_key(kind, oid):
    """item_vis key. Waypoint ids repeat between arms, so those keys carry the
    arm; saved points (shared by every arm) and objects (and arm ids like
    sim_cursor@arm2) do not."""
    if kind == "wp":
        return "%s%s:%s" % (kind, arms.ns(), oid)
    return "%s:%s" % (kind, oid)


def item_vis_view():
    """item_vis as the page reads it: the listed arm's waypoint keys without the arm."""
    out = {}
    ns = arms.ns()
    for k, v in item_vis.items():
        kind, _, rest = k.partition(":")
        if kind == "wp" + ns:
            out["wp:%s" % rest] = v
        elif kind.split("_")[0] != "wp":
            out[k] = v
    return out


def set_item_visible(kind, oid, on=None):
    if kind == "obj":
        base, aid = arms.split_id(oid)
        if aid is not None and aid != arms.current():
            with arms.using(aid):
                return set_item_visible(kind, oid, on)
    key = _vis_key(kind, oid)
    if kind == "pose" and str(oid) == "real":
        item_vis[key] = False
        refresh_pose_markers()
        return False
    cur = item_vis.get(key, True)
    item_vis[key] = (not cur) if on is None else bool(on)
    shown = bool(item_vis[key] and list_vis.get(kind if kind != "obj" else "obj", True))
    if kind == "pose":
        refresh_pose_markers()
        return item_vis[key]
    if kind == "wp":
        try:
            i = int(oid)
            if 0 <= i < len(waypoints):
                waypoints[i]["visible"] = shown
                item = waypoints[i]
                for key in ("sphere", "arrow", "side", "halo"):
                    h = item.get(key)
                    if h is None:
                        continue
                    try:
                        h.visible = shown
                    except Exception:
                        pass
                giz = item.get("gizmo")
                if giz is not None:
                    try:
                        giz.visible = bool(shown and selected_idx == i)
                    except Exception:
                        pass
                for which in ("out_ctrl", "in_ctrl"):
                    h = item.get(which)
                    if h is None:
                        continue
                    try:
                        h.visible = False
                    except Exception:
                        pass
                draw_path()
        except Exception:
            pass
    elif kind == "obj":
        base, aid = arms.split_id(oid)
        if aid is not None and base == "sim_bot":
            simulatorview.set_sim_bot_visible(shown)
        elif aid is not None and base == "sim_cursor":
            try:
                import robot_runtime as rr
                if rr._cursor_sim is not None:
                    rr._cursor_sim.visible = shown
            except Exception:
                pass
        elif aid is not None and base == "real_cursor":
            try:
                import robot_runtime as rr
                if rr._cursor_real is not None:
                    rr._cursor_real.visible = shown
            except Exception:
                pass
        else:
            obj = simulatorview._find_object(oid)
            if obj is not None:
                obj["visible"] = shown
                h = obj.get("handle")
                if h is not None:
                    try:
                        h.visible = shown and obj.get("vis_src", True) is not False
                    except Exception:
                        pass
    elif kind == "pose":
        h = _pose_markers.get(str(oid))
        if h is not None:
            try:
                h.visible = shown
            except Exception:
                pass
    return shown


# Scene objects and cameras live in simulatorview.py; importing it exposes
# its functions here as wp.<name>.
import simulatorview  # noqa: E402,F401
import offsets  # noqa: E402
import colliders  # noqa: E402
