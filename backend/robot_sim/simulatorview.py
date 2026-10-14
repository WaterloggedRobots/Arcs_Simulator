"""Scene objects in the sim world: primitives, imported meshes, parenting,
the measure tool and camera objects (sim / OpenCV / Sony previews + recording).

Shared document state (scene_objects, selection, sim_camera, extra_tracks)
stays in waypoints.py so undo/redo snapshots keep covering it; this module
reads and writes it as wp.<name>. Every function defined here is also exposed
as waypoints.<name> (see the end of the file), so existing callers such as
wp.add_scene_object(...) keep working unchanged.
"""
from pathlib import Path
import numpy as np

import app_paths
import waypoints as wp


def _front_of_arm(dist=1.0):
    pos = np.array([dist, 0.0, 0.3], dtype=float)
    wxyz = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    try:
        import robot_runtime as rr
        ee, ww = rr.get_ee_pose("sim") if hasattr(rr, "get_ee_pose") else (None, None)
        if ee is None:
            ee, ww = rr.get_sim_cursor_pose()
        if ee is not None:
            fwd = rr._dir_from_wxyz(ww)
            pos = np.asarray(ee, dtype=float).reshape(3) + np.asarray(fwd, dtype=float).reshape(3) * float(dist)
            wxyz = np.asarray(ww, dtype=float).reshape(4)
    except Exception:
        pass
    return pos, wxyz


def _qmul(a, b):
    a = np.asarray(a, dtype=float).reshape(4)
    b = np.asarray(b, dtype=float).reshape(4)
    w0, x0, y0, z0 = a
    w1, x1, y1, z1 = b
    return np.array([
        w0 * w1 - x0 * x1 - y0 * y1 - z0 * z1,
        w0 * x1 + x0 * w1 + y0 * z1 - z0 * y1,
        w0 * y1 - x0 * z1 + y0 * w1 + z0 * x1,
        w0 * z1 + x0 * y1 - y0 * x1 + z0 * w1,
    ], dtype=float)


def _qinv(q):
    q = np.asarray(q, dtype=float).reshape(4)
    n = float(np.dot(q, q)) or 1.0
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=float) / n


def _qrot(q, v):
    q = np.asarray(q, dtype=float).reshape(4)
    v = np.asarray(v, dtype=float).reshape(3)
    w, x, y, z = q
    uv = np.array([
        2.0 * (y * v[2] - z * v[1]),
        2.0 * (z * v[0] - x * v[2]),
        2.0 * (x * v[1] - y * v[0]),
    ])
    return v + w * uv + np.array([
        y * uv[2] - z * uv[1],
        z * uv[0] - x * uv[2],
        x * uv[1] - y * uv[0],
    ])


def _obj_parent(obj):
    pid = (obj or {}).get("parent_id")
    if not pid:
        return None
    if str(pid) == "sim_cursor":
        try:
            import robot_runtime as rr
            pos, wxyz = rr.get_sim_cursor_pose()
            return {
                "id": "sim_cursor",
                "pos": np.asarray(pos if pos is not None else [0, 0, 0], dtype=float).reshape(3),
                "wxyz": np.asarray(wxyz if wxyz is not None else [1, 0, 0, 0], dtype=float).reshape(4),
                "parent_id": None,
            }
        except Exception:
            return None
    return _find_object(pid)


def _set_world_from_local(obj):
    loc = np.asarray(obj.get("local_pos", obj.get("pos", [0, 0, 0])), dtype=float).reshape(3)
    lq = np.asarray(obj.get("local_wxyz", obj.get("wxyz", [1, 0, 0, 0])), dtype=float).reshape(4)
    n = float(np.linalg.norm(lq)) or 1.0
    lq = lq / n
    par = _obj_parent(obj)
    if par is None:
        obj["pos"] = loc
        obj["wxyz"] = lq
        return
    pw = np.asarray(par.get("pos"), dtype=float).reshape(3)
    pq = np.asarray(par.get("wxyz", [1, 0, 0, 0]), dtype=float).reshape(4)
    obj["pos"] = pw + _qrot(pq, loc)
    wq = _qmul(pq, lq)
    n = float(np.linalg.norm(wq)) or 1.0
    obj["wxyz"] = wq / n


def _set_local_from_world(obj):
    wpos = np.asarray(obj.get("pos", [0, 0, 0]), dtype=float).reshape(3)
    wq = np.asarray(obj.get("wxyz", [1, 0, 0, 0]), dtype=float).reshape(4)
    n = float(np.linalg.norm(wq)) or 1.0
    wq = wq / n
    par = _obj_parent(obj)
    if par is None:
        obj["local_pos"] = wpos.copy()
        obj["local_wxyz"] = wq.copy()
        return
    pw = np.asarray(par.get("pos"), dtype=float).reshape(3)
    pq = np.asarray(par.get("wxyz", [1, 0, 0, 0]), dtype=float).reshape(4)
    inv = _qinv(pq)
    obj["local_pos"] = _qrot(inv, wpos - pw)
    lq = _qmul(inv, wq)
    n = float(np.linalg.norm(lq)) or 1.0
    obj["local_wxyz"] = lq / n


def _propagate_children(oid):
    for child in wp.scene_objects:
        if child.get("parent_id") == oid:
            _set_world_from_local(child)
            _apply_object_xform(child)
            _refresh_camera_viz(child)
            _propagate_children(child.get("id"))


def reparent_object(oid, parent_id=None):
    obj = _find_object(oid)
    if obj is None or str(oid) == "sim_cursor":
        return False
    pid = None if not parent_id or parent_id in ("", None) else str(parent_id)
    if pid == "sim_cursor":
        pid = "sim_cursor"
    if pid == oid:
        return False
    walk = pid
    while walk:
        if walk == oid:
            return False
        p = _find_object(walk)
        walk = None if p is None else p.get("parent_id")
    obj["parent_id"] = pid
    _set_local_from_world(obj)
    _apply_object_xform(obj)
    return True


def _item_pos(kind, key):
    if kind == "sim":
        try:
            import robot_runtime as rr
            p, _ = rr.get_sim_cursor_pose()
            if p is not None:
                return np.asarray(p, dtype=float).reshape(3)
        except Exception:
            return None
    if kind == "obj":
        obj = _find_object(key)
        if obj is None:
            return None
        return np.asarray(obj.get("pos"), dtype=float).reshape(3)
    if kind == "wp":
        try:
            i = int(key)
        except Exception:
            return None
        if 0 <= i < len(wp.waypoints):
            return np.asarray(wp.waypoints[i].get("pos"), dtype=float).reshape(3)
    if kind == "pose":
        for p in wp.sim_poses or []:
            if str(p.get("id")) == str(key):
                ik = p.get("ik") or [0, 0, 0]
                return np.array([float(ik[0]), float(ik[1]), float(ik[2])], dtype=float)
        return None
    return None


def measure_selected_objects():
    items = list(wp._selection)
    if len(items) != 2:
        return None
    p0 = _item_pos(items[0][0], items[0][1])
    p1 = _item_pos(items[1][0], items[1][1])
    if p0 is None or p1 is None:
        return None
    d = p1 - p0
    return {
        "x": round(float(d[0]), 2),
        "y": round(float(d[1]), 2),
        "z": round(float(d[2]), 2),
        "d": round(float(np.linalg.norm(d)), 2),
    }


def _clear_measure_line():
    if wp._measure_line is not None:
        try:
            wp._measure_line.remove()
        except Exception:
            pass
        wp._measure_line = None


def _update_measure_line():
    _clear_measure_line()
    if wp.server is None or len(wp._selection) != 2:
        return
    p0 = _item_pos(wp._selection[0][0], wp._selection[0][1])
    p1 = _item_pos(wp._selection[1][0], wp._selection[1][1])
    if p0 is None or p1 is None:
        return
    pts = wp._dotted_points(p0, p1, 16)
    if pts is None or len(pts) < 2:
        return
    try:
        wp._measure_line = wp.server.scene.add_line_segments(
            name="/measure/line",
            points=np.asarray(pts, dtype=float),
            colors=(255, 220, 40),
            line_width=2.5,
        )
    except Exception:
        wp._measure_line = None


def _hide_all_object_gizmos():
    for obj in wp.scene_objects:
        giz = obj.get("gizmo")
        if giz is None:
            continue
        try:
            giz.visible = False
        except Exception:
            pass


def _find_object(oid):
    for obj in wp.scene_objects:
        if obj.get("id") == oid:
            return obj
    return None


_PRIM_KINDS = ("empty", "sphere", "cube", "camera", "mesh", "import")


def _is_mesh_kind(kind):
    return str(kind or "").lower() in ("mesh", "import")


def _apply_object_xform(obj):
    h = obj.get("handle")
    if h is None:
        return
    pos = np.asarray(obj["pos"], dtype=float).reshape(3)
    wxyz = np.asarray(obj.get("wxyz", [1, 0, 0, 0]), dtype=float).reshape(4)
    sc = np.asarray(obj.get("scale", [1, 1, 1]), dtype=float).reshape(3)
    try:
        h.position = tuple(float(v) for v in pos)
        h.wxyz = tuple(float(v) for v in wxyz)
    except Exception:
        pass
    kind = obj.get("kind")
    try:
        if _is_mesh_kind(kind) and obj.get("_mesh_verts") is not None:
            prev = obj.get("_applied_scale")
            cur = (float(sc[0]), float(sc[1]), float(sc[2]))
            if prev != cur:
                obj["_applied_scale"] = cur
                _respawn_imported_mesh(obj)
        elif kind in ("cube", "camera") and hasattr(h, "dimensions"):
            base = 0.06 if kind == "camera" else 0.1
            h.dimensions = (base * float(sc[0]), base * float(sc[1]), base * float(sc[2]))
        elif kind in ("sphere", "empty") and hasattr(h, "radius"):
            base = 0.02 if kind == "empty" else 0.05
            h.radius = base * max(1e-4, float(np.mean(np.abs(sc))))
    except Exception:
        pass
    giz = obj.get("gizmo")
    if giz is not None:
        try:
            giz.position = tuple(float(v) for v in pos)
            giz.wxyz = wp._gizmo_display_wxyz(wxyz)
        except Exception:
            pass


def _bind_object_gizmo(obj, visible=False):
    old = obj.get("gizmo")
    if old is not None:
        try:
            old.remove()
        except Exception:
            pass
    name = "/obj_%s/gizmo_%s" % (obj["id"], wp.gizmo_space)
    giz = wp._new_transform_gizmo(name, obj["pos"], wp._gizmo_display_wxyz(obj.get("wxyz")), 0.18, False, visible)
    obj["gizmo"] = giz

    @giz.on_update
    def _(_):
        wp.selected_obj_id = obj.get("id")
        if obj.get("id") not in wp.selected_obj_ids:
            wp.selected_obj_ids = [obj.get("id")]
        wp.selection_kind = "object"
        wp.selected_idx = None
        if wp._suspend_gizmo_cb:
            return
        old_pos = np.asarray(obj.get("pos"), dtype=float).reshape(3)
        old_q = np.asarray(obj.get("wxyz", [1, 0, 0, 0]), dtype=float).reshape(4)
        new_pos = np.array(giz.position, dtype=float).reshape(3)
        new_q = wp._keep_heading(old_q, wp._orientation_from_gizmo(giz, old_q, obj.get("_rot0", old_q)))
        delta = new_pos - old_pos
        dq = _qmul(new_q, _qinv(old_q))
        obj["pos"] = new_pos
        obj["wxyz"] = new_q
        _set_local_from_world(obj)
        _apply_object_xform(obj)
        _refresh_camera_viz(obj)
        _propagate_children(obj.get("id"))
        for oid in list(wp.selected_obj_ids):
            if oid == obj.get("id"):
                continue
            other = _find_object(oid)
            if other is None:
                continue
            other["pos"] = np.asarray(other.get("pos"), dtype=float).reshape(3) + delta
            oq = np.asarray(other.get("wxyz", [1, 0, 0, 0]), dtype=float).reshape(4)
            nq = _qmul(dq, oq)
            n = float(np.linalg.norm(nq)) or 1.0
            other["wxyz"] = nq / n
            _set_local_from_world(other)
            _apply_object_xform(other)
            _propagate_children(other.get("id"))
        wp._arm_drag_release(giz)

    return giz


def _read_stl_arrays(path):
    raw = Path(path).read_bytes()
    if raw[:5].lower() == b"solid" and b"facet" in raw[:800].lower():
        text = raw.decode("utf-8", errors="ignore")
        verts = []
        for line in text.splitlines():
            parts = line.strip().split()
            if len(parts) >= 4 and parts[0].lower() == "vertex":
                try:
                    verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
                except Exception:
                    pass
        if len(verts) >= 3:
            v = np.asarray(verts, dtype=np.float32)
            n = (len(v) // 3) * 3
            v = v[:n]
            f = np.arange(n, dtype=np.uint32).reshape(-1, 3)
            return v, f
    if len(raw) < 84:
        return None, None
    ntri = int.from_bytes(raw[80:84], "little", signed=False)
    need = 84 + ntri * 50
    if ntri <= 0 or len(raw) < need:
        return None, None
    verts = np.empty((ntri * 3, 3), dtype=np.float32)
    off = 84
    for i in range(ntri):
        chunk = raw[off + 12:off + 48]
        verts[i * 3:i * 3 + 3] = np.frombuffer(chunk, dtype="<f4").reshape(3, 3)
        off += 50
    faces = np.arange(ntri * 3, dtype=np.uint32).reshape(-1, 3)
    return verts, faces


def _fit_import_mesh(verts):
    """Center an imported mesh and bring mm-scale CAD down to the robot scene."""
    v = np.asarray(verts, dtype=np.float32)
    if v.size == 0:
        return v
    lo = v.min(axis=0)
    hi = v.max(axis=0)
    center = 0.5 * (lo + hi)
    v = v - center
    span = float(np.max(hi - lo))
    if span > 2.0:
        v = v * (0.25 / span)
    return v.astype(np.float32)


def _load_mesh_arrays(path):
    if not path:
        return None, None, "no mesh path"
    p = Path(app_paths.relocate(path, app_paths.MESHES_DIR))
    if not p.is_file():
        return None, None, "mesh missing: %s" % path
    suffix = p.suffix.lower()
    err = None
    try:
        import trimesh
        loaded = trimesh.load(str(p), force="mesh", skip_materials=True)
        if isinstance(loaded, trimesh.Scene):
            geoms = [g for g in loaded.geometry.values() if hasattr(g, "vertices")]
            loaded = trimesh.util.concatenate(geoms) if geoms else None
        if loaded is not None and hasattr(loaded, "vertices") and hasattr(loaded, "faces"):
            v = np.asarray(loaded.vertices, dtype=np.float32)
            f = np.asarray(loaded.faces, dtype=np.uint32)
            if len(v) and len(f):
                return _fit_import_mesh(v), f, None
        err = "trimesh produced no faces"
    except Exception as e:
        err = str(e)
    if suffix == ".stl":
        try:
            v, f = _read_stl_arrays(p)
            if v is not None and f is not None and len(v) and len(f):
                return _fit_import_mesh(v), f, None
        except Exception as e:
            err = str(e)
    return None, None, err or ("cannot load %s" % suffix)


def _bind_mesh_click(obj):
    handle = obj.get("handle")
    if handle is None:
        return
    oid = obj["id"]
    def _on_mesh(_e=None, _oid=oid):
        mode = "add" if wp._select_ctrl else ("range" if wp._select_shift else "replace")
        select_scene_object(_oid, mode=mode)
    try:
        handle.on_click(_on_mesh)
    except Exception:
        pass


def _respawn_imported_mesh(obj):
    if wp.server is None:
        return
    verts = obj.get("_mesh_verts")
    faces = obj.get("_mesh_faces")
    name = "/obj_%s/mesh" % obj["id"]
    pos = tuple(float(v) for v in np.asarray(obj["pos"]).reshape(3))
    wxyz = tuple(float(v) for v in np.asarray(obj.get("wxyz", [1, 0, 0, 0])).reshape(4))
    sc = np.asarray(obj.get("scale", [1, 1, 1]), dtype=float).reshape(3)
    obj["_applied_scale"] = (float(sc[0]), float(sc[1]), float(sc[2]))
    old = obj.get("handle")
    if old is not None:
        try:
            old.remove()
        except Exception:
            pass
        obj["handle"] = None
    handle = None
    if verts is not None and faces is not None:
        scaled = np.asarray(verts, dtype=np.float32).copy()
        scaled[:, 0] *= float(sc[0])
        scaled[:, 1] *= float(sc[1])
        scaled[:, 2] *= float(sc[2])
        faces = np.asarray(faces, dtype=np.uint32)
        try:
            handle = wp.server.scene.add_mesh_simple(
                name=name,
                vertices=scaled,
                faces=faces,
                color=(200, 200, 200),
                wxyz=wxyz,
                position=pos,
            )
        except TypeError:
            try:
                handle = wp.server.scene.add_mesh_simple(name, scaled, faces)
            except Exception as e:
                print("add_mesh_simple failed", e)
                handle = None
        except Exception as e:
            print("add_mesh_simple failed", e)
            handle = None
    if handle is None:
        handle = wp.server.scene.add_box(
            name=name,
            dimensions=(0.1 * float(sc[0]), 0.1 * float(sc[1]), 0.1 * float(sc[2])),
            position=pos,
            wxyz=wxyz,
            color=(180, 160, 90),
        )
    obj["handle"] = handle
    _bind_mesh_click(obj)


def _spawn_object_mesh(obj):
    kind = obj.get("kind")
    name = "/obj_%s/mesh" % obj["id"]
    pos = tuple(float(v) for v in np.asarray(obj["pos"]).reshape(3))
    wxyz = tuple(float(v) for v in np.asarray(obj.get("wxyz", [1, 0, 0, 0])).reshape(4))
    sc = np.asarray(obj.get("scale", [1, 1, 1]), dtype=float).reshape(3)
    handle = None
    if _is_mesh_kind(kind):
        verts, faces, err = _load_mesh_arrays(obj.get("mesh_path"))
        obj["_mesh_verts"] = verts
        obj["_mesh_faces"] = faces
        obj["mesh_error"] = err
        if err:
            print("import mesh", obj.get("id"), err)
        _respawn_imported_mesh(obj)
        return
    if kind == "camera":
        kind = "cube"
        obj["kind"] = "cube"
        if not obj.get("camera"):
            obj["camera"] = {
                "type": "sim", "vfov": 60.0, "hfov": 90.0,
                "script": "", "model": "", "enabled": True, "cv_index": 1,
            }
    if kind == "cube":
        handle = wp.server.scene.add_box(
            name=name,
            dimensions=(0.1 * float(sc[0]), 0.1 * float(sc[1]), 0.1 * float(sc[2])),
            position=pos,
            wxyz=wxyz,
            color=(220, 220, 220),
        )
    else:
        base = 0.02 if kind == "empty" else 0.05
        color = (80, 160, 255) if kind == "empty" else (220, 220, 220)
        handle = wp.server.scene.add_icosphere(
            name=name,
            radius=base * max(1e-4, float(np.mean(np.abs(sc)))),
            position=pos,
            wxyz=wxyz,
            color=color,
        )
        if kind == "empty":
            try:
                handle.opacity = 0.4
            except Exception:
                pass
    obj["handle"] = handle
    if obj.get("camera"):
        try:
            _refresh_camera_viz(obj)
        except Exception:
            pass

    oid = obj["id"]
    def _on_mesh(_e=None, _oid=oid):
        mode = "add" if wp._select_ctrl else ("range" if wp._select_shift else "replace")
        select_scene_object(_oid, mode=mode)
    try:
        handle.on_click(_on_mesh)
    except Exception:
        pass


def clear_scene_objects():
    for obj in list(wp.scene_objects):
        delete_scene_object(obj.get("id"))
    wp.scene_objects[:] = []
    wp.selected_obj_id = None


def _camera_track_name(obj):
    raw = str((obj or {}).get("name") or (obj or {}).get("id") or "object").strip()
    if raw.lower().endswith(" camera"):
        return raw
    return raw + " camera"


def sync_camera_tracks():
    """One extra timeline track per camera component; id cam_<oid>."""
    cams = [o for o in wp.scene_objects if o.get("camera")]
    if wp.sim_camera:
        cams = [{"id": "sim", "name": "Sim cursor", "camera": wp.sim_camera}] + list(cams)
    ids = {str(o.get("id")) for o in cams}
    kept = []
    seen = set()
    for tr in list(wp.extra_tracks or []):
        if not isinstance(tr, dict):
            kept.append(tr)
            continue
        tid = str(tr.get("id") or "")
        if tid.startswith("cam_"):
            oid = tid[4:]
            if oid not in ids:
                continue
            obj = next((o for o in cams if str(o.get("id")) == oid), None)
            if obj is not None:
                tr["name"] = _camera_track_name(obj)
            seen.add(oid)
            kept.append(tr)
        else:
            kept.append(tr)
    for obj in cams:
        oid = str(obj.get("id"))
        if oid in seen:
            continue
        kept.append({
            "id": "cam_" + oid,
            "name": _camera_track_name(obj),
            "keys": [],
        })
    wp.extra_tracks = kept
    return wp.extra_tracks


def restore_scene_object(rec):
    kind = str((rec or {}).get("kind") or "empty").lower()
    if kind == "import":
        kind = "mesh"
    if kind not in _PRIM_KINDS:
        kind = "empty"
    oid = str((rec or {}).get("id") or "")
    if not oid:
        wp._obj_seq += 1
        oid = "o%d" % wp._obj_seq
    else:
        try:
            if oid.startswith("o"):
                wp._obj_seq = max(wp._obj_seq, int(oid[1:]))
        except Exception:
            pass
    obj = {
        "id": oid,
        "kind": kind,
        "name": str((rec or {}).get("name") or kind),
        "pos": np.asarray((rec or {}).get("pos") or [1, 0, 0.2], dtype=float).reshape(3),
        "wxyz": np.asarray((rec or {}).get("wxyz") or [1, 0, 0, 0], dtype=float).reshape(4),
        "local_pos": np.asarray((rec or {}).get("local_pos") or (rec or {}).get("pos") or [1, 0, 0.2], dtype=float).reshape(3),
        "local_wxyz": np.asarray((rec or {}).get("local_wxyz") or (rec or {}).get("wxyz") or [1, 0, 0, 0], dtype=float).reshape(4),
        "parent_id": (rec or {}).get("parent_id"),
        "space": (rec or {}).get("space") or "world",
        "scale": np.asarray((rec or {}).get("scale") or [1, 1, 1], dtype=float).reshape(3),
        "handle": None,
        "gizmo": None,
        "camera": dict((rec or {}).get("camera") or {}) if ((rec or {}).get("camera") or kind == "camera") else None,
        "mesh_path": str((rec or {}).get("mesh_path") or "") if _is_mesh_kind(kind) else None,
        "mesh_error": None,
    }
    wp.scene_objects.append(obj)
    if wp.server is not None:
        _spawn_object_mesh(obj)
        _bind_object_gizmo(obj, visible=False)
    return obj


def add_scene_object(kind="empty", mesh_path=None, name=None):
    kind = str(kind or "empty").lower()
    if kind == "import":
        kind = "mesh"
    add_cam = (kind == "camera")
    if add_cam:
        kind = "cube"
    if kind not in _PRIM_KINDS:
        kind = "empty"
    wp._push_history()
    wp._obj_seq += 1
    oid = "o%d" % wp._obj_seq
    pos = np.array([1.0, 0.0, 0.2], dtype=float)
    wxyz = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    names = {"empty": "Empty", "sphere": "Sphere", "cube": "Cube", "camera": "Camera", "mesh": "Import"}
    label = name
    if not label:
        if _is_mesh_kind(kind) and mesh_path:
            label = Path(str(mesh_path)).stem or ("Import %d" % wp._obj_seq)
        elif add_cam:
            label = "Camera %d" % wp._obj_seq
        else:
            label = "%s %d" % (names.get(kind, "Object"), wp._obj_seq)
    obj = {
        "id": oid,
        "kind": kind,
        "name": str(label),
        "pos": np.asarray(pos, dtype=float).reshape(3),
        "wxyz": np.asarray(wxyz, dtype=float).reshape(4),
        "local_pos": np.asarray(pos, dtype=float).reshape(3),
        "local_wxyz": np.asarray(wxyz, dtype=float).reshape(4),
        "parent_id": None,
        "space": "world",
        "scale": np.array([0.5, 0.4, 0.3] if add_cam else [1.0, 1.0, 1.0], dtype=float),
        "handle": None,
        "gizmo": None,
        "camera": None,
        "mesh_path": str(mesh_path) if (_is_mesh_kind(kind) and mesh_path) else (str(mesh_path) if mesh_path else None),
        "mesh_error": None,
    }
    wp.scene_objects.append(obj)
    if wp.server is not None:
        try:
            _spawn_object_mesh(obj)
        except Exception as e:
            print("spawn object mesh failed", kind, oid, e)
        try:
            _bind_object_gizmo(obj, visible=False)
        except Exception as e:
            print("bind object gizmo failed", oid, e)
    if add_cam:
        add_camera_component(oid)
    try:
        select_scene_object(oid)
    except Exception:
        pass
    print("added object", kind, oid, pos)
    if add_cam:
        sync_camera_tracks()
        try:
            _ensure_cv_thread()
        except Exception:
            pass
    return obj


def select_scene_object(oid, mode="replace"):
    import time as _t
    wp._obj_select_hold = _t.time() + 1.5
    if str(oid) == "sim_cursor":
        return wp.select_item("sim", None, mode=mode)
    obj = _find_object(oid)
    if obj is None:
        return False
    try:
        import robot_runtime as rr
        if rr.consume_track_pick(obj.get("name")):
            return True
    except Exception:
        pass
    return wp.select_item("obj", oid, mode=mode)


def update_scene_object(oid, name=None, pos=None, wxyz=None, scale=None, space=None, camera=None, remove_camera=False):
    obj = _find_object(oid)
    if obj is None:
        return False
    if name is not None:
        obj["name"] = str(name)
        if obj.get("kind") == "camera":
            sync_camera_tracks()
    if space is not None:
        obj["space"] = "local" if str(space).lower().startswith("loc") else "world"
    if pos is not None:
        p = np.asarray(pos, dtype=float).reshape(3)
        if obj.get("space") == "local":
            obj["local_pos"] = p
            _set_world_from_local(obj)
        else:
            obj["pos"] = p
            _set_local_from_world(obj)
    if wxyz is not None:
        q = np.asarray(wxyz, dtype=float).reshape(-1)[:4]
        n = float(np.linalg.norm(q)) or 1.0
        q = q / n
        if obj.get("space") == "local":
            obj["local_wxyz"] = q
            _set_world_from_local(obj)
        else:
            obj["wxyz"] = q
            _set_local_from_world(obj)
    if scale is not None:
        obj["scale"] = np.asarray(scale, dtype=float).reshape(3)
    if remove_camera or camera is False:
        obj["camera"] = None
        fr = obj.get("frustum")
        if fr is not None:
            try:
                fr.visible = False
            except Exception:
                pass
        try:
            sync_camera_tracks()
        except Exception:
            pass
    elif camera and isinstance(camera, dict):
        cam = dict(obj.get("camera") or {})
        cam.update(camera)
        obj["camera"] = cam
        try:
            sync_camera_tracks()
        except Exception:
            pass
    _apply_object_xform(obj)
    _refresh_camera_viz(obj)
    _propagate_children(obj.get("id"))
    return True


def delete_scene_object(oid):
    obj = _find_object(oid)
    if obj is None:
        return False
    for key in ("gizmo", "handle", "frustum"):
        h = obj.get(key)
        if h is not None:
            try:
                h.remove()
            except Exception:
                pass
        obj[key] = None
    for child in wp.scene_objects:
        if child.get("parent_id") == oid:
            child["parent_id"] = obj.get("parent_id")
            _set_local_from_world(child)
    wp.scene_objects[:] = [o for o in wp.scene_objects if o.get("id") != oid]
    if wp.selected_obj_id == oid:
        wp.selected_obj_id = None
    print("deleted object", oid)
    sync_camera_tracks()
    return True


def add_camera_component(oid=None):
    """Attach a camera component to a scene object or the sim cursor."""
    cam = {
        "type": "sim",
        "vfov": 60.0,
        "hfov": 90.0,
        "script": "",
        "model": "",
        "enabled": True,
        "manual": False,
        "recording": False,
        "cv_index": 1,
    }
    if oid in (None, "", "sim") or str(oid) == "sim":
        if wp.sim_camera:
            return {"id": "sim", "kind": "sim", "camera": dict(wp.sim_camera)}
        if not wp.sim_camera:
            wp.sim_camera = dict(cam)
        else:
            merged = dict(cam)
            merged.update(wp.sim_camera)
            wp.sim_camera = merged
        try:
            refresh_sim_camera_viz()
        except Exception:
            pass
        try:
            sync_camera_tracks()
        except Exception:
            pass
        try:
            _tick_sim_previews()
        except Exception:
            pass
        return {"id": "sim", "kind": "sim", "camera": dict(wp.sim_camera)}
    obj = _find_object(oid)
    if obj is None and wp.selected_obj_id:
        obj = _find_object(wp.selected_obj_id)
    if obj is None:
        return None
    if obj.get("camera"):
        return selected_object_props()
    existing = obj.get("camera") if isinstance(obj.get("camera"), dict) else {}
    merged = dict(cam)
    merged.update(existing or {})
    merged["enabled"] = True
    obj["camera"] = merged
    try:
        _refresh_camera_viz(obj)
    except Exception:
        pass
    try:
        sync_camera_tracks()
    except Exception:
        pass
    try:
        _ensure_cv_thread()
    except Exception:
        pass
    try:
        _tick_sim_previews()
    except Exception:
        pass
    return selected_object_props()


def selected_object_props():
    obj = _find_object(wp.selected_obj_id) if wp.selected_obj_id else None
    if obj is None:
        return None
    space = obj.get("space") or "world"
    if space == "local":
        p = obj.get("local_pos", obj["pos"])
        q = obj.get("local_wxyz", obj.get("wxyz", [1, 0, 0, 0]))
    else:
        p = obj["pos"]
        q = obj.get("wxyz", [1, 0, 0, 0])
    return {
        "kind": obj.get("kind"),
        "id": obj.get("id"),
        "name": obj.get("name"),
        "editable": True,
        "pos": [float(v) for v in np.asarray(p).reshape(3)],
        "wxyz": [float(v) for v in np.asarray(q).reshape(4)],
        "scale": [float(v) for v in np.asarray(obj.get("scale", [1, 1, 1])).reshape(3)],
        "parent_id": obj.get("parent_id"),
        "space": obj.get("space") or "world",
        "local_pos": [float(v) for v in np.asarray(obj.get("local_pos", obj["pos"])).reshape(3)],
        "local_wxyz": [float(v) for v in np.asarray(obj.get("local_wxyz", obj.get("wxyz", [1, 0, 0, 0]))).reshape(4)],
        "camera": dict(obj.get("camera")) if obj.get("camera") else None,
    }


_cam_track_handles = []
_cam_cv_mod = {}
_cam_frame_jpeg = {}
_cam_cv_thread = None
_cam_cv_stop = False
_cam_recorders = {}


def _playhead_t():
    try:
        import robot_runtime as rr
        st = getattr(rr, "_rp", None) or {}
        if st.get("active"):
            return float(st.get("t") or 0.0)
    except Exception:
        pass
    try:
        import robot_sim_run as viewer_main
        return float(getattr(viewer_main, "playhead_t", 0.0) or 0.0)
    except Exception:
        return 0.0


REC_LEAD_S = 0.5


def camera_record_at(oid, t=None, lead=None):
    """Last rec key on cam_<oid> at or before t+lead. Command 0.5s before the knot."""
    cam = wp.sim_camera if str(oid) == "sim" else (( _find_object(oid) or {}).get("camera") if oid else None)
    if isinstance(cam, dict) and cam.get("manual"):
        return bool(cam.get("recording"))
    if t is None:
        t = _playhead_t()
    if lead is None:
        lead = REC_LEAD_S
    tid = "cam_" + str(oid)
    keys = []
    for tr in list(wp.extra_tracks or []):
        if isinstance(tr, dict) and str(tr.get("id") or "") == tid:
            keys = list(tr.get("keys") or [])
            break
    keys.sort(key=lambda k: float((k or {}).get("t") or 0.0))
    rec = False
    cut = float(t) + float(lead) + 1e-9
    for k in keys:
        if float((k or {}).get("t") or 0.0) <= cut:
            rec = bool((k or {}).get("rec") or (k or {}).get("record") or (k or {}).get("v"))
        else:
            break
    return rec


def _real_playing():
    try:
        import robot_runtime as rr
        st = getattr(rr, "_rp", None) or {}
        return bool(st.get("active") and st.get("armed"))
    except Exception:
        return False


_sony_rec_on = set()


def _set_sony_rec(oid, on):
    on = bool(on)
    if on and oid in _sony_rec_on:
        return
    if (not on) and oid not in _sony_rec_on:
        return
    try:
        import sony_liveview
        ok = True
        if hasattr(sony_liveview, "set_recording"):
            ok = bool(sony_liveview.set_recording(on))
        if on and not ok:
            return
    except Exception:
        return
    if on:
        _sony_rec_on.add(oid)
    else:
        _sony_rec_on.discard(oid)


def stop_all_camera_records():
    for oid, rec in list(_cam_recorders.items()):
        try:
            rec.release()
        except Exception:
            pass
        _cam_recorders.pop(oid, None)
    for oid in list(_sony_rec_on):
        _set_sony_rec(oid, False)


def _record_bgr(oid, cam, frame_bgr):
    """Write .mp4 on the PC while real path + rec. USB Streaming cannot record on the card."""
    import cv2
    typ = str((cam or {}).get("type") or "")
    cam_rec = None
    if str(oid) == "sim":
        cam_rec = wp.sim_camera
    else:
        found = _find_object(oid)
        cam_rec = (found or {}).get("camera") if found else None
    manual = bool(isinstance(cam_rec, dict) and cam_rec.get("manual"))
    want = camera_record_at(oid) and (manual or _real_playing())
    rec = _cam_recorders.get(oid)
    if typ == "sony":
        # USB Streaming blocks gallery / in-camera movie. Still poke PTP; file is PC-side.
        try:
            _set_sony_rec(oid, want)
        except Exception:
            pass
    if not want or frame_bgr is None:
        if rec is not None:
            try:
                rec.release()
            except Exception:
                pass
            _cam_recorders.pop(oid, None)
        return
    h, w = int(frame_bgr.shape[0]), int(frame_bgr.shape[1])
    if rec is None:
        dest = app_paths.ensure_dir(app_paths.RECORDINGS_DIR)
        path = dest / ("%s_%s.mp4" % (typ or "cam", oid))
        four = cv2.VideoWriter_fourcc(*"mp4v")
        rec = cv2.VideoWriter(str(path), four, 20.0, (w, h))
        _cam_recorders[oid] = rec
        print("camera record", path)
    rec.write(frame_bgr)


def _spawn_camera_mesh(obj, name, pos, wxyz, sc):
    cam = obj.get("camera") or {}
    model = ""
    try:
        import json
        raw = json.loads(app_paths.SETTINGS_PATH.read_text(encoding="utf-8"))
        model = str((raw or {}).get("camera_urdf") or "").strip()
    except Exception:
        model = str(cam.get("model") or "").strip()
    handle = None
    if model and wp.server is not None:
        try:
            if model.lower().endswith(".stl") and hasattr(wp.server.scene, "add_mesh_simple"):
                handle = None
        except Exception:
            handle = None
    if handle is None:
        handle = wp.server.scene.add_box(
            name=name,
            dimensions=(0.06 * float(sc[0]), 0.04 * float(sc[1]), 0.03 * float(sc[2])),
            position=pos,
            wxyz=wxyz,
            color=(20, 20, 20),
        )
    obj["handle"] = handle
    _refresh_camera_viz(obj)
    return handle


def _refresh_camera_viz(obj):
    if not obj or not obj.get("camera") or wp.server is None:
        return
    if obj.get("camera", {}).get("enabled", True) is False:
        fr = obj.get("frustum")
        if fr is not None:
            try:
                fr.visible = False
            except Exception:
                pass
        return
    cam = obj.get("camera") or {}
    vfov = float(cam.get("vfov") or 60.0) * np.pi / 180.0
    hfov = float(cam.get("hfov") or 90.0) * np.pi / 180.0
    depth = 0.35
    hx = depth * np.tan(hfov * 0.5)
    hy = depth * np.tan(vfov * 0.5)
    origin = np.zeros(3)
    corners = [
        np.array([hx, hy, -depth]),
        np.array([-hx, hy, -depth]),
        np.array([-hx, -hy, -depth]),
        np.array([hx, -hy, -depth]),
    ]
    segs = []
    for c in corners:
        segs.append([origin, c])
    for i in range(4):
        segs.append([corners[i], corners[(i + 1) % 4]])
    pts = np.asarray(segs, dtype=np.float32)
    name = "/obj_%s/frustum" % obj["id"]
    pos = tuple(float(v) for v in np.asarray(obj["pos"]).reshape(3))
    wxyz = tuple(float(v) for v in np.asarray(obj.get("wxyz", [1, 0, 0, 0])).reshape(4))
    sig = (round(float(cam.get("vfov") or 60), 2), round(float(cam.get("hfov") or 90), 2))
    old = obj.get("frustum")
    if old is not None and obj.get("_frustum_sig") == sig:
        try:
            old.position = pos
            old.wxyz = wxyz
            old.visible = True
            return
        except Exception:
            try:
                old.remove()
            except Exception:
                pass
            obj["frustum"] = None
    elif old is not None:
        try:
            old.remove()
        except Exception:
            pass
        obj["frustum"] = None
    try:
        h = wp.server.scene.add_line_segments(
            name=name,
            points=pts,
            colors=(80, 200, 255),
            line_width=1.5,
        )
        h.position = pos
        h.wxyz = wxyz
        obj["frustum"] = h
        obj["_frustum_sig"] = sig
    except Exception:
        obj["frustum"] = None


def has_camera_object():
    if wp.sim_camera and wp.sim_camera.get("enabled", True):
        return True
    return any(bool(o.get("camera")) and (o.get("camera") or {}).get("enabled", True) for o in wp.scene_objects)


def _pixel_to_cam(x, y, d, w, h, hfov_deg, vfov_deg):
    nx = (float(x) / max(1.0, float(w)) - 0.5) * 2.0
    ny = (0.5 - float(y) / max(1.0, float(h))) * 2.0
    hf = float(hfov_deg) * np.pi / 180.0
    vf = float(vfov_deg) * np.pi / 180.0
    z = float(d)
    return np.array([z * np.tan(hf * 0.5) * nx, z * np.tan(vf * 0.5) * ny, -z], dtype=float)


def _update_cam_tracks(obj, tracks, frame_wh):
    global _cam_track_handles
    if wp.server is None:
        return
    pts = []
    w, hgt = frame_wh
    cam = obj.get("camera") or {}
    try:
        cpos = np.asarray(obj.get("pos"), dtype=float).reshape(3)
        cq = np.asarray(obj.get("wxyz", [1, 0, 0, 0]), dtype=float).reshape(4)
    except Exception:
        return
    for rec in list(tracks or []):
        if rec is None:
            continue
        try:
            x, y, d = float(rec[0]), float(rec[1]), float(rec[2])
        except Exception:
            continue
        if d <= 0:
            continue
        local = _pixel_to_cam(x, y, d, w, hgt, cam.get("hfov") or 90, cam.get("vfov") or 60)
        world = cpos + _qrot(cq, local)
        pts.append(world)
    while len(_cam_track_handles) > len(pts):
        h = _cam_track_handles.pop()
        try:
            h.remove()
        except Exception:
            pass
    for i, world in enumerate(pts):
        pos = tuple(float(v) for v in world)
        if i < len(_cam_track_handles):
            try:
                _cam_track_handles[i].position = pos
                continue
            except Exception:
                try:
                    _cam_track_handles[i].remove()
                except Exception:
                    pass
                _cam_track_handles[i] = None
        try:
            h = wp.server.scene.add_icosphere(
                name="/cam_track/%d" % i,
                radius=0.02,
                color=(255, 220, 40),
                position=pos,
            )
            if i < len(_cam_track_handles):
                _cam_track_handles[i] = h
            else:
                _cam_track_handles.append(h)
        except Exception as e:
            print("cam track sphere failed", e)


def refresh_sim_camera_viz():
    """Frustum on the sim cursor when it hosts a camera component."""
    dummy = {
        "id": "sim",
        "pos": None,
        "wxyz": None,
        "camera": wp.sim_camera,
        "frustum": getattr(refresh_sim_camera_viz, "_frustum", None),
    }
    if not wp.sim_camera or wp.sim_camera.get("enabled", True) is False:
        fr = dummy.get("frustum")
        if fr is not None:
            try:
                fr.visible = False
            except Exception:
                pass
        return
    try:
        import robot_runtime as rr
        pos, wxyz = rr.get_sim_cursor_pose()
    except Exception:
        return
    dummy["pos"] = pos
    dummy["wxyz"] = wxyz
    dummy["_frustum_sig"] = getattr(refresh_sim_camera_viz, "_sig", None)
    _refresh_camera_viz(dummy)
    refresh_sim_camera_viz._frustum = dummy.get("frustum")
    refresh_sim_camera_viz._sig = dummy.get("_frustum_sig")


preview_cam_id = None


def set_preview_cam(oid):
    global preview_cam_id
    preview_cam_id = None if oid in (None, "", "all") else str(oid)


def _tick_sim_previews():
    sims = []
    if wp.sim_camera and wp.sim_camera.get("enabled", True) and str(wp.sim_camera.get("type") or "sim") == "sim":
        try:
            import robot_runtime as rr
            pos, wxyz = rr.get_sim_cursor_pose()
        except Exception:
            pos, wxyz = None, None
        sims.append({"id": "sim", "pos": pos, "wxyz": wxyz, "camera": wp.sim_camera})
        try:
            refresh_sim_camera_viz()
        except Exception:
            pass
    for obj in wp.scene_objects:
        cam = obj.get("camera") or {}
        if not cam or cam.get("enabled", True) is False:
            continue
        if str(cam.get("type") or "sim") != "sim":
            continue
        sims.append(obj)
    if preview_cam_id:
        sims.sort(key=lambda o: 0 if str(o.get("id")) == str(preview_cam_id) else 1)
    if not sims:
        return
    rows = []
    for o in sims:
        if o.get("pos") is None:
            continue
        cam = o.get("camera") or {}
        rows.append((
            o.get("pos"),
            o.get("wxyz"),
            cam.get("vfov") or 60,
            cam.get("hfov") or 90,
        ))
    if not rows:
        return
    try:
        import robot_runtime as rr
        if len(rows) > 1 and hasattr(rr, "apply_scene_camera_previews"):
            rr.apply_scene_camera_previews(rows)
        elif hasattr(rr, "apply_scene_camera_preview"):
            rr.apply_scene_camera_preview(*rows[0])
    except Exception:
        pass


def last_camera_rec_t():
    """Latest time a camera rec key is ON. 0 if recording is unused."""
    tmax = 0.0
    any_on = False
    for tr in list(wp.extra_tracks or []):
        if not isinstance(tr, dict):
            continue
        if not str(tr.get("id") or "").startswith("cam_"):
            continue
        for k in tr.get("keys") or []:
            on = bool((k or {}).get("rec") or (k or {}).get("record") or (k or {}).get("v"))
            if not on:
                continue
            try:
                tmax = max(tmax, float((k or {}).get("t") or 0.0))
                any_on = True
            except Exception:
                pass
    return tmax if any_on else 0.0


def earliest_rec_on_t():
    tmin = None
    for tr in list(wp.extra_tracks or []):
        if not isinstance(tr, dict):
            continue
        if not str(tr.get("id") or "").startswith("cam_"):
            continue
        for k in tr.get("keys") or []:
            on = bool((k or {}).get("rec") or (k or {}).get("record") or (k or {}).get("v"))
            if not on:
                continue
            try:
                tk = float((k or {}).get("t") or 0.0)
            except Exception:
                continue
            tmin = tk if tmin is None else min(tmin, tk)
    return tmin


def rec_preroll_s():
    t0 = earliest_rec_on_t()
    if t0 is None:
        return 0.0
    if t0 <= 1e-6:
        return float(REC_LEAD_S)
    return 0.0


def _tick_sony_cameras():
    try:
        import sony_liveview
    except Exception:
        return
    for obj in list(wp.scene_objects):
        cam = obj.get("camera") or {}
        if not cam or cam.get("enabled", True) is False:
            continue
        if str(cam.get("type") or "sim") != "sony":
            continue
        oid = obj.get("id")
        manual = bool(cam.get("manual"))
        want = camera_record_at(oid) and (manual or _real_playing())
        if want:
            _set_sony_rec(oid, True)
        elif oid in _sony_rec_on:
            _set_sony_rec(oid, False)
        blob = None
        try:
            idx = cam.get("cv_index")
            if idx is not None:
                try:
                    idx = int(idx)
                    import os
                    os.environ["SONY_OPENCV_INDEX"] = str(idx)
                except Exception:
                    idx = None
            blob = sony_liveview.grab_jpeg(index=idx)
        except TypeError:
            try:
                blob = sony_liveview.grab_jpeg(index=idx)
            except Exception as e:
                pass
                blob = None
        except Exception:
            blob = None
        if blob:
            _cam_frame_jpeg[oid] = blob
            cam["error"] = ""
        else:
            _cam_frame_jpeg.pop(oid, None)
            cam["error"] = "Sony not connected"
            if want:
                try:
                    import cv2
                    import numpy as np
                    arr = np.frombuffer(blob, dtype=np.uint8)
                    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                    if frame is not None:
                        _record_bgr(oid, cam, frame)
                except Exception:
                    pass


def _resolve_cv_script(cam):
    import os
    path = str(cam.get("script") or "").strip()
    if path and os.path.isfile(path):
        return path
    moved = app_paths.relocate(path, app_paths.CAM_SCRIPTS_DIR, app_paths.EXAMPLE_CAM_SCRIPTS_DIR)
    if path and os.path.isfile(moved):
        cam["script"] = moved
        return moved
    here = os.path.dirname(os.path.abspath(__file__))
    for name in (os.path.join(str(app_paths.CAM_SCRIPTS_DIR), "d435i_track.py"), "d435i_track.py", os.path.join("templates", "d435i_track.py")):
        cand = os.path.join(here, name)
        if os.path.isfile(cand):
            if not path:
                cam["script"] = cand
            return cand
    return path


def _tick_cv_cameras():
    import cv2
    _tick_sony_cameras()
    for obj in list(wp.scene_objects):
        cam = obj.get("camera") or {}
        if not cam or cam.get("enabled", True) is False:
            continue
        if str(cam.get("type") or "sim") not in ("cv", "opencv"):
            continue
        path = _resolve_cv_script(cam)
        if not path:
            cam["error"] = "No OpenCV script attached"
            continue
        mod = _cam_cv_mod.get(obj["id"])
        if mod is None or getattr(mod, "_path", None) != path:
            try:
                import importlib.util
                spec = importlib.util.spec_from_file_location("user_cam_%s" % obj["id"], path)
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                mod._path = path
                _cam_cv_mod[obj["id"]] = mod
                cam["error"] = ""
            except Exception as e:
                cam["error"] = "script load failed: %s" % e
                n = int(cam.get("_load_fail", 0) or 0)
                if n < 2:
                    print("camera script load failed", path, e)
                cam["_load_fail"] = n + 1
                continue
        frame = None
        try:
            clk = cam.get("click")
            if clk and len(clk) >= 2:
                if hasattr(mod, "set_target"):
                    mod.set_target(clk[0], clk[1])
                else:
                    mod.target_uv = (clk[0], clk[1])
            if hasattr(mod, "get_frame"):
                frame = mod.get_frame()
        except Exception as e:
            n = int(getattr(mod, "_gf_fail", 0) or 0) + 1
            try:
                mod._gf_fail = n
            except Exception:
                pass
            if n <= 2:
                print("camera get_frame failed", e)
            cam["error"] = "get_frame failed: %s" % e
            frame = None
        err = str(getattr(mod, "last_error", "") or cam.get("error") or "")
        w, hgt = 640, 480
        if frame is not None:
            try:
                arr = np.asarray(frame)
                if arr.ndim == 2:
                    arr = np.stack([arr, arr, arr], axis=-1)
                ok, buf = cv2.imencode(".jpg", arr, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
                if ok:
                    _cam_frame_jpeg[obj["id"]] = bytes(buf)
                    cam["error"] = ""
                hgt, w = int(arr.shape[0]), int(arr.shape[1])
                _record_bgr(obj["id"], cam, arr)
            except Exception as e:
                cam["error"] = "jpeg failed: %s" % e
                if not cam.get("_jpeg_logged"):
                    print("camera jpeg failed", e)
                    cam["_jpeg_logged"] = True
        elif err:
            cam["error"] = err
        tracks = list(getattr(mod, "tracked_cam_pos", None) or [])
        _update_cam_tracks(obj, tracks, (w, hgt))


def _ensure_cv_thread():
    global _cam_cv_thread, _cam_cv_stop
    import threading
    if _cam_cv_thread is not None and _cam_cv_thread.is_alive():
        return
    _cam_cv_stop = False

    def _loop():
        import time
        while not _cam_cv_stop:
            t0 = time.time()
            try:
                _tick_cv_cameras()
            except Exception as e:
                print("cv camera loop", e)
            dt = time.time() - t0
            time.sleep(max(0.0, 0.04 - dt))

    _cam_cv_thread = threading.Thread(target=_loop, name="cam-cv", daemon=True)
    _cam_cv_thread.start()


def tick_cameras():
    _tick_sim_previews()
    _ensure_cv_thread()


def set_camera_click(oid, u, v):
    for obj in wp.scene_objects:
        if obj.get("id") != oid:
            continue
        cam = obj.get("camera")
        if not isinstance(cam, dict):
            cam = {}
            obj["camera"] = cam
        cam["click"] = [float(u), float(v)]
        return True
    return False


def camera_status():
    cams = [o for o in wp.scene_objects if o.get("camera") and (o.get("camera") or {}).get("enabled", True)]
    if wp.sim_camera and wp.sim_camera.get("enabled", True):
        cams = [{"id": "sim", "name": "Sim cursor", "camera": wp.sim_camera}] + cams
    if not cams:
        return {"active": False}
    obj = None
    for o in cams:
        if o.get("id") == wp.selected_obj_id:
            obj = o
            break
    if obj is None:
        obj = cams[0]
    cam = obj.get("camera") or {}
    typ = str(cam.get("type") or "sim")
    frame = _cam_frame_jpeg.get(obj.get("id"))
    sony = {}
    if typ == "sony":
        try:
            import sony_liveview
            sony = sony_liveview.status()
        except Exception:
            sony = {}
    return {
        "active": True,
        "id": obj.get("id"),
        "type": typ,
        "connected": bool(frame) if typ in ("cv", "opencv", "sony") else typ == "sim",
        "has_frame": bool(frame),
        "error": str(cam.get("error") or ""),
        "script": cam.get("script") or "",
        "sony": sony,
        "vfov": float(cam.get("vfov") or 60),
        "recording": bool(typ == "sony" and camera_record_at(obj.get("id")) and _real_playing()),
        "rec_want": bool(camera_record_at(obj.get("id"))),
        "rec_real": bool(_real_playing()),
        "rec_t": float(_playhead_t()),
        "rec_on": bool(obj.get("id") in _sony_rec_on),
        "hfov": float(cam.get("hfov") or 90),
        "pos": [float(v) for v in np.asarray(obj.get("pos", [0, 0, 0])).reshape(3)],
        "wxyz": [float(v) for v in np.asarray(obj.get("wxyz", [1, 0, 0, 0])).reshape(4)],
    }


# Expose this module's functions on waypoints (wp.add_scene_object, ...).
for _name, _obj in list(globals().items()):
    if callable(_obj) and getattr(_obj, "__module__", None) == __name__:
        setattr(wp, _name, _obj)
