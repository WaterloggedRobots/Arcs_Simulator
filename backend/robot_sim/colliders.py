"""Collider component: sphere / cube collider boxes on scene objects, and one
robot-shaped collider on the sim cursor.

Scene objects hold a list of collider boxes, each a local transform relative
to the object's origin plus a full size in meters (independent of the
object's scale). The sim cursor's collider is the whole sim robot: one convex
hull per URDF link, grown by a margin. Its shape and size are fixed.

Data lives on the owner so undo snapshots and saves carry it:
  scene object -> obj["collider"]
      {"enabled": bool, "stop": bool,
       "items": [{"shape": "cube"|"sphere", "pos": [x, y, z],
                  "wxyz": [w, x, y, z], "scale": [sx, sy, sz]}]}
  sim cursor   -> wp.sim_collider
      {"enabled": bool, "stop": bool, "margin": float}

Green outlines are drawn only while the Collider panel is clicked (focused);
selecting the object again hides them. During simulation, owners whose
colliders touch another owner's are tinted red, and a new contact pauses the
simulation when either side has "stop simulation if collide" ticked.
"""
import functools
import threading

import numpy as np

import waypoints as wp

OUTLINE_RGB = (60, 220, 90)
HIT_RGB = (230, 40, 40)
SHAPES = ("cube", "sphere")
MIN_SIZE = 1e-3

_lock = threading.RLock()
_focus_owner = None  # owner whose Collider panel was clicked
_viz = {}            # owner -> {"sig": ..., "handles": [(handle, kind, ref)]}
_viz_seq = 0
_robot_geom = {"model_id": None, "links": []}
_tinted = {}         # owner -> original rgb (objects) / True (robot)
_contacts = set()    # frozenset({owner_a, owner_b}) touching on the last tick
_status = {"hits": [], "pairs": [], "paused_by": None, "pause_seq": 0}


def _locked(fn):
    @functools.wraps(fn)
    def inner(*a, **k):
        with _lock:
            return fn(*a, **k)
    return inner


def _sv():
    import simulatorview
    return simulatorview


def _rr():
    import robot_runtime
    return robot_runtime


def _norm_owner(owner):
    o = str(owner or "")
    return "sim" if o in ("sim", "sim_cursor") else o


def _vec(v, n, fill):
    a = np.asarray(v if v is not None else fill, dtype=float).reshape(-1)
    if a.size < n:
        a = np.concatenate([a, np.asarray(fill, dtype=float).reshape(-1)[a.size:n]])
    return a[:n]


def _unit_q(q):
    q = _vec(q, 4, [1, 0, 0, 0])
    n = float(np.linalg.norm(q)) or 1.0
    return q / n


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------
def clean_item(rec):
    rec = rec if isinstance(rec, dict) else {}
    shape = str(rec.get("shape") or "cube").lower()
    if shape not in SHAPES:
        shape = "cube"
    sc = np.abs(_vec(rec.get("scale"), 3, [0.1, 0.1, 0.1]))
    sc = np.maximum(sc, MIN_SIZE)
    if shape == "sphere":
        sc = np.full(3, float(np.max(sc)))
    return {
        "shape": shape,
        "pos": [float(v) for v in _vec(rec.get("pos"), 3, [0, 0, 0])],
        "wxyz": [float(v) for v in _unit_q(rec.get("wxyz"))],
        "scale": [float(v) for v in sc],
    }


def clean_collider(rec):
    """JSON-safe copy of a scene-object collider record, or None."""
    if not isinstance(rec, dict):
        return None
    return {
        "enabled": bool(rec.get("enabled", True)),
        "stop": bool(rec.get("stop", False)),
        "items": [clean_item(it) for it in (rec.get("items") or []) if isinstance(it, dict)],
    }


def clean_sim_collider(rec):
    """JSON-safe copy of the sim cursor's collider record, or None."""
    if not isinstance(rec, dict):
        return None
    try:
        margin = max(0.0, float(rec.get("margin") or 0.0))
    except (TypeError, ValueError):
        margin = 0.0
    return {
        "enabled": bool(rec.get("enabled", True)),
        "stop": bool(rec.get("stop", False)),
        "margin": margin,
    }


def get_collider(owner):
    owner = _norm_owner(owner)
    if owner == "sim":
        return getattr(wp, "sim_collider", None)
    obj = _sv()._find_object(owner)
    return None if obj is None else obj.get("collider")


def _set_record(owner, rec):
    owner = _norm_owner(owner)
    if owner == "sim":
        wp.sim_collider = rec
        return True
    obj = _sv()._find_object(owner)
    if obj is None:
        return False
    obj["collider"] = rec
    return True


def collider_status(owner):
    """Props-panel payload."""
    owner = _norm_owner(owner)
    rec = get_collider(owner)
    if not rec:
        return {"added": False, "enabled": False}
    out = clean_sim_collider(rec) if owner == "sim" else clean_collider(rec)
    out["added"] = True
    out["robot"] = owner == "sim"
    out["focused"] = _focus_owner == owner
    out["hit"] = owner in _status["hits"]
    return out


def default_item(obj):
    """A collider box that wraps the object as it is now."""
    sv = _sv()
    kind = str((obj or {}).get("kind") or "empty").lower()
    sc = np.abs(_vec((obj or {}).get("scale"), 3, [1, 1, 1]))
    h = (obj or {}).get("handle")
    item = {"shape": "cube", "pos": [0.0, 0.0, 0.0], "wxyz": [1.0, 0.0, 0.0, 0.0], "scale": [0.1, 0.1, 0.1]}
    try:
        verts = (obj or {}).get("_mesh_verts")
        if sv._is_mesh_kind(kind) and verts is not None and len(verts):
            v = np.asarray(verts, dtype=float).reshape(-1, 3) * sc
            lo, hi = v.min(axis=0), v.max(axis=0)
            item["pos"] = [float(x) for x in 0.5 * (lo + hi)]
            item["scale"] = [float(x) for x in (hi - lo)]
        elif h is not None and hasattr(h, "radius") and kind in ("sphere", "empty"):
            d = 2.0 * float(h.radius)
            item["shape"] = "sphere"
            item["scale"] = [d, d, d]
        elif kind in ("sphere", "empty"):
            d = 2.0 * (0.02 if kind == "empty" else 0.05) * max(1e-4, float(np.mean(sc)))
            item["shape"] = "sphere"
            item["scale"] = [d, d, d]
        elif h is not None and hasattr(h, "dimensions"):
            item["scale"] = [float(x) for x in np.asarray(h.dimensions, dtype=float).reshape(3)]
        else:
            item["scale"] = [float(x) for x in 0.1 * sc]
    except Exception as e:
        print("collider default size failed", e)
    return clean_item(item)


@_locked
def add_collider_component(owner):
    owner = _norm_owner(owner)
    if get_collider(owner):
        return collider_status(owner)
    if owner == "sim":
        rec = {"enabled": True, "stop": False, "margin": 0.0}
    else:
        obj = _sv()._find_object(owner)
        if obj is None:
            return None
        wp._push_history()
        rec = {"enabled": True, "stop": False, "items": [default_item(obj)]}
    if not _set_record(owner, rec):
        return None
    _sv().note_component(owner, "collider")
    _refresh_focus_viz()
    return collider_status(owner)


def _truthy(v):
    return v is True or str(v).lower() in ("1", "true", "on", "yes")


@_locked
def update_collider(owner, act="update", enabled=None, stop=None, margin=None,
                    index=None, shape=None, pos=None, wxyz=None, scale=None):
    owner = _norm_owner(owner)
    rec = get_collider(owner)
    act = str(act or "update").lower()
    if act == "remove":
        if rec and owner != "sim":
            wp._push_history()
        _set_record(owner, None)
        if _focus_owner == owner:
            end_collider_edit()
        _untint(owner)
        _drop_viz(owner)
        return collider_status(owner)
    if not rec:
        return collider_status(owner)
    if owner != "sim":
        wp._push_history()
    rec = dict(rec)
    if enabled is not None:
        rec["enabled"] = _truthy(enabled)
    if stop is not None:
        rec["stop"] = _truthy(stop)
    if owner == "sim":
        # The robot collider's shape and size come from the URDF; only the margin changes.
        if margin is not None:
            rec["margin"] = margin
        _set_record(owner, clean_sim_collider(rec))
        _refresh_focus_viz()
        return collider_status(owner)
    items = [dict(it) for it in (rec.get("items") or [])]
    if act == "add_item":
        obj = _sv()._find_object(owner)
        items.append(default_item(obj))
    elif act in ("remove_item", "item"):
        try:
            i = int(index)
        except (TypeError, ValueError):
            i = -1
        if 0 <= i < len(items):
            if act == "remove_item":
                items.pop(i)
            else:
                it = items[i]
                if shape is not None:
                    new_shape = str(shape).lower()
                    if new_shape == "sphere" and it.get("shape") != "sphere":
                        d = float(np.max(np.abs(_vec(it.get("scale"), 3, [0.1] * 3))))
                        it["scale"] = [d, d, d]
                    it["shape"] = new_shape
                if pos is not None:
                    it["pos"] = pos
                if wxyz is not None:
                    it["wxyz"] = wxyz
                if scale is not None:
                    it["scale"] = scale
                items[i] = clean_item(it)
    rec["items"] = items
    _set_record(owner, clean_collider(rec))
    _refresh_focus_viz()
    return collider_status(owner)


# ---------------------------------------------------------------------------
# World-space shapes
# ---------------------------------------------------------------------------
def _qmat(q):
    w, x, y, z = _unit_q(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def _owner_pose(owner):
    if owner == "sim":
        return None
    obj = _sv()._find_object(owner)
    if obj is None:
        return None
    return _vec(obj.get("pos"), 3, [0, 0, 0]), _unit_q(obj.get("wxyz"))


def _item_world(owner_pos, owner_q, it):
    """World center, rotation matrix and quaternion of one collider box."""
    sv = _sv()
    oq = _unit_q(owner_q)
    c = np.asarray(owner_pos, dtype=float).reshape(3) + sv._qrot(oq, _vec(it.get("pos"), 3, [0, 0, 0]))
    q = _unit_q(sv._qmul(oq, _unit_q(it.get("wxyz"))))
    return c, _qmat(q), q


def _object_shapes(owner, rec):
    pose = _owner_pose(owner)
    if pose is None:
        return []
    out = []
    for it in rec.get("items") or []:
        c, Rm, _q = _item_world(pose[0], pose[1], it)
        sc = np.abs(_vec(it.get("scale"), 3, [0.1] * 3))
        if it.get("shape") == "sphere":
            r = 0.5 * float(np.max(sc))
            out.append({"t": "sphere", "c": c, "r": r, "br": r})
        else:
            h = 0.5 * sc
            out.append({"t": "box", "c": c, "R": Rm, "h": h, "br": float(np.linalg.norm(h))})
    return out


def _link_mesh(link):
    import trimesh
    # URDF collision geometry first: the i5 forearm visual mesh carries a stray
    # part metres away that would blow its hull up to fill the scene.
    for attr in ("collision_mesh", "concatenated_visual_mesh", "visual_mesh"):
        m = getattr(link, attr, None)
        if m is None:
            continue
        if isinstance(m, (list, tuple)):
            parts = [g for g in m if hasattr(g, "vertices") and len(g.vertices)]
            if not parts:
                continue
            m = trimesh.util.concatenate(parts)
        if hasattr(m, "vertices") and len(m.vertices) >= 4:
            return m
    return None


def _robot_links():
    """Convex hull per sim-robot link, in link-local coordinates. Rebuilt
    when the robot model changes."""
    rr = _rr()
    model = getattr(rr, "robot_sim", None)
    if model is None:
        return []
    if _robot_geom["model_id"] == id(model):
        return _robot_geom["links"]
    links = []
    try:
        import trimesh
    except Exception as e:
        print("collider: trimesh missing", e)
        return []
    for link in getattr(model, "link_list", []) or []:
        mesh = _link_mesh(link)
        if mesh is None:
            continue
        try:
            hull = trimesh.convex.convex_hull(np.asarray(mesh.vertices, dtype=float))
        except Exception as e:
            print("collider hull failed", getattr(link, "name", "?"), e)
            continue
        V = np.asarray(hull.vertices, dtype=float)
        fn = np.asarray(hull.face_normals, dtype=float)
        N = np.unique(np.round(fn, 4), axis=0)
        N = N / np.maximum(np.linalg.norm(N, axis=1, keepdims=True), 1e-12)
        D = np.max(V @ N.T, axis=0)
        try:
            sharp = np.asarray(hull.face_adjacency_angles) > 0.08
            edges = np.asarray(hull.face_adjacency_edges)[sharp]
        except Exception:
            edges = np.asarray(hull.edges_unique)
        try:
            vn = np.asarray(hull.vertex_normals, dtype=float)
        except Exception:
            vn = V - V.mean(axis=0)
            vn = vn / np.maximum(np.linalg.norm(vn, axis=1, keepdims=True), 1e-12)
        links.append({"link": link, "V": V, "N": N, "D": D, "edges": edges, "vn": vn,
                      "name": getattr(link, "name", "")})
    _robot_geom["model_id"] = id(model)
    _robot_geom["links"] = links
    return links


def _link_pose(link):
    p = np.asarray(link.worldpos(), dtype=float).reshape(3)
    Rm = np.asarray(link.worldrot(), dtype=float).reshape(3, 3)
    return p, Rm


def _robot_shapes(rec):
    m = float(rec.get("margin") or 0.0)
    out = []
    for g in _robot_links():
        try:
            p, Rm = _link_pose(g["link"])
        except Exception:
            continue
        V = g["V"] @ Rm.T + p
        N = g["N"] @ Rm.T
        D = g["D"] + N @ p
        c = V.mean(axis=0)
        br = float(np.max(np.linalg.norm(V - c, axis=1))) + m
        out.append({"t": "hull", "V": V, "N": N, "D": D, "m": m, "c": c, "br": br})
    return out


def _owner_shapes(owner):
    rec = get_collider(owner)
    if not rec or not rec.get("enabled", True):
        return []
    if owner == "sim":
        return _robot_shapes(rec)
    return _object_shapes(owner, rec)


# ---------------------------------------------------------------------------
# Overlap tests
# ---------------------------------------------------------------------------
def _sphere_box(s, b):
    d = b["R"].T @ (s["c"] - b["c"])
    q = np.clip(d, -b["h"], b["h"])
    return float(np.linalg.norm(d - q)) <= s["r"]


def _box_box(a, b):
    Ra, Rb = a["R"], b["R"]
    t = b["c"] - a["c"]
    axes = [Ra[:, i] for i in range(3)] + [Rb[:, i] for i in range(3)]
    for i in range(3):
        for j in range(3):
            ax = np.cross(Ra[:, i], Rb[:, j])
            if float(np.linalg.norm(ax)) > 1e-9:
                axes.append(ax)
    for ax in axes:
        ra = float(np.sum(a["h"] * np.abs(Ra.T @ ax)))
        rb = float(np.sum(b["h"] * np.abs(Rb.T @ ax)))
        if abs(float(t @ ax)) > ra + rb:
            return False
    return True


def _hull_sphere(h, s):
    # Distance to the farthest face plane: exact inside the hull, and a little
    # generous past edges and corners, which is fine for a sandbox check.
    return float(np.max(h["N"] @ s["c"] - h["D"])) <= s["r"] + h["m"]


def _hull_box(h, b):
    Rb = b["R"]
    axes = np.vstack([h["N"], Rb.T])
    proj = h["V"] @ axes.T
    lo = proj.min(axis=0) - h["m"]
    hi = proj.max(axis=0) + h["m"]
    bc = axes @ b["c"]
    br = np.abs(axes @ Rb) @ b["h"]
    return not bool(np.any((bc + br < lo) | (bc - br > hi)))


def shapes_touch(a, b):
    ca = a.get("c")
    cb = b.get("c")
    if ca is not None and cb is not None:
        if float(np.linalg.norm(ca - cb)) > a["br"] + b["br"]:
            return False
    rank = {"sphere": 0, "box": 1, "hull": 2}
    if rank[a["t"]] > rank[b["t"]]:
        a, b = b, a
    ta, tb = a["t"], b["t"]
    if ta == "sphere" and tb == "sphere":
        return float(np.linalg.norm(a["c"] - b["c"])) <= a["r"] + b["r"]
    if ta == "sphere" and tb == "box":
        return _sphere_box(a, b)
    if ta == "box" and tb == "box":
        return _box_box(a, b)
    if ta == "sphere" and tb == "hull":
        return _hull_sphere(b, a)
    if ta == "box" and tb == "hull":
        return _hull_box(b, a)
    return False


def _owners():
    out = []
    if getattr(wp, "sim_collider", None):
        out.append("sim")
    for obj in getattr(wp, "scene_objects", None) or []:
        if obj.get("collider"):
            out.append(str(obj.get("id")))
    return out


def find_contacts():
    """Pairs of owners whose colliders overlap. Boxes of the same owner never
    count against each other."""
    shapes = {o: _owner_shapes(o) for o in _owners()}
    owners = [o for o in shapes if shapes[o]]
    pairs = set()
    for i, a in enumerate(owners):
        for b in owners[i + 1:]:
            hit = False
            for sa in shapes[a]:
                for sb in shapes[b]:
                    if shapes_touch(sa, sb):
                        hit = True
                        break
                if hit:
                    break
            if hit:
                pairs.add(frozenset((a, b)))
    return pairs


# ---------------------------------------------------------------------------
# Red tint on colliding models
# ---------------------------------------------------------------------------
def _robot_handles():
    rr = _rr()
    viewer = getattr(rr, "viewer", None)
    model = getattr(rr, "robot_sim", None)
    handles = getattr(viewer, "_linkid_to_handle", None) or {}
    if model is None:
        return []
    out = []
    for link in getattr(model, "link_list", []) or []:
        h = handles.get(str(id(link)))
        if h is not None:
            out.append(h)
    return out


def _tint(owner):
    if owner in _tinted:
        return
    if owner == "sim":
        for h in _robot_handles():
            try:
                h.color = HIT_RGB
            except Exception:
                pass
        _tinted[owner] = True
        return
    obj = _sv()._find_object(owner)
    h = None if obj is None else obj.get("handle")
    if h is None:
        return
    try:
        base = tuple(int(v) for v in np.asarray(h.color).reshape(-1)[:3])
    except Exception:
        base = (220, 220, 220)
    try:
        h.color = HIT_RGB
    except Exception:
        return
    _tinted[owner] = base


def _untint(owner):
    base = _tinted.pop(owner, None)
    if base is None:
        return
    if owner == "sim":
        rgb = tuple(getattr(_rr(), "SIM_RGB", (90, 150, 230)))
        for h in _robot_handles():
            try:
                h.color = rgb
            except Exception:
                pass
        return
    obj = _sv()._find_object(owner)
    h = None if obj is None else obj.get("handle")
    if h is None:
        return
    try:
        h.color = base
    except Exception:
        pass


def _clear_hits():
    for owner in list(_tinted.keys()):
        _untint(owner)
    _contacts.clear()
    _status["hits"] = []
    _status["pairs"] = []


def _owner_name(owner):
    if owner == "sim":
        return getattr(wp, "SIM_CURSOR_NAME", "Sim cursor")
    obj = _sv()._find_object(owner)
    return str((obj or {}).get("name") or owner)


@_locked
def tick_collisions():
    """Viewer-loop tick: test colliders while a simulation runs or is paused."""
    rr = _rr()
    pb = getattr(rr, "_pb", {}) or {}
    if not pb.get("active"):
        if _tinted or _contacts:
            _clear_hits()
        _status["paused_by"] = None
        return
    try:
        pairs = find_contacts()
    except Exception as e:
        print("collision check failed", e)
        return
    hits = set()
    for p in pairs:
        hits |= set(p)
    for owner in list(_tinted.keys()):
        if owner not in hits:
            _untint(owner)
    for owner in hits:
        _tint(owner)
    new = pairs - _contacts
    _contacts.clear()
    _contacts.update(pairs)
    _status["hits"] = sorted(hits)
    _status["pairs"] = [sorted(_owner_name(o) for o in p) for p in pairs]
    # Pause on a new contact only, so Resume plays on while the same two
    # stay touching instead of pausing again straight away.
    if new and not pb.get("paused"):
        for p in new:
            if any((get_collider(o) or {}).get("stop") for o in p):
                rr.toggle_sim_pause()
                _status["paused_by"] = sorted(_owner_name(o) for o in p)
                _status["pause_seq"] += 1
                print("collision: paused on", _status["paused_by"])
                break


def collision_status():
    return {
        "hits": list(_status["hits"]),
        "pairs": [list(p) for p in _status["pairs"]],
        "paused_by": _status["paused_by"],
        "pause_seq": int(_status["pause_seq"]),
    }


# ---------------------------------------------------------------------------
# Green outlines (only while the Collider panel is focused)
# ---------------------------------------------------------------------------
def _box_edges(h):
    hx, hy, hz = [float(v) for v in h]
    c = np.array([[sx * hx, sy * hy, sz * hz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
    segs = []
    for i in range(8):
        for j in range(i + 1, 8):
            if int(np.sum(np.abs(c[i] - c[j]) > 1e-12)) == 1:
                segs.append([c[i], c[j]])
    return np.asarray(segs, dtype=np.float32)


def _sphere_edges(r, n=40):
    a = np.linspace(0, 2 * np.pi, n + 1)
    ca, sa = np.cos(a) * r, np.sin(a) * r
    z = np.zeros_like(a)
    rings = [np.stack([ca, sa, z], 1), np.stack([ca, z, sa], 1), np.stack([z, ca, sa], 1)]
    segs = []
    for ring in rings:
        segs.extend([[ring[k], ring[k + 1]] for k in range(n)])
    return np.asarray(segs, dtype=np.float32)


def _drop_viz(owner):
    v = _viz.pop(owner, None)
    if not v:
        return
    for h, _kind, _ref in v.get("handles") or []:
        try:
            h.remove()
        except Exception:
            pass


def _viz_sig(owner):
    rec = get_collider(owner)
    if not rec or not rec.get("enabled", True):
        return None
    if owner == "sim":
        return ("sim", round(float(rec.get("margin") or 0.0), 6), _robot_geom.get("model_id"))
    return tuple((it.get("shape"), tuple(np.round(it.get("scale"), 6))) for it in rec.get("items") or [])


def _add_lines(name, segs):
    return wp.server.scene.add_line_segments(
        name=name, points=segs, colors=OUTLINE_RGB, line_width=2.0)


def _make_viz(owner):
    global _viz_seq
    rec = get_collider(owner)
    if wp.server is None or not rec or not rec.get("enabled", True):
        return None
    _viz_seq += 1
    base = "/collider_%s_%d" % (owner, _viz_seq)
    handles = []
    try:
        if owner == "sim":
            m = float(rec.get("margin") or 0.0)
            for k, g in enumerate(_robot_links()):
                V = g["V"] + g["vn"] * m
                e = np.asarray(g["edges"], dtype=int)
                if not len(e):
                    continue
                segs = np.stack([V[e[:, 0]], V[e[:, 1]]], axis=1).astype(np.float32)
                handles.append((_add_lines("%s/l%d" % (base, k), segs), "link", g["link"]))
        else:
            for k, it in enumerate(rec.get("items") or []):
                sc = np.abs(_vec(it.get("scale"), 3, [0.1] * 3))
                segs = _sphere_edges(0.5 * float(np.max(sc))) if it.get("shape") == "sphere" else _box_edges(0.5 * sc)
                handles.append((_add_lines("%s/c%d" % (base, k), segs), "item", k))
    except Exception as e:
        print("collider outline failed", e)
    v = {"sig": _viz_sig(owner), "handles": handles}
    _viz[owner] = v
    return v


def _pose_key(owner):
    try:
        if owner == "sim":
            return tuple(np.round(np.asarray(_rr().robot_sim.angle_vector(), dtype=float).reshape(-1), 6))
        pose = _owner_pose(owner)
        if pose is None:
            return None
        # Item edits move outlines too, not just the owner.
        items = tuple((tuple(it.get("pos") or ()), tuple(it.get("wxyz") or ()))
                      for it in (get_collider(owner) or {}).get("items") or [])
        return tuple(np.round(np.concatenate(pose), 6)), items
    except Exception:
        return None


def _place_viz(owner, v):
    # Viser sends every assignment, so only move outlines when the owner moved.
    key = _pose_key(owner)
    if key is not None and key == v.get("last"):
        return
    v["last"] = key
    rec = get_collider(owner) or {}
    if owner == "sim":
        from scipy.spatial.transform import Rotation as R
        for h, _kind, link in v["handles"]:
            try:
                p, Rm = _link_pose(link)
                xyzw = R.from_matrix(Rm).as_quat()
                h.position = tuple(float(x) for x in p)
                h.wxyz = (float(xyzw[3]), float(xyzw[0]), float(xyzw[1]), float(xyzw[2]))
            except Exception:
                pass
        return
    pose = _owner_pose(owner)
    if pose is None:
        return
    items = rec.get("items") or []
    for h, _kind, k in v["handles"]:
        if k >= len(items):
            continue
        c, _Rm, q = _item_world(pose[0], pose[1], items[k])
        try:
            h.position = tuple(float(x) for x in c)
            h.wxyz = tuple(float(x) for x in q)
        except Exception:
            pass


def _refresh_focus_viz():
    owner = _focus_owner
    for o in list(_viz.keys()):
        if o != owner:
            _drop_viz(o)
    if owner is None or wp.server is None:
        return
    if owner == "sim":
        _robot_links()
    sig = _viz_sig(owner)
    v = _viz.get(owner)
    if sig is None:
        _drop_viz(owner)
        return
    if v is None or v.get("sig") != sig:
        _drop_viz(owner)
        v = _make_viz(owner)
    if v is not None:
        _place_viz(owner, v)


def _show_owner_gizmo(owner, on):
    try:
        import offsets
        offsets._show_owner_gizmo(owner, on)
    except Exception:
        pass


@_locked
def begin_collider_edit(owner):
    """Collider panel clicked: draw this owner's outlines, park its gizmo."""
    global _focus_owner
    owner = _norm_owner(owner)
    if not get_collider(owner):
        return False
    try:
        import offsets
        offsets.end_offset_edit()
    except Exception:
        pass
    if _focus_owner and _focus_owner != owner:
        end_collider_edit()
    _focus_owner = owner
    _show_owner_gizmo(owner, False)
    _refresh_focus_viz()
    return True


@_locked
def end_collider_edit():
    global _focus_owner
    owner = _focus_owner
    _focus_owner = None
    for o in list(_viz.keys()):
        _drop_viz(o)
    if owner is None:
        return
    try:
        import offsets
        if owner in offsets._selected_owners():
            _show_owner_gizmo(owner, True)
    except Exception:
        pass


@_locked
def on_selection_changed():
    """Selecting the object (or anything else) hides its collider outlines."""
    end_collider_edit()


@_locked
def forget_owner(owner):
    """Object being deleted: drop its outlines, tint and contacts only."""
    global _focus_owner
    owner = _norm_owner(owner)
    if _focus_owner == owner:
        _focus_owner = None
    _drop_viz(owner)
    _tinted.pop(owner, None)
    for p in [p for p in _contacts if owner in p]:
        _contacts.discard(p)


@_locked
def clear_collider_viz():
    global _focus_owner
    _focus_owner = None
    for o in list(_viz.keys()):
        _drop_viz(o)
    _clear_hits()


@_locked
def tick_colliders():
    """Viewer-loop tick: keep outlines on their owner, then test contacts."""
    if _focus_owner is not None:
        if not get_collider(_focus_owner):
            end_collider_edit()
        else:
            _refresh_focus_viz()
    tick_collisions()
