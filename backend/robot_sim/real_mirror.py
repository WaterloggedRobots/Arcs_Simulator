"""The real side of each Aubo Arm (README: "Real side").

Everything that rides on an arm's Sim Arm Cursor (its children, at any depth)
is mirrored onto the arm's Real Arm Cursor: each object keeps, relative to the
real cursor, the transform it has relative to the sim cursor. While the robot
is not connected the real side hangs on a stand-in, "Robot Not Connected",
placed on the Sim Arm Cursor, so the two sides overlap.

- Objects with a Sim / Real pair (a gimbal) put their Real cursor there: the
  Real Gimbal Cursor sits on the mirrored mount. Children of a Sim Gimbal
  Cursor mirror onto the Real Gimbal Cursor's camera frame (so they follow the
  angles read from the real gimbal).
- Other objects get a ghost: the same shape, more see-through, transform only
  (read-only in Properties). Preferences can turn ghosts off; the real
  cursors stay.

tick() runs on the viewer thread (robot_sim_run) and places the nodes; view()
is the last result, for /api/status (the object list builds the rows from it).
"""
import numpy as np

import arms
import waypoints as wp

GHOST_OPACITY = 0.5   # times the object's own opacity ("50 % more transparent")
NOT_CONNECTED_NAME = "Robot Not Connected"
GHOST_PREFIX = "ghost:"

_settings = {"ghosts": True}
_ghosts = {}          # ghost id -> {"handle": node, "sig": tuple}
_view = {"arms": [], "ghosts": True}


def set_ghosts(on):
    _settings["ghosts"] = bool(on)


def ghosts_on():
    return bool(_settings["ghosts"])


def none_id(aid):
    """Row id of an arm's "Robot Not Connected" stand-in."""
    return "real_none" if aid == arms.DEFAULT_ID else "real_none@%s" % aid


def ghost_id(oid, aid):
    return "%s%s@%s" % (GHOST_PREFIX, oid, aid)


# ---------------------------------------------------------------------------
# Transforms: (pos (3,), wxyz (4,))
# ---------------------------------------------------------------------------
def _qmul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


def _qinv(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])


def _qrot(q, v):
    return _qmul(_qmul(q, np.array([0.0, v[0], v[1], v[2]])), _qinv(q))[1:]


def _xf(pos, wxyz):
    p = np.asarray(pos if pos is not None else [0, 0, 0], dtype=float).reshape(3)
    q = np.asarray(wxyz if wxyz is not None else [1, 0, 0, 0], dtype=float).reshape(4)
    return p, q / (np.linalg.norm(q) or 1.0)


def _compose(a, b):
    """a then b (b given in a's frame)."""
    return a[0] + _qrot(a[1], b[0]), _qmul(a[1], b[1]) / (np.linalg.norm(_qmul(a[1], b[1])) or 1.0)


def _relative(parent, child):
    """child in parent's frame."""
    inv = _qinv(parent[1])
    return _qrot(inv, child[0] - parent[0]), _qmul(inv, child[1])


# ---------------------------------------------------------------------------
# The mirror
# ---------------------------------------------------------------------------
def _gimbal():
    import gimbal
    return gimbal


def _children(pid):
    pid = str(pid)
    return [o for o in wp.scene_objects if isinstance(o, dict) and str(o.get("parent_id") or "") == pid]


def _real_cam_frame(obj, mount_xf):
    """The Real Gimbal Cursor's camera frame with its handle at mount_xf
    (angles read; the Sim target while nothing reads)."""
    g = _gimbal()
    import gimbal_model as gmodel
    model = obj.get("_model_used") or g.model_of(obj)[0]
    rt = g._rt.get(str(obj.get("id"))) or {}
    angles = rt.get("read") if g.real_ok(obj.get("id")) else None
    if angles is None:
        angles = (obj.get("gimbal") or {}).get("target")
    q, p = gmodel.camera_pose(model, gmodel.joint_values(model, angles))
    return _compose(mount_xf, (np.asarray(p, float), np.asarray(q, float)))


def _walk(aid, pid, sim_parent, real_parent, row_parent, rows, ghosts, gimbals, depth=0):
    if depth > 32:
        return
    for obj in _children(pid):
        oid = str(obj.get("id"))
        sim_w = _xf(obj.get("pos"), obj.get("wxyz"))
        local = _relative(sim_parent, sim_w)
        mirror = _compose(real_parent, local)
        gid = ghost_id(oid, aid)
        show_ghost = ghosts_on()
        row = {"id": gid, "name": obj.get("name") or obj.get("kind") or oid, "parent_id": row_parent,
               "ghost": True, "src": oid, "kind": obj.get("kind"),
               "pos": [float(v) for v in mirror[0]], "wxyz": [float(v) for v in mirror[1]],
               "local_pos": [float(v) for v in local[0]], "local_wxyz": [float(v) for v in local[1]]}
        if show_ghost:
            rows.append(row)
            ghosts.append((gid, obj, mirror))
        child_parent = gid if show_ghost else row_parent
        if obj.get("kind") == "gimbal":
            g = _gimbal()
            gimbals[oid] = mirror
            twin = g.part_id(oid, "real")
            rows.append({"id": twin, "name": "Real Gimbal Cursor", "parent_id": child_parent, "twin": "real",
                         "dot": "real", "reading": bool(g.real_ok(oid)), "src": oid})
            # What rides on the Sim Gimbal Cursor follows the Real one's camera.
            _walk(aid, g.part_id(oid, "sim"), _xf(*g.twin_camera_world(obj, "sim")),
                  _real_cam_frame(obj, mirror), twin, rows, ghosts, gimbals, depth + 1)
        _walk(aid, oid, sim_w, mirror, child_parent, rows, ghosts, gimbals, depth + 1)


def compute():
    """Every arm's real side: {"arms": [...], "gimbals": {oid: mount xf}, "ghosts": [...]}."""
    import robot_runtime as rr
    out_arms, ghosts, gimbals = [], [], {}
    for aid in arms.each():
        with arms.using(aid):
            try:
                sim_xf = _xf(*rr.get_sim_cursor_pose())
            except Exception:
                continue
            connected = bool(arms.real_ok.get(aid, False))
            real_xf = sim_xf
            if connected:
                try:
                    real_xf = _xf(*rr.get_real_cursor_pose())
                except Exception:
                    connected, real_xf = False, sim_xf
            anchor = arms.cid("real_cursor", aid) if connected else none_id(aid)
            rows = []
            _walk(aid, arms.cid("sim_cursor", aid), sim_xf, real_xf, anchor, rows, ghosts, gimbals)
            out_arms.append({
                "id": aid, "connected": connected, "anchor_id": anchor,
                "anchor_name": "Real Arm Cursor" if connected else NOT_CONNECTED_NAME,
                "anchor_pos": [float(v) for v in real_xf[0]], "anchor_wxyz": [float(v) for v in real_xf[1]],
                "rows": rows,
            })
    return {"arms": out_arms, "gimbals": gimbals, "ghosts": ghosts}


def view():
    """The last tick's real side for the page (no nodes)."""
    return {"arms": _view.get("arms", []), "ghosts": ghosts_on()}


def find_row(rid):
    """A real-side row (ghost, stand-in) by id, for read-only Properties."""
    for a in _view.get("arms", []):
        if a["anchor_id"] == rid and not a["connected"]:
            return {"id": rid, "name": a["anchor_name"], "pos": a["anchor_pos"], "wxyz": a["anchor_wxyz"],
                    "local_pos": a["anchor_pos"], "local_wxyz": a["anchor_wxyz"], "kind": "empty"}
        for r in a["rows"]:
            if r["id"] == rid:
                return r
    return None


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------
def _ghost_sig(obj):
    sc = tuple(round(float(v), 5) for v in np.asarray(obj.get("scale", [1, 1, 1]), dtype=float).reshape(3))
    model_sig = obj.get("_model_sig") if obj.get("kind") == "gimbal" else None
    return (obj.get("kind"), sc, obj.get("mesh_path"), id(obj.get("_mesh_verts")), str(model_sig))


def _build_ghost(name, obj):
    server = wp.server
    kind = obj.get("kind")
    sc = np.asarray(obj.get("scale", [1, 1, 1]), dtype=float).reshape(3)
    if kind == "gimbal":
        import gimbal_model as gmodel
        g = _gimbal()
        model = obj.get("_model_used") or g.model_of(obj)[0]
        return gmodel.build_base(server, name, model, g.HANDLE_RGB, None, opacity=0.7 * GHOST_OPACITY)
    verts, faces = obj.get("_mesh_verts"), obj.get("_mesh_faces")
    if verts is not None and faces is not None:
        v = np.asarray(verts, dtype=np.float32).copy() * sc.astype(np.float32)
        return server.scene.add_mesh_simple(name=name, vertices=v, faces=np.asarray(faces, dtype=np.uint32),
                                            color=(200, 200, 200), opacity=GHOST_OPACITY)
    if kind == "cube":
        h = server.scene.add_box(name=name, dimensions=tuple(float(0.1 * s) for s in sc), color=(220, 220, 220))
        h.opacity = GHOST_OPACITY
        return h
    if kind == "sphere":
        h = server.scene.add_icosphere(name=name, radius=0.05 * max(1e-4, float(np.mean(np.abs(sc)))),
                                       color=(220, 220, 220))
        h.opacity = GHOST_OPACITY
        return h
    h = server.scene.add_icosphere(name=name, radius=0.02 * max(1e-4, float(np.mean(np.abs(sc)))),
                                   color=(150, 150, 158))
    h.opacity = 0.4 * GHOST_OPACITY
    return h


def _drop_ghost(gid):
    rec = _ghosts.pop(gid, None)
    if rec and rec.get("handle") is not None:
        try:
            rec["handle"].remove()
        except Exception:
            pass


def tick():
    """Viewer thread: place the Real Gimbal Cursors and the ghosts."""
    global _view
    if wp.server is None:
        return
    try:
        res = compute()
    except Exception as e:
        print("real side failed:", e)
        return
    _view = {"arms": res["arms"]}
    g = _gimbal()
    for obj in g.gimbal_objects():
        oid = str(obj.get("id"))
        xf = res["gimbals"].get(oid)
        if xf is None:
            xf = _xf(obj.get("pos"), obj.get("wxyz"))  # on its own mount
        g.place_real_root(obj, xf[0], xf[1])
    keep = set()
    for gid, obj, xf in res["ghosts"]:
        keep.add(gid)
        sig = _ghost_sig(obj)
        rec = _ghosts.get(gid)
        if rec is None or rec["sig"] != sig:
            _drop_ghost(gid)
            try:
                rec = _ghosts[gid] = {"handle": _build_ghost("/real_ghost_%s" % gid.replace(":", "_").replace("@", "_"), obj),
                                      "sig": sig, "pose": None, "vis": None}
            except Exception as e:
                print("ghost failed", gid, e)
                continue
        h = rec["handle"]
        pose = tuple(np.round(np.concatenate(xf), 6))
        if rec["pose"] != pose:
            rec["pose"] = pose
            try:
                h.position = tuple(float(v) for v in xf[0])
                h.wxyz = tuple(float(v) for v in xf[1])
            except Exception:
                pass
        vis = obj.get("visible", True) is not False
        if rec["vis"] != vis:
            rec["vis"] = vis
            try:
                h.visible = vis
            except Exception:
                pass
    for gid in list(_ghosts):
        if gid not in keep:
            _drop_ghost(gid)
