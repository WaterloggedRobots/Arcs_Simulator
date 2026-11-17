"""Animate component: keyframed transforms (position, rotation, scale) for 3D
objects, laid out like the Sim Arm Cursor's path.

Data (`obj["anim"]`, saved with the object, in undo snapshots):
    {"added": True, "max_speed": m/s, "max_acc": m/s^2,
     "keys": [{"id", "t", "pos", "wxyz", "scale", "out", "in"}],   # parent frame
     "spans": [{"a": key id, "b": key id, "curve": sig, "samples": [{"u", "t"}]}]}

Each key is a waypoint in 3D: a sphere with arrows, Bezier handles (`out` of a
key, `in` of the next) at the span's thirds when a span is made, a gizmo when
selected (a handle: move only). Keys are in the object's parent frame (world
when it has none); they are drawn through the parent's current pose.

Sample frames: interior frames of each span, evenly spaced along the curve and
timed at constant speed when the span is made (`u` = fraction of the span's arc
length, `t` = time). They are not drawn in 3D; the timeline shows them with the
speed / acceleration between them (status too_fast over `max_speed` /
`max_acc`). Dragging a key retimes the samples of its two spans in proportion
(`STRETCH_SAMPLES`); the Sim Arm Cursor's rule (frames keep their times, red
once outside their span) is kept as `_cursor_rule_times`, unused. A spatial
edit (key moved, handle moved) makes that span's frames again.

Playback: `tick(t)` (viewer loop) puts every animated object at its pose at the
timeline time when that time (or the animation) changed, so an object moved by
hand stays where it is until the playhead moves. The pose at t: the span's
progress is piecewise linear through its frames, the position is the curve at
that arc fraction, rotation slerps and scale lerps by the same progress.

Look At: `pos_at_name(name, t)` gives an animated object's world position at
timeline time t; the bake aims its samples with it (robot_runtime
`_track_target_pos(name, t)`), only while baking.
"""
from __future__ import annotations

import math
import threading
import time

import numpy as np

TRACK_PREFIX = "anim_"
SAMPLE_SPACING_M = 0.02     # sample frames along a span
SAMPLE_MIN_DT = 0.04        # at most one frame per 40 ms
SAMPLE_ROT_DT = 0.1         # a span that does not move (rotation / scale only): one frame per 0.1 s
MAX_SAMPLES = 200           # per span
KEY_MIN_GAP = 0.02          # s between keys
DEFAULT_MAX_SPEED = 1.0     # m/s
DEFAULT_MAX_ACC = 2.0       # m/s^2
STRETCH_SAMPLES = True      # key drag retimes its spans' samples in proportion
KINDS = ("empty", "sphere", "cube", "camera", "mesh", "import", "gimbal")

KEY_COLOR = (235, 235, 240)
KEY_SEL_COLOR = (255, 216, 74)
ARROW_COLOR = (190, 140, 255)
PATH_COLOR = (170, 110, 255)
CTRL_COLOR = (255, 180, 50)

_lock = threading.RLock()
_viz = {}            # oid -> {"keys": [{"id", "sphere", "arrow", "side"}], "path": h, "sig": ...}
_sel = None          # {"oid", "kind": "key" | "ctrl", "id": key id, "which": "out" | "in"}
_sel_h = {"gizmo": None, "ctrls": [], "dots": []}
_drag = {"rot0": None, "dirty": None, "last": 0.0, "checkpoint": False}
_last_t = None
_dirty = set()       # oids whose pose must be put back at the playhead
_key_seq = [0]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _wp():
    import waypoints as wp
    return wp


def _sv():
    import simulatorview as sv
    return sv


def _v3(v, d=(0.0, 0.0, 0.0)):
    try:
        a = np.asarray(v if v is not None else d, dtype=float).reshape(-1)
        return a[:3] if a.size >= 3 else np.asarray(d, dtype=float)
    except Exception:
        return np.asarray(d, dtype=float)


def _q4(v):
    try:
        q = np.asarray(v if v is not None else (1, 0, 0, 0), dtype=float).reshape(-1)[:4]
        n = float(np.linalg.norm(q))
        return q / n if n > 1e-12 else np.array([1.0, 0, 0, 0])
    except Exception:
        return np.array([1.0, 0, 0, 0])


def _qmul(a, b):
    wa, xa, ya, za = a
    wb, xb, yb, zb = b
    return np.array([
        wa * wb - xa * xb - ya * yb - za * zb,
        wa * xb + xa * wb + ya * zb - za * yb,
        wa * yb - xa * zb + ya * wb + za * xb,
        wa * zb + xa * yb - ya * xb + za * wb,
    ])


def _qinv(q):
    return np.array([q[0], -q[1], -q[2], -q[3]]) / max(1e-12, float(np.dot(q, q)))


def _qrot(q, v):
    return _qmul(_qmul(q, np.array([0.0, v[0], v[1], v[2]])), _qinv(q))[1:]


def _slerp(q0, q1, u):
    q0, q1 = _q4(q0), _q4(q1)
    d = float(np.dot(q0, q1))
    if d < 0:
        q1, d = -q1, -d
    if d > 0.9995:
        return _q4(q0 + u * (q1 - q0))
    th = math.acos(min(1.0, d))
    s = math.sin(th)
    return _q4((math.sin((1 - u) * th) * q0 + math.sin(u * th) * q1) / s)


def _qangle(a, b):
    d = abs(float(np.dot(_q4(a), _q4(b))))
    return 2.0 * math.acos(min(1.0, d))


def _f(v, d=0.0):
    try:
        return float(v)
    except Exception:
        return d


def _objects():
    return [o for o in (getattr(_wp(), "scene_objects", None) or []) if isinstance(o, dict)]


def _find(oid):
    for o in _objects():
        if str(o.get("id")) == str(oid):
            return o
    return None


def anim_of(obj):
    a = (obj or {}).get("anim")
    return a if isinstance(a, dict) and a.get("added") else None


def animated_objects():
    return [o for o in _objects() if anim_of(o)]


def track_id(oid):
    return TRACK_PREFIX + str(oid)


def _new_key_id():
    _key_seq[0] += 1
    return "k%d_%d" % (int(time.time() * 1000) % 100000000, _key_seq[0])


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------
def clean_anim(raw):
    """A saved / snapshot record -> plain lists (None when not added)."""
    if not isinstance(raw, dict) or not raw.get("added"):
        return None
    keys = []
    for k in list(raw.get("keys") or []):
        if not isinstance(k, dict) or k.get("t") is None:
            continue
        keys.append({
            "id": str(k.get("id") or _new_key_id()),
            "t": round(_f(k.get("t")), 4),
            "pos": [float(v) for v in _v3(k.get("pos"))],
            "wxyz": [float(v) for v in _q4(k.get("wxyz"))],
            "scale": [float(v) for v in _v3(k.get("scale"), (1, 1, 1))],
            "out": None if k.get("out") is None else [float(v) for v in _v3(k.get("out"))],
            "in": None if k.get("in") is None else [float(v) for v in _v3(k.get("in"))],
        })
    keys.sort(key=lambda k: k["t"])
    spans = []
    for s in list(raw.get("spans") or []):
        if not isinstance(s, dict):
            continue
        spans.append({
            "a": str(s.get("a") or ""), "b": str(s.get("b") or ""),
            "curve": s.get("curve"),
            "samples": [{"u": _f(m.get("u")), "t": _f(m.get("t"))} for m in (s.get("samples") or [])
                        if isinstance(m, dict) and m.get("t") is not None],
        })
    out = {
        "added": True,
        "max_speed": max(1e-3, _f(raw.get("max_speed"), DEFAULT_MAX_SPEED)),
        "max_acc": max(1e-3, _f(raw.get("max_acc"), DEFAULT_MAX_ACC)),
        "keys": keys,
        "spans": spans,
    }
    _resync_spans(out)
    return out


def _ctrl_pts(a, b):
    p0, p3 = _v3(a["pos"]), _v3(b["pos"])
    p1 = _v3(a["out"]) if a.get("out") is not None else p0 + (p3 - p0) / 3.0
    p2 = _v3(b["in"]) if b.get("in") is not None else p3 - (p3 - p0) / 3.0
    return p0, p1, p2, p3


def _curve_sig(a, b):
    return [round(float(v), 6) for p in _ctrl_pts(a, b) for v in p]


def _bez(p0, p1, p2, p3, s):
    s = np.asarray(s, dtype=float).reshape(-1, 1)
    return ((1 - s) ** 3) * p0 + 3 * ((1 - s) ** 2) * s * p1 + 3 * (1 - s) * (s ** 2) * p2 + (s ** 3) * p3


_LUT_N = 64


def _lut(a, b):
    """(bezier params, cumulative arc length) of span a -> b."""
    pts4 = _ctrl_pts(a, b)
    s = np.linspace(0.0, 1.0, _LUT_N + 1)
    xyz = _bez(*pts4, s)
    seg = np.linalg.norm(np.diff(xyz, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    return s, cum, pts4


def _arc_point(a, b, u):
    """Point at arc fraction u of span a -> b."""
    s, cum, pts4 = _lut(a, b)
    L = float(cum[-1])
    if L < 1e-9:
        return _v3(a["pos"])
    par = float(np.interp(min(1.0, max(0.0, u)) * L, cum, s))
    return _bez(*pts4, par)[0]


def _span_len(a, b):
    return float(_lut(a, b)[1][-1])


def _new_samples(a, b):
    t0, t1 = float(a["t"]), float(b["t"])
    dt = max(0.0, t1 - t0)
    L = _span_len(a, b)
    if L > 1e-4:
        n = int(round(L / SAMPLE_SPACING_M)) - 1
    else:
        n = int(dt / SAMPLE_ROT_DT) - 1
    n = max(0, min(n, MAX_SAMPLES, int(dt / SAMPLE_MIN_DT) - 1))
    return [{"u": (k + 1) / (n + 1), "t": t0 + (k + 1) / (n + 1) * dt} for k in range(n)]


def _map_t(t, t0o, t1o, t0n, t1n):
    if abs(t1o - t0o) < 1e-9:
        return t0n
    u = min(1.0, max(0.0, (t - t0o) / (t1o - t0o)))
    return t0n + u * (t1n - t0n)


def _stretch_times(span, t0o, t1o, t0n, t1n):
    """Key drag: the span's frames keep their fraction of the span's time."""
    for m in span["samples"]:
        m["t"] = _map_t(m["t"], t0o, t1o, t0n, t1n)


def _cursor_rule_times(span, t0o, t1o, t0n, t1n):
    """The Sim Arm Cursor's rule, unused here (STRETCH_SAMPLES = False picks it):
    frames keep their times when a key moves; the timeline shows a frame
    outside its span red ("out") until it is put back or the span is made again
    (`sample_rows` marks them)."""
    return None


def _resync_spans(anim, old_times=None):
    """One span per pair of neighbouring keys. A span with the same two keys
    and the same curve keeps its frames (retimed when its keys' times changed);
    otherwise its frames are made again. old_times: {key id: t before the edit}."""
    keys = anim["keys"]
    keys.sort(key=lambda k: k["t"])
    by_pair = {(s["a"], s["b"]): s for s in anim.get("spans") or []}
    old_times = old_times or {}
    spans = []
    for a, b in zip(keys, keys[1:]):
        sig = _curve_sig(a, b)
        old = by_pair.get((a["id"], b["id"]))
        if old is not None and old.get("curve") == sig:
            t0o = old_times.get(a["id"], a["t"])
            t1o = old_times.get(b["id"], b["t"])
            if abs(t0o - a["t"]) > 1e-9 or abs(t1o - b["t"]) > 1e-9:
                if STRETCH_SAMPLES:
                    _stretch_times(old, t0o, t1o, a["t"], b["t"])
                else:
                    _cursor_rule_times(old, t0o, t1o, a["t"], b["t"])
            old["samples"].sort(key=lambda m: m["u"])
            spans.append(old)
        else:
            spans.append({"a": a["id"], "b": b["id"], "curve": sig, "samples": _new_samples(a, b)})
    anim["spans"] = spans
    return spans


def _place_ctrls(anim, i):
    """Handles of span i at its thirds when it has none (like a new path span)."""
    keys = anim["keys"]
    if not (0 <= i < len(keys) - 1):
        return
    a, b = keys[i], keys[i + 1]
    p0, p3 = _v3(a["pos"]), _v3(b["pos"])
    if a.get("out") is None:
        a["out"] = [float(v) for v in p0 + (p3 - p0) / 3.0]
    if b.get("in") is None:
        b["in"] = [float(v) for v in p3 - (p3 - p0) / 3.0]


def _unfold_ctrls(anim, i):
    """After a key moved: a span whose handles both sit on its keys (both keys
    were at one place) gets handles at its thirds again."""
    keys = anim["keys"]
    for j in (i - 1, i):
        if not (0 <= j < len(keys) - 1):
            continue
        a, b = keys[j], keys[j + 1]
        pa, pb = _v3(a["pos"]), _v3(b["pos"])
        if float(np.linalg.norm(pb - pa)) < 1e-6:
            continue
        on = a.get("out") is None or float(np.linalg.norm(_v3(a["out"]) - pa)) < 1e-6
        inn = b.get("in") is None or float(np.linalg.norm(_v3(b["in"]) - pb)) < 1e-6
        if on and inn:
            a["out"] = None
            b["in"] = None
            _place_ctrls(anim, j)


# ---------------------------------------------------------------------------
# pose at a time
# ---------------------------------------------------------------------------
def _span_at(anim, t):
    keys = anim["keys"]
    for i in range(len(keys) - 1):
        if keys[i]["t"] - 1e-9 <= t <= keys[i + 1]["t"] + 1e-9:
            return i
    return None


def _progress(anim, i, t):
    a, b = anim["keys"][i], anim["keys"][i + 1]
    span = anim["spans"][i] if i < len(anim.get("spans") or []) else {"samples": []}
    ts = [a["t"]] + [m["t"] for m in span["samples"]] + [b["t"]]
    us = [0.0] + [m["u"] for m in span["samples"]] + [1.0]
    if ts[-1] - ts[0] < 1e-9:
        return 1.0
    return float(np.interp(t, ts, us))


def local_pose_at(obj, t):
    """(pos, wxyz, scale) in the parent frame at timeline time t, or None."""
    anim = anim_of(obj)
    if not anim or not anim["keys"]:
        return None
    keys = anim["keys"]
    t = float(t)
    if t <= keys[0]["t"] or len(keys) == 1:
        k = keys[0]
        return _v3(k["pos"]), _q4(k["wxyz"]), _v3(k["scale"], (1, 1, 1))
    if t >= keys[-1]["t"]:
        k = keys[-1]
        return _v3(k["pos"]), _q4(k["wxyz"]), _v3(k["scale"], (1, 1, 1))
    i = _span_at(anim, t)
    if i is None:
        k = keys[-1]
        return _v3(k["pos"]), _q4(k["wxyz"]), _v3(k["scale"], (1, 1, 1))
    a, b = keys[i], keys[i + 1]
    u = _progress(anim, i, t)
    pos = _arc_point(a, b, u)
    q = _slerp(a["wxyz"], b["wxyz"], u)
    sc = _v3(a["scale"], (1, 1, 1)) * (1 - u) + _v3(b["scale"], (1, 1, 1)) * u
    return pos, q, sc


def _parent_pose(obj):
    try:
        par = _sv()._obj_parent(obj)
    except Exception:
        par = None
    if par is None:
        return np.zeros(3), np.array([1.0, 0, 0, 0])
    return _v3(par.get("pos")), _q4(par.get("wxyz"))


def _to_world(obj, p, q=None):
    pp, pq = _parent_pose(obj)
    wpos = pp + _qrot(pq, _v3(p))
    return wpos if q is None else (wpos, _q4(_qmul(pq, _q4(q))))


def _to_local(obj, p, q=None):
    pp, pq = _parent_pose(obj)
    inv = _qinv(pq)
    lpos = _qrot(inv, _v3(p) - pp)
    return lpos if q is None else (lpos, _q4(_qmul(inv, _q4(q))))


def world_pos_at(obj, t):
    pose = local_pose_at(obj, t)
    if pose is None:
        return None
    return _to_world(obj, pose[0])


def pos_at_name(name, t):
    """World position at timeline time t of the animated object called name
    (Look At targets are named), or None (not animated / no keys)."""
    key = str(name or "").strip().lower()
    if not key or t is None:
        return None
    for o in animated_objects():
        if str(o.get("name") or "").strip().lower() == key and anim_of(o)["keys"]:
            return world_pos_at(o, t)
    return None


def is_animated_name(name):
    key = str(name or "").strip().lower()
    return any(str(o.get("name") or "").strip().lower() == key and len(anim_of(o)["keys"]) >= 2
               for o in animated_objects())


def signature_name(name):
    """Changes when that object's animation changes (the bake re-aims spans aimed at it)."""
    key = str(name or "").strip().lower()
    for o in animated_objects():
        if str(o.get("name") or "").strip().lower() == key:
            a = anim_of(o)
            return (tuple((k["t"], tuple(k["pos"]), tuple(k["out"] or ()), tuple(k["in"] or ())) for k in a["keys"]),
                    tuple(tuple(round(m["t"], 4) for m in s["samples"]) for s in a["spans"]),
                    tuple(_parent_pose(o)[0].round(5)))
    return None


def _apply_pose(obj, t):
    pose = local_pose_at(obj, t)
    if pose is None:
        return
    sv = _sv()
    pos, q, sc = pose
    obj["local_pos"] = np.asarray(pos, dtype=float)
    obj["local_wxyz"] = np.asarray(q, dtype=float)
    old = _v3(obj.get("scale"), (1, 1, 1))
    if float(np.max(np.abs(old - sc))) > 1e-6:
        obj["scale"] = np.asarray(sc, dtype=float)
    sv._set_world_from_local(obj)
    sv._apply_object_xform(obj)
    try:
        sv._refresh_camera_viz(obj)
    except Exception:
        pass
    sv._propagate_children(obj.get("id"))


def tick(t):
    """Viewer loop: put animated objects at their pose at timeline time t when
    it changed, or after an edit of their animation."""
    global _last_t
    with _lock:
        _drag_idle_check()
        objs = animated_objects()
        if not objs:
            _last_t = t
            return
        moved = _last_t is None or abs(float(t) - float(_last_t)) > 1e-6
        for o in objs:
            if moved or str(o.get("id")) in _dirty:
                try:
                    _apply_pose(o, t)
                except Exception as e:
                    print("animate: pose failed", o.get("id"), e)
        _dirty.clear()
        _last_t = t
        _follow_parents(objs)


def _playhead():
    try:
        return float(_sv()._playhead_t())
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# timeline rows
# ---------------------------------------------------------------------------
def _nodes(obj):
    """[(t, s_m, wxyz, kind, ref)] along the whole animation: keys and frames."""
    anim = anim_of(obj)
    out = []
    s_acc = 0.0
    keys = anim["keys"]
    for i, k in enumerate(keys):
        if i == 0:
            out.append((k["t"], 0.0, _q4(k["wxyz"]), "key", i))
        if i + 1 >= len(keys):
            break
        b = keys[i + 1]
        L = _span_len(k, b)
        span = anim["spans"][i]
        for j, m in enumerate(span["samples"]):
            out.append((m["t"], s_acc + m["u"] * L, _slerp(k["wxyz"], b["wxyz"], m["u"]), "sample", (i, j)))
        s_acc += L
        out.append((b["t"], s_acc, _q4(b["wxyz"]), "key", i + 1))
    return out


def track_rows(obj):
    """Timeline data for the object's track: keys, sample frames (speed,
    acceleration, turn rate) and the hops between them."""
    anim = anim_of(obj)
    if not anim:
        return {}
    nodes = _nodes(obj)
    vmax, amax = anim["max_speed"], anim["max_acc"]
    hops = []
    for (ta, sa, qa, _ka, _ra), (tb, sb, qb, _kb, _rb) in zip(nodes, nodes[1:]):
        dt = tb - ta
        v = (sb - sa) / dt if dt > 1e-9 else 0.0
        w = math.degrees(_qangle(qa, qb)) / dt if dt > 1e-9 else 0.0
        hops.append({"t0": ta, "t1": tb, "v": v, "w": w, "status": "too_fast" if v > vmax * 1.0001 else "ok"})
    acc = [0.0] * len(nodes)
    for j in range(1, len(nodes) - 1):
        h0, h1 = hops[j - 1], hops[j]
        dt = 0.5 * ((h0["t1"] - h0["t0"]) + (h1["t1"] - h1["t0"]))
        acc[j] = (h1["v"] - h0["v"]) / dt if dt > 1e-9 else 0.0
    keys, samples = [], []
    for j, (t, s, _q, kind, ref) in enumerate(nodes):
        vin = hops[j - 1]["v"] if j > 0 else 0.0
        vout = hops[j]["v"] if j < len(hops) else 0.0
        over = max(vin, vout) > vmax * 1.0001 or abs(acc[j]) > amax * 1.0001
        rec = {"t": t, "v": 0.5 * (vin + vout) if 0 < j < len(nodes) - 1 else (vout if j == 0 else vin),
               "a": acc[j], "status": "too_fast" if over else "ok"}
        if kind == "key":
            rec["index"] = ref
            keys.append(rec)
        else:
            rec["seg"], rec["k"] = ref
            a, b = anim["keys"][ref[0]], anim["keys"][ref[0] + 1]
            if not STRETCH_SAMPLES and not (a["t"] - 1e-6 <= t <= b["t"] + 1e-6):
                rec["status"] = "out"
            samples.append(rec)
    if len(anim["keys"]) == 1:
        keys = [{"t": anim["keys"][0]["t"], "index": 0, "v": 0.0, "a": 0.0, "status": "ok"}]
    sel = None
    if _sel and str(_sel.get("oid")) == str(obj.get("id")):
        i = _key_index(anim, _sel.get("id"))
        if i is not None:
            sel = i
    return {"keys": keys, "samples": samples, "hops": hops, "sel": sel,
            "max_speed": vmax, "max_acc": amax}


def sync_tracks(tracks):
    """One timeline track per animated object (anim_<oid>), orphans dropped."""
    objs = {str(o.get("id")): o for o in animated_objects()}
    out, seen = [], set()
    for tr in list(tracks or []):
        tid = str((tr or {}).get("id") or "") if isinstance(tr, dict) else ""
        if tid.startswith(TRACK_PREFIX):
            oid = tid[len(TRACK_PREFIX):]
            if oid not in objs or oid in seen:
                continue
            tr["name"] = str(objs[oid].get("name") or oid)
            tr["keys"] = []
            seen.add(oid)
        out.append(tr)
    for oid, o in objs.items():
        if oid not in seen:
            out.append({"id": track_id(oid), "name": str(o.get("name") or oid), "keys": []})
    return out


def display_track(tr):
    """A track for the page: its keys (t) and the rows (samples, speeds)."""
    oid = str(tr.get("id"))[len(TRACK_PREFIX):]
    o = _find(oid)
    if o is None or not anim_of(o):
        return tr
    with _lock:
        rows = track_rows(o)
    return dict(tr, keys=[{"t": k["t"], "index": k["index"]} for k in rows.get("keys") or []], anim=rows)


def status(oid):
    """Properties: the component's settings and the selected key."""
    o = _find(oid)
    anim = anim_of(o)
    if not anim:
        return {"added": False}
    keys = anim["keys"]
    sel = None
    if _sel and str(_sel.get("oid")) == str(oid):
        i = _key_index(anim, _sel.get("id"))
        if i is not None:
            k = keys[i]
            sel = {"index": i, "kind": _sel.get("kind"), "which": _sel.get("which"),
                   "t": k["t"], "pos": k["pos"], "wxyz": k["wxyz"], "scale": k["scale"]}
    return {
        "added": True,
        "count": len(keys),
        "t0": keys[0]["t"] if keys else None,
        "t1": keys[-1]["t"] if keys else None,
        "max_speed": anim["max_speed"],
        "max_acc": anim["max_acc"],
        "sel": sel,
        "parented": bool(o.get("parent_id")),
    }


# ---------------------------------------------------------------------------
# edits (web requests / gizmos); each pushes one undo step
# ---------------------------------------------------------------------------
def _key_index(anim, kid):
    for i, k in enumerate(anim["keys"]):
        if k["id"] == kid:
            return i
    return None


def _edited(obj):
    _dirty.add(str(obj.get("id")))
    _rebuild_viz(obj)


def _sync_tracks_now():
    try:
        _sv().sync_camera_tracks()
    except Exception as e:
        print("animate: track sync failed", e)


def add_component(oid):
    o = _find(oid)
    if o is None or o.get("kind") not in KINDS:
        return False, "Animate is for 3D objects"
    if anim_of(o):
        return True, ""
    _wp()._push_history()
    with _lock:
        o["anim"] = {"added": True, "max_speed": DEFAULT_MAX_SPEED, "max_acc": DEFAULT_MAX_ACC, "keys": [], "spans": []}
        order = list(o.get("comp_order") or [])
        if "anim" not in order:
            order.append("anim")
        o["comp_order"] = order
    _sync_tracks_now()
    return True, ""


def remove_component(oid):
    o = _find(oid)
    if o is None or not anim_of(o):
        return False, "no Animate component"
    _wp()._push_history()
    with _lock:
        if _sel and str(_sel.get("oid")) == str(oid):
            deselect()
        o["anim"] = None
        o["comp_order"] = [c for c in (o.get("comp_order") or []) if c != "anim"]
        forget(oid)
    _sync_tracks_now()
    return True, ""


def set_limits(oid, max_speed=None, max_acc=None):
    o = _find(oid)
    anim = anim_of(o)
    if not anim:
        return False, "no Animate component"
    _wp()._push_history()
    if max_speed is not None:
        anim["max_speed"] = max(1e-3, _f(max_speed, anim["max_speed"]))
    if max_acc is not None:
        anim["max_acc"] = max(1e-3, _f(max_acc, anim["max_acc"]))
    return True, ""


def _snap_t(t):
    return round(max(0.0, _f(t)), 4)


def insert_key(oid, t):
    """A key at time t from the object's current transform (replaces a key at t)."""
    o = _find(oid)
    anim = anim_of(o)
    if not anim:
        return False, "no Animate component"
    t = _snap_t(t)
    _wp()._push_history()
    with _lock:
        lp = _v3(o.get("local_pos", o.get("pos")))
        lq = _q4(o.get("local_wxyz", o.get("wxyz")))
        sc = _v3(o.get("scale"), (1, 1, 1))
        hit = next((k for k in anim["keys"] if abs(k["t"] - t) < 0.005), None)
        if hit is not None:
            delta = lp - _v3(hit["pos"])
            hit["pos"] = [float(v) for v in lp]
            hit["wxyz"] = [float(v) for v in lq]
            hit["scale"] = [float(v) for v in sc]
            for w in ("out", "in"):
                if hit.get(w) is not None:
                    hit[w] = [float(v) for v in _v3(hit[w]) + delta]
            kid = hit["id"]
        else:
            kid = _new_key_id()
            anim["keys"].append({"id": kid, "t": t, "pos": [float(v) for v in lp], "wxyz": [float(v) for v in lq],
                                 "scale": [float(v) for v in sc], "out": None, "in": None})
            anim["keys"].sort(key=lambda k: k["t"])
            i = _key_index(anim, kid)
            # The new key splits a span: the spans either side get fresh handles.
            if i > 0:
                anim["keys"][i - 1]["out"] = None
            if i + 1 < len(anim["keys"]):
                anim["keys"][i + 1]["in"] = None
            _place_ctrls(anim, i - 1)
            _place_ctrls(anim, i)
        i = _key_index(anim, kid)
        _unfold_ctrls(anim, i)
        _resync_spans(anim)
        _select(o, {"oid": str(oid), "kind": "key", "id": kid})
        _edited(o)
    _sync_tracks_now()
    return True, ""


def delete_keys(oid, indices=None, ts=None):
    o = _find(oid)
    anim = anim_of(o)
    if not anim:
        return False, "no Animate component"
    keys = anim["keys"]
    drop = set()
    for i in indices or []:
        try:
            if 0 <= int(i) < len(keys):
                drop.add(keys[int(i)]["id"])
        except Exception:
            pass
    for t in ts or []:
        for k in keys:
            if abs(k["t"] - _f(t)) < 1e-4:
                drop.add(k["id"])
    if not drop:
        return False, "no such key"
    _wp()._push_history()
    with _lock:
        anim["keys"] = [k for k in keys if k["id"] not in drop]
        # The span that now bridges the gap gets handles at its thirds.
        for i in range(len(anim["keys"]) - 1):
            a, b = anim["keys"][i], anim["keys"][i + 1]
            if not any(s["a"] == a["id"] and s["b"] == b["id"] for s in anim["spans"]):
                a["out"], b["in"] = None, None
                _place_ctrls(anim, i)
        if anim["keys"]:
            anim["keys"][0]["in"] = None
            anim["keys"][-1]["out"] = None
        _resync_spans(anim)
        if _sel and _sel.get("id") in drop:
            deselect()
        _edited(o)
    return True, ""


def move_keys(oid, moves):
    """moves: [{index | t0, t}]: retime keys (their spans' frames stretch)."""
    o = _find(oid)
    anim = anim_of(o)
    if not anim:
        return False, "no Animate component"
    keys = anim["keys"]
    todo = []
    for m in moves or []:
        k = None
        if m.get("index") is not None and 0 <= int(m["index"]) < len(keys):
            k = keys[int(m["index"])]
        elif m.get("t0") is not None:
            k = next((x for x in keys if abs(x["t"] - _f(m["t0"])) < 1e-4), None)
        if k is not None and m.get("t") is not None:
            todo.append((k, _snap_t(m["t"])))
    if not todo:
        return False, "no such key"
    moving = {id(k) for k, _ in todo}
    for k, t in todo:
        for other in keys:
            if id(other) not in moving and abs(other["t"] - t) < KEY_MIN_GAP - 1e-6:
                return False, "keys must be %.2f s apart" % KEY_MIN_GAP
    _wp()._push_history()
    with _lock:
        old = {k["id"]: k["t"] for k in keys}
        order = [k["id"] for k in keys]
        for k, t in todo:
            k["t"] = t
        keys.sort(key=lambda k: k["t"])
        if [k["id"] for k in keys] != order:
            # Order changed: the spans are new pairs (frames made again), with handles at thirds.
            for k in keys:
                k["out"] = k["in"] = None
            for i in range(len(keys) - 1):
                _place_ctrls(anim, i)
        _resync_spans(anim, old)
        _edited(o)
    return True, ""


def move_sample(oid, seg, k, t):
    """Retime one sample frame, kept between its neighbours."""
    o = _find(oid)
    anim = anim_of(o)
    if not anim:
        return False, "no Animate component"
    try:
        seg, k = int(seg), int(k)
        span = anim["spans"][seg]
        m = span["samples"][k]
    except Exception:
        return False, "no such frame"
    a, b = anim["keys"][seg], anim["keys"][seg + 1]
    lo = span["samples"][k - 1]["t"] if k > 0 else a["t"]
    hi = span["samples"][k + 1]["t"] if k + 1 < len(span["samples"]) else b["t"]
    eps = 0.002
    _wp()._push_history()
    with _lock:
        m["t"] = min(max(_f(t), lo + eps), hi - eps) if hi - lo > 2 * eps else 0.5 * (lo + hi)
        _dirty.add(str(oid))
    return True, ""


def edit_key(oid, index, t=None, pos=None, wxyz=None, scale=None):
    """Properties fields of a key (pos / rotation in the parent frame)."""
    o = _find(oid)
    anim = anim_of(o)
    if not anim:
        return False, "no Animate component"
    try:
        k = anim["keys"][int(index)]
    except Exception:
        return False, "no such key"
    if t is not None:
        ok, err = move_keys(oid, [{"index": int(index), "t": t}])
        if not ok:
            return ok, err
    if pos is None and wxyz is None and scale is None:
        return True, ""
    if t is None:
        _wp()._push_history()
    with _lock:
        if pos is not None:
            _move_key_pos(anim, k, _v3(pos))
        if wxyz is not None:
            k["wxyz"] = [float(v) for v in _q4(wxyz)]
        if scale is not None:
            k["scale"] = [float(v) for v in _v3(scale, (1, 1, 1))]
        _resync_spans(anim)
        _edited(o)
    return True, ""


def _move_key_pos(anim, k, p):
    """A key moved in space: its handles move with it."""
    delta = p - _v3(k["pos"])
    k["pos"] = [float(v) for v in p]
    for w in ("out", "in"):
        if k.get(w) is not None:
            k[w] = [float(v) for v in _v3(k[w]) + delta]
    _unfold_ctrls(anim, _key_index(anim, k["id"]))


def key_to_object(oid, index):
    """Put the object at a key's transform (and the playhead there)."""
    o = _find(oid)
    anim = anim_of(o)
    if not anim or not (0 <= int(index) < len(anim["keys"])):
        return None
    return anim["keys"][int(index)]["t"]


# ---------------------------------------------------------------------------
# 3D: keys, path, handles, gizmos
# ---------------------------------------------------------------------------
def _server():
    return getattr(_wp(), "server", None)


def _rm(h):
    if h is None:
        return
    try:
        h.remove()
    except Exception:
        pass


def forget(oid):
    """Remove an object's animation drawing (deleted / component removed)."""
    with _lock:
        v = _viz.pop(str(oid), None)
        if v:
            for k in v.get("keys") or []:
                for name in ("sphere", "arrow", "side"):
                    _rm(k.get(name))
            _rm(v.get("path"))
        if _sel and str(_sel.get("oid")) == str(oid):
            deselect()


def forget_all():
    for oid in list(_viz.keys()):
        forget(oid)


def _node(oid):
    return "/anim/%s" % oid


def _visible(obj):
    return obj.get("visible", True) is not False


def _rebuild_viz(obj, gizmo=True):
    srv = _server()
    if srv is None:
        return
    oid = str(obj.get("id"))
    anim = anim_of(obj)
    if not anim:
        forget(oid)
        return
    with _lock:
        v = _viz.setdefault(oid, {"keys": [], "path": None})
        keys = anim["keys"]
        shown = _visible(obj)
        # Key markers: one per key, reused by position in the list.
        while len(v["keys"]) > len(keys):
            k = v["keys"].pop()
            for name in ("sphere", "arrow", "side"):
                _rm(k.get(name))
        for i, k in enumerate(keys):
            wpos, wq = _to_world(obj, k["pos"], k["wxyz"])
            sel = bool(_sel and _sel.get("oid") == oid and _sel.get("id") == k["id"] and _sel.get("kind") == "key")
            col = KEY_SEL_COLOR if sel else KEY_COLOR
            if i >= len(v["keys"]):
                v["keys"].append(_make_key_node(obj, i))
            rec = v["keys"][i]
            rec["id"] = k["id"]
            for name in ("sphere", "arrow", "side"):
                h = rec.get(name)
                if h is None:
                    continue
                try:
                    h.position = tuple(float(x) for x in wpos)
                    h.wxyz = tuple(float(x) for x in wq)
                    h.visible = shown
                except Exception:
                    pass
            try:
                if rec.get("col") != col:
                    rec["sphere"].color = col
                    rec["col"] = col
            except Exception:
                pass
        # Path: the curve of every span.
        segs = []
        for a, b in zip(keys, keys[1:]):
            pts4 = _ctrl_pts(a, b)
            pts = _bez(*pts4, np.linspace(0, 1, 25))
            pts = np.array([_to_world(obj, p) for p in pts])
            for j in range(len(pts) - 1):
                segs.append([pts[j], pts[j + 1]])
        if segs:
            arr = np.asarray(segs, dtype=np.float32)
            cols = np.tile(np.asarray(PATH_COLOR, dtype=np.uint8), (len(segs), 2, 1))
            h = v.get("path")
            try:
                if h is not None:
                    h.points = arr
                    h.colors = cols
                    h.visible = shown
                else:
                    raise RuntimeError
            except Exception:
                _rm(h)
                v["path"] = srv.scene.add_line_segments(name=_node(oid) + "/path", points=arr, colors=cols,
                                                        line_width=3.5, visible=shown)
        else:
            _rm(v.get("path"))
            v["path"] = None
        v["parent"] = tuple(np.round(np.concatenate(_parent_pose(obj)), 6))
        if _sel and _sel.get("oid") == oid:
            _show_sel_handles(obj, gizmo=gizmo)


def _make_key_node(obj, i):
    srv = _server()
    oid = str(obj.get("id"))
    name = "%s/key_%d_%d" % (_node(oid), i, int(time.time() * 1000) % 100000)
    sphere = srv.scene.add_icosphere(name=name + "/sphere", radius=0.018, color=KEY_COLOR)
    rec = {"sphere": sphere, "col": KEY_COLOR}
    try:
        pts = np.array([[[0.0, 0.0, 0.0], [0.0, 0.0, 0.07]]], dtype=np.float32)
        rec["arrow"] = srv.scene.add_arrows(name=name + "/arrow", points=pts, colors=np.asarray(ARROW_COLOR, np.uint8),
                                            shaft_radius=0.0035, head_radius=0.009, head_length=0.016)
        pts = np.array([[[0.0, 0.0, 0.0], [0.032, 0.0, 0.0]]], dtype=np.float32)
        rec["side"] = srv.scene.add_arrows(name=name + "/side", points=pts, colors=np.asarray(ARROW_COLOR, np.uint8),
                                           shaft_radius=0.0025, head_radius=0.006, head_length=0.011)
    except Exception:
        pass

    @sphere.on_click
    def _(_e=None, _rec=rec, _oid=oid):
        o = _find(_oid)
        anim = anim_of(o)
        if not anim:
            return
        if _key_index(anim, _rec.get("id")) is None:
            return
        select_key(_oid, key_id=_rec.get("id"))
    return rec


def _follow_parents(objs):
    """Redraw an animation whose parent moved (keys are in its frame)."""
    for o in objs:
        v = _viz.get(str(o.get("id")))
        if v is None:
            _rebuild_viz(o)
            continue
        cur = tuple(np.round(np.concatenate(_parent_pose(o)), 6))
        if cur != v.get("parent"):
            _rebuild_viz(o)


def refresh_all():
    """After load / undo: draw every animation again, drop the others."""
    with _lock:
        live = {str(o.get("id")) for o in animated_objects()}
        for oid in list(_viz.keys()):
            if oid not in live:
                forget(oid)
        if _sel and _sel.get("oid") not in live:
            deselect()
        for o in animated_objects():
            _rebuild_viz(o)
            _dirty.add(str(o.get("id")))


def refresh_visibility(oid):
    o = _find(oid)
    if o is not None and anim_of(o):
        _rebuild_viz(o)


# selection --------------------------------------------------------------------
def _clear_sel_handles():
    _rm(_sel_h.get("gizmo"))
    _sel_h["gizmo"] = None
    for h in _sel_h["ctrls"] + _sel_h["dots"]:
        _rm(h)
    _sel_h["ctrls"], _sel_h["dots"] = [], []


def deselect():
    global _sel
    with _lock:
        old = _sel
        _sel = None
        _clear_sel_handles()
        if old:
            o = _find(old.get("oid"))
            if o is not None and anim_of(o):
                _rebuild_viz(o)


def on_selection_changed(selection):
    """waypoints._apply_selection_viz: a key stays selected only while its object is."""
    if not _sel:
        return
    if ("obj", _sel.get("oid")) not in [(k, str(v)) for k, v in (selection or [])]:
        deselect()
    else:
        _hide_object_gizmo(_find(_sel.get("oid")))


def _hide_object_gizmo(obj):
    giz = (obj or {}).get("gizmo")
    if giz is not None:
        try:
            giz.visible = False
        except Exception:
            pass


def _select(obj, sel):
    global _sel
    wp = _wp()
    oid = str(obj.get("id"))
    if wp.selected_obj_id != oid or len(getattr(wp, "selected_obj_ids", []) or []) != 1:
        try:
            wp.select_item("obj", oid)
        except Exception as e:
            print("animate: select object failed", e)
    _sel = dict(sel)
    _drag["rot0"] = None
    _hide_object_gizmo(obj)
    _rebuild_viz(obj)


def select_key(oid, index=None, key_id=None, which=None):
    """Select a key (which None) or one of its handles (which out / in)."""
    o = _find(oid)
    anim = anim_of(o)
    if not anim:
        return False
    with _lock:
        if key_id is None:
            if index is None or not (0 <= int(index) < len(anim["keys"])):
                deselect()
                return True
            key_id = anim["keys"][int(index)]["id"]
        _select(o, {"oid": str(oid), "kind": "ctrl" if which else "key", "id": key_id, "which": which})
    return True


def _show_sel_handles(obj, gizmo=True):
    """Gizmo on the selected key / handle; the handles of the key's two spans
    with dotted lines to their keys. gizmo False: keep the gizmo there is (a drag)."""
    srv = _server()
    if gizmo:
        _clear_sel_handles()
    else:
        for h in _sel_h["ctrls"] + _sel_h["dots"]:
            _rm(h)
        _sel_h["ctrls"], _sel_h["dots"] = [], []
    if srv is None or not _sel:
        return
    anim = anim_of(obj)
    i = _key_index(anim, _sel.get("id")) if anim else None
    if i is None:
        return
    keys = anim["keys"]
    k = keys[i]
    wp = _wp()
    oid = str(obj.get("id"))
    handles = []
    if i + 1 < len(keys) and k.get("out") is not None:
        handles.append((i, "out"))
    if i > 0 and k.get("in") is not None:
        handles.append((i, "in"))
    if i > 0 and keys[i - 1].get("out") is not None:
        handles.append((i - 1, "out"))
    if i + 1 < len(keys) and keys[i + 1].get("in") is not None:
        handles.append((i + 1, "in"))
    for j, which in handles:
        kk = keys[j]
        wpos = _to_world(obj, kk[which])
        kpos = _to_world(obj, kk["pos"])
        h = srv.scene.add_icosphere(name="%s/ctrl_%s_%d" % (_node(oid), which, j), radius=0.014,
                                    color=KEY_SEL_COLOR if (_sel.get("kind") == "ctrl" and _sel.get("which") == which
                                                            and _sel.get("id") == kk["id"]) else CTRL_COLOR,
                                    position=tuple(float(x) for x in wpos))

        @h.on_click
        def _(_e=None, _kid=kk["id"], _w=which, _oid=oid):
            select_key(_oid, key_id=_kid, which=_w)
        _sel_h["ctrls"].append(h)
        pts = wp._dotted_points(kpos, wpos, 12)
        if pts is not None:
            _sel_h["dots"].append(srv.scene.add_line_segments(
                name="%s/dot_%s_%d" % (_node(oid), which, j), points=pts, colors=(200, 200, 200), line_width=2.0))
    # Gizmo on the selected key (move + rotate) or handle (move only).
    if not gizmo:
        return
    if _sel.get("kind") == "ctrl":
        kk = next((x for x in keys if x["id"] == _sel.get("id")), None)
        if kk is None or kk.get(_sel.get("which")) is None:
            return
        gpos = _to_world(obj, kk[_sel["which"]])
        giz = wp._new_transform_gizmo("%s/ctrl_gizmo" % _node(oid), gpos, wp.WORLD_WXYZ, 0.10, True, True)
    else:
        gpos, gq = _to_world(obj, k["pos"], k["wxyz"])
        giz = wp._new_transform_gizmo("%s/key_gizmo" % _node(oid), gpos, wp._gizmo_display_wxyz(gq), 0.16, False, True)
    _sel_h["gizmo"] = giz
    giz._anim_sel = dict(_sel)

    @giz.on_update
    def _(_e=None, _g=giz):
        _on_gizmo(_g)

    for name in ("on_drag_end",):
        fn = getattr(giz, name, None)
        if callable(fn):
            try:
                fn(lambda _e=None: _drag_end())
            except Exception:
                pass


def _on_gizmo(giz):
    wp = _wp()
    if getattr(wp, "_suspend_gizmo_cb", False):
        return
    sel = getattr(giz, "_anim_sel", None)
    if not sel or sel != _sel:
        return
    with _lock:
        o = _find(sel["oid"])
        anim = anim_of(o)
        if not anim:
            return
        i = _key_index(anim, sel["id"])
        if i is None:
            return
        k = anim["keys"][i]
        gpos = np.asarray(giz.position, dtype=float)
        if not _drag["checkpoint"]:
            wp._push_history()
            _drag["checkpoint"] = True
        if sel["kind"] == "ctrl":
            lp = _to_local(o, gpos)
            if np.allclose(lp, _v3(k[sel["which"]]), atol=1e-8):
                return
            k[sel["which"]] = [float(v) for v in lp]
        else:
            wcur = _to_world(o, k["pos"], k["wxyz"])
            if _drag["rot0"] is None:
                _drag["rot0"] = np.array(wcur[1])
            wq = _q4(wp._orientation_from_gizmo(giz, wcur[1], _drag["rot0"]))
            lp, lq = _to_local(o, gpos, wq)
            if np.allclose(lp, _v3(k["pos"]), atol=1e-8) and _qangle(lq, k["wxyz"]) < 1e-7:
                return
            _move_key_pos(anim, k, lp)
            k["wxyz"] = [float(v) for v in lq]
        _drag["dirty"] = sel["oid"]
        _drag["last"] = time.time()
        _redraw_during_drag(o)


def _redraw_during_drag(o):
    """Move the markers and the curve; the handles' dots follow (the dragged
    gizmo stays); the sample frames wait for the release."""
    _rebuild_viz(o, gizmo=False)


def _drag_end():
    wp = _wp()
    with _lock:
        _drag["checkpoint"] = False
        _drag["rot0"] = None
        oid = _drag.get("dirty")
        _drag["dirty"] = None
        if oid is None:
            return
        o = _find(oid)
        anim = anim_of(o)
        if not anim:
            return
        _resync_spans(anim)          # a spatial edit: the touched spans' frames are made again
        _dirty.add(str(oid))
        g = _sel_h.get("gizmo")
        if g is not None and getattr(wp, "gizmo_space", "global") != "local":
            wp._suspend_gizmo_cb = True
            try:
                g.wxyz = wp.WORLD_WXYZ
            except Exception:
                pass
            finally:
                wp._suspend_gizmo_cb = False


def _drag_idle_check():
    """No drag-end event (some viser versions): end it once the gizmo is still."""
    if _drag.get("dirty") is not None and time.time() - _drag["last"] > 0.35:
        _drag_end()
