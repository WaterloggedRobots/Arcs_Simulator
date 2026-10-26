# version: beta_091701
# prior bake timing: 0916 (diamond ±1e-4 clamp; see project memory)
"""
Bake preview samples + IK along editor waypoints.
Samples are NOT editor waypoints.

Future motion fields (unused now, reserved):
  sample["t"], sample["speed"], sample["accel"]
  path["defaults"a]["speed"], path["defaults"]["accel"]
"""
import json
import math
import threading
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

import app_paths
import waypoints as wp
import robot_runtime as rr

SPACING_DEFAULT = 0.05
JOINT_JUMP_MAX = 0.5

COLOR_PENDING = (255, 160, 40)
COLOR_OK = (80, 220, 80)
COLOR_FAIL = (220, 60, 60)
COLOR_FK = (74, 163, 255)

spacing = SPACING_DEFAULT
import_raw_json = False
segments = []          # list of segment dicts, index i = wp i → i+1
sample_handles = []
_bake_generation = 0

# Export is manual via File > Export. Bake does not write files.
export_path = False
EXPORT_NAME = "export.json"
SAVE_NAME = "path.arcspath"
DEFAULT_ACC = 0.2
baked_hops = []
baked_points = []
_speed_valid = False
LINEAR_C1 = [1.0 / 3.0, 1.0 / 3.0]
LINEAR_C2 = [2.0 / 3.0, 2.0 / 3.0]
custom_curves = {}
clipboard = None
DEFAULT_BLEND = 0.0
DEFAULT_SPEED = 0.05
FORMAT_VERSION = "beta.0904"
BAKE_SEED_DEG = [0.0, 0.0, 90.0, 90.0, 0.0, 0.0]
_tl_cache = None


def _as3(x, default=(0.0, 0.0, 0.0)):
    if x is None:
        return np.array(default, dtype=float)
    flat = []

    def walk(v):
        if v is None:
            return
        if isinstance(v, (list, tuple)):
            for y in v:
                walk(y)
            return
        if isinstance(v, np.ndarray):
            if v.dtype == object:
                for y in v.reshape(-1):
                    walk(y)
            else:
                for y in np.asarray(v, dtype=float).reshape(-1):
                    flat.append(float(y))
            return
        try:
            flat.append(float(v))
        except Exception:
            pass

    walk(x)
    if len(flat) < 3:
        print("path_bake bad xyz", type(x), getattr(x, "shape", None), x)
        return np.array(default, dtype=float)
    return np.array(flat[:3], dtype=float)


def _as4(x, default=(1.0, 0.0, 0.0, 0.0)):
    if x is None:
        return np.array(default, dtype=float)
    arr = np.asarray(x, dtype=object)
    flat = []
    try:
        flat = [float(v) for v in np.asarray(x, dtype=float).reshape(-1)]
    except Exception:
        if arr.dtype == object:
            for v in arr.reshape(-1):
                try:
                    flat.extend(np.asarray(v, dtype=float).reshape(-1).tolist())
                except Exception:
                    pass
    if len(flat) < 4:
        return np.array(default, dtype=float)
    return np.array(flat[:4], dtype=float)


def _wxyz_to_xyzw(wxyz):
    w, x, y, z = _as4(wxyz)
    return np.array([x, y, z, w], dtype=float)


def _xyzw_to_wxyz(xyzw):
    x, y, z, w = np.asarray(xyzw, dtype=float).reshape(4)
    return np.array([w, x, y, z], dtype=float)


def _slerp_wxyz(w0, w1, u):
    r0 = R.from_quat(_wxyz_to_xyzw(w0))
    r1 = R.from_quat(_wxyz_to_xyzw(w1))
    slerp = Slerp([0.0, 1.0], R.concatenate([r0, r1]))
    return _xyzw_to_wxyz(slerp([float(np.clip(u, 0.0, 1.0))]).as_quat()[0])


def _wp_sig(i):
    w = wp.waypoints[i]
    nxt = wp.waypoints[i + 1] if i + 1 < len(wp.waypoints) else None
    return (
        tuple(np.round(_as3(w.get("pos")), 5)),
        tuple(np.round(_as4(w.get("wxyz")), 5)),
        tuple(np.round(_as3(w.get("out_ctrl_pos")), 5)) if w.get("out_ctrl_pos") is not None else None,
        tuple(np.round(_as3(nxt.get("in_ctrl_pos")), 5)) if nxt is not None and nxt.get("in_ctrl_pos") is not None else None,
        bool(w.get("force_fk")),
        bool(w.get("q_dirty")),
    )


def _dense_cubic(p1, p2, c1, c2, n=80):
    if hasattr(wp, "make_arc_with_controls"):
        return np.asarray(wp.make_arc_with_controls(p1, p2, c1, c2, num_points=n), dtype=float)
    t = np.linspace(0, 1, n).reshape(-1, 1)
    p1, p2, c1, c2 = map(_as3, (p1, p2, c1, c2))
    return (
        (1 - t) ** 3 * p1
        + 3 * (1 - t) ** 2 * t * c1
        + 3 * (1 - t) * t ** 2 * c2
        + t ** 3 * p2
    )


def _polyline_samples(pts, spacing_m):
    pts = np.asarray(pts, dtype=float)
    if pts.ndim == 1:
        if pts.size % 3 == 0:
            pts = pts.reshape(-1, 3)
        else:
            pts = _as3(pts).reshape(1, 3)
    elif pts.ndim >= 2:
        pts = pts.reshape(-1, pts.shape[-1])[:, :3]
    if len(pts) < 2:
        return pts.copy(), np.array([0.0])
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    total = float(s[-1])
    if total < 1e-9:
        return pts[:1].copy(), np.array([0.0])
    out_s = [0.0]
    d = spacing_m
    # Interiors only while a full spacing step still fits. Endpoints always kept,
    # so a short span (or huge spacing) bakes as waypoint-to-waypoint with 0 ticks.
    while d < total - 0.55 * spacing_m:
        out_s.append(d)
        d += spacing_m
    if out_s[-1] < total - 1e-9:
        out_s.append(total)
    out_s = np.array(out_s, dtype=float)
    xyz = np.empty((len(out_s), 3), dtype=float)
    for k, sk in enumerate(out_s):
        j = int(np.searchsorted(s, sk, side="right") - 1)
        j = max(0, min(j, len(seg) - 1))
        span = seg[j] if seg[j] > 1e-12 else 1.0
        u = (sk - s[j]) / span
        xyz[k] = pts[j] * (1 - u) + pts[j + 1] * u
    return xyz, out_s / total


def _point_seg_dist(p, a, b):
    a = np.asarray(a, dtype=float).reshape(3)
    b = np.asarray(b, dtype=float).reshape(3)
    p = np.asarray(p, dtype=float).reshape(3)
    ab = b - a
    den = float(np.dot(ab, ab))
    if den < 1e-18:
        return float(np.linalg.norm(p - a))
    u = max(0.0, min(1.0, float(np.dot(p - a, ab)) / den))
    return float(np.linalg.norm(p - (a + u * ab)))


def _rdp_indices(pts, eps):
    """Ramer–Douglas–Peucker vertex set. Min points for chord error ≤ eps."""
    pts = np.asarray(pts, dtype=float).reshape(-1, 3)
    n = len(pts)
    if n <= 2:
        return list(range(n))
    keep = {0, n - 1}

    def rec(i0, i1):
        if i1 - i0 < 2:
            return
        maxd, maxi = -1.0, i0
        a, b = pts[i0], pts[i1]
        for i in range(i0 + 1, i1):
            d = _point_seg_dist(pts[i], a, b)
            if d > maxd:
                maxd, maxi = d, i
        if maxd > eps:
            keep.add(maxi)
            rec(i0, maxi)
            rec(maxi, i1)

    rec(0, n - 1)
    return sorted(keep)


def _curvature_min_count(pts, spacing_m):
    """Min samples from local radius / sagitta, capped by spacing count.

    For radius R and allowed chord error ε:
        θ = 2 acos(clamp(1 - ε/R, -1, 1))
        Δs ≈ R θ
    Flat spans (R → ∞) need only the two endpoints.
    """
    pts = np.asarray(pts, dtype=float).reshape(-1, 3)
    if len(pts) < 2:
        return 2
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    length = float(np.sum(seg))
    spacing_m = max(1e-4, float(spacing_m))
    n_cap = len(_polyline_samples(pts, spacing_m)[0])
    n_cap = max(2, int(n_cap))
    eps = max(0.002, min(0.5 * spacing_m, 0.02))
    n_rdp = max(2, len(_rdp_indices(pts, eps)))
    # Also estimate from discrete curvature: n ≈ 1 + Σ ds / (2 √(2 R ε))
    extra = 0.0
    for i in range(1, len(pts) - 1):
        v0 = pts[i] - pts[i - 1]
        v1 = pts[i + 1] - pts[i]
        l0 = float(np.linalg.norm(v0))
        l1 = float(np.linalg.norm(v1))
        if l0 < 1e-9 or l1 < 1e-9:
            continue
        cross = np.cross(v0, v1)
        sin_a = float(np.linalg.norm(cross)) / (l0 * l1)
        sin_a = min(1.0, max(0.0, sin_a))
        if sin_a < 1e-6:
            continue
        # R ≈ ds / θ, θ ≈ sin_a for small angles
        ds = 0.5 * (l0 + l1)
        radius = ds / max(sin_a, 1e-6)
        if radius < 1e-4:
            extra += 1.0
            continue
        if eps >= radius:
            extra += 1.0
            continue
        theta = 2.0 * math.acos(max(-1.0, min(1.0, 1.0 - eps / radius)))
        step = max(radius * theta, 1e-4)
        extra += ds / step
    n_curve = 2 + int(math.ceil(extra))
    n_min = max(2, min(n_cap, max(n_rdp, n_curve if extra > 0.35 else n_rdp)))
    if length < 0.5 * spacing_m:
        return 2
    return n_min


def _arc_length_samples(pts, n):
    pts = np.asarray(pts, dtype=float).reshape(-1, 3)
    n = max(2, int(n))
    if len(pts) < 2:
        p = pts[0] if len(pts) else np.zeros(3)
        return np.vstack([p, p]), np.array([0.0, 1.0])
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    total = float(s[-1])
    if total < 1e-9:
        return np.vstack([pts[0], pts[-1]]), np.array([0.0, 1.0])
    out_s = np.linspace(0.0, total, n)
    xyz = np.empty((n, 3), dtype=float)
    for k, sk in enumerate(out_s):
        j = int(np.searchsorted(s, sk, side="right") - 1)
        j = max(0, min(j, len(seg) - 1))
        span = seg[j] if seg[j] > 1e-12 else 1.0
        u = (sk - s[j]) / span
        xyz[k] = pts[j] * (1 - u) + pts[j + 1] * u
    return xyz, out_s / total


def _fill_segment_n(i, n):
    """Rebuild samples for segment i with exactly n points (including ends)."""
    a = wp.waypoints[i]
    b = wp.waypoints[i + 1]
    if _same_pose(a, b):
        segments[i] = _build_segment(i)
        return segments[i]
    pts = _segment_curve_pts(i)
    n = max(2, int(n))
    xyz, u = _arc_length_samples(pts, n)
    w0 = _as4(a.get("wxyz"))
    w1 = _as4(b.get("wxyz", w0))
    pts = np.asarray(pts, dtype=float).reshape(-1, 3)
    length = float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1))) if len(pts) >= 2 else 0.0
    s_abs = np.asarray(u, dtype=float).reshape(-1) * max(length, 1e-9)
    old = segments[i] if i < len(segments) else {}
    samples = [_empty_sample(xyz[k], _slerp_wxyz(w0, w1, u[k]), u[k], s_abs[k]) for k in range(len(xyz))]
    segments[i] = {
        "i0": i,
        "i1": i + 1,
        "status": "pending",
        "spacing": float(spacing),
        "sig": (_wp_sig(i), _wp_sig(i + 1)),
        "samples": samples,
        "fail_index": None,
        "speed": None,
        "accel": None,
        "dwell": False,
        "length": float(length),
        "transition": old.get("transition") if old else None,
    }
    _aim_samples_at_track(i, samples)
    segments[i]["samples"] = samples
    return segments[i]


def _segment_curve_pts(i):
    a = wp.waypoints[i]
    b = wp.waypoints[i + 1]
    p1 = _as3(a.get("pos"))
    p2 = _as3(b.get("pos"))
    ca = str(a.get("clip_id") or "")
    cb = str(b.get("clip_id") or "")
    if ca != cb:
        return np.stack([p1, p2], axis=0)
    c1 = a.get("out_ctrl_pos")
    c2 = b.get("in_ctrl_pos")
    if c1 is None or c2 is None:
        return np.stack([p1, p2], axis=0)
    return _dense_cubic(p1, p2, c1, c2)


def _track_spec_for_seg(i):
    wps = wp.waypoints
    for j in (i, i + 1):
        if 0 <= j < len(wps):
            rec = wps[j].get("track_object") or {}
            if rec.get("on") and rec.get("name"):
                return rec
    return None


def _aim_samples_at_track(i, samples):
    spec = _track_spec_for_seg(i)
    if not spec or not hasattr(rr, "look_at_wxyz_offset"):
        return
    for sm in samples or []:
        if sm.get("pos") is None:
            continue
        wxyz = rr.look_at_wxyz_offset(sm["pos"], spec.get("name"), spec.get("offset"), spec.get("pos_offset"))
        if wxyz is not None:
            sm["wxyz"] = np.asarray(wxyz, dtype=float).reshape(4)


def update_look_bars():
    """Look At timeline bars, rebuilt on bake only: one bar per run of spans
    aimed at the same target (same rule the bake uses to aim samples)."""
    bars = []
    wps = wp.waypoints
    for i in range(max(0, len(wps) - 1)):
        spec = _track_spec_for_seg(i)
        if not spec:
            continue
        t0, t1 = wps[i].get("t"), wps[i + 1].get("t")
        if t0 is None or t1 is None:
            continue
        name = str(spec.get("name") or "")
        t0, t1 = float(min(t0, t1)), float(max(t0, t1))
        if bars and bars[-1]["name"] == name and abs(bars[-1]["t1"] - t0) < 1e-6:
            bars[-1]["t1"] = t1
        else:
            bars.append({"t0": t0, "t1": t1, "name": name})
    wp.look_bars = bars
    return bars


def _empty_sample(pos, wxyz, u, s_abs):
    return {
        "pos": _as3(pos),
        "wxyz": _as4(wxyz),
        "u": float(u),
        "s": float(s_abs),
        "q": None,
        "t": None,
        "speed": None,
        "accel": None,
    }


def _same_pose(a, b, pos_eps=1e-4, quat_eps=1e-3):
    pa = _as3(a.get("pos"))
    pb = _as3(b.get("pos"))
    if float(np.linalg.norm(pa - pb)) > pos_eps:
        return False
    qa = np.asarray(_as4(a.get("wxyz")), dtype=float)
    qb = np.asarray(_as4(b.get("wxyz")), dtype=float)
    if qa[0] < 0:
        qa = -qa
    if qb[0] < 0:
        qb = -qb
    return float(np.linalg.norm(qa - qb)) <= quat_eps


def _seg_force_fk(i):
    return bool(i < len(wp.waypoints) and wp.waypoints[i].get("force_fk"))


def _home_q():
    try:
        return np.deg2rad(np.array(rr.HOME_Q_DEG, dtype=float)).reshape(-1)[:6]
    except Exception:
        return bake_seed_q()


def _usable_wp_q(w):
    if not w or w.get("q") is None or w.get("q_dirty"):
        return None
    return np.asarray(w.get("q"), dtype=float).reshape(-1)[:6]


def _prev_knot_q(i, seed=None):
    if i > 0 and i - 1 < len(segments):
        sms = segments[i - 1].get("samples") or []
        for sm in reversed(sms):
            if sm.get("q") is not None:
                return np.asarray(sm["q"], dtype=float).reshape(-1)[:6]
    if i > 0 and i - 1 < len(wp.waypoints):
        q = _usable_wp_q(wp.waypoints[i - 1])
        if q is not None:
            return q
    return None if seed is None else np.asarray(seed, dtype=float).reshape(-1)[:6]


def _build_segment(i):
    a = wp.waypoints[i]
    b = wp.waypoints[i + 1]
    if _seg_force_fk(i):
        w0 = _as4(a.get("wxyz"))
        samples = [
            _empty_sample(_as3(a.get("pos")), w0, 0.0, 0.0),
            _empty_sample(_as3(b.get("pos")), _as4(b.get("wxyz", w0)), 1.0, 0.0),
        ]
        qa = a.get("q")
        qb = b.get("q")
        if qa is not None and not a.get("q_dirty"):
            samples[0]["q"] = np.asarray(qa, dtype=float).reshape(-1)[:6]
        if qb is not None and not b.get("q_dirty"):
            samples[-1]["q"] = np.asarray(qb, dtype=float).reshape(-1)[:6]
        _aim_samples_at_track(i, samples)
        ready = samples[0].get("q") is not None and samples[-1].get("q") is not None
        return {
            "i0": i,
            "i1": i + 1,
            "status": "pending",
            "spacing": float(spacing),
            "sig": (_wp_sig(i), _wp_sig(i + 1)),
            "samples": samples,
            "fail_index": None,
            "speed": None,
            "accel": None,
            "dwell": False,
            "force_fk": True,
            "length": float(np.linalg.norm(_as3(b.get("pos")) - _as3(a.get("pos")))),
            "transition": a.get("transition") or None,
        }
    if _same_pose(a, b):
        w0 = _as4(a.get("wxyz"))
        p0 = _as3(a.get("pos"))
        samples = [
            _empty_sample(p0, w0, 0.0, 0.0),
            _empty_sample(_as3(b.get("pos")), _as4(b.get("wxyz", w0)), 1.0, 0.0),
        ]
        _aim_samples_at_track(i, samples)
        return {
            "i0": i,
            "i1": i + 1,
            "status": "pending",
            "spacing": float(spacing),
            "sig": (_wp_sig(i), _wp_sig(i + 1)),
            "samples": samples,
            "fail_index": None,
            "speed": None,
            "accel": None,
            "dwell": True,
            "length": 0.0,
            "transition": "linear",
        }
    pts = _segment_curve_pts(i)
    xyz, u = _polyline_samples(pts, spacing)
    w0 = _as4(a.get("wxyz"))
    w1 = _as4(b.get("wxyz", w0))
    pts = np.asarray(pts, dtype=float).reshape(-1, 3)
    chord = float(np.linalg.norm(pts[-1] - pts[0])) if len(pts) else 0.0
    if len(pts) >= 2:
        length = float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))
    else:
        length = chord
    s_abs = np.asarray(u, dtype=float).reshape(-1) * max(chord, length, 1e-9)
    samples = [_empty_sample(xyz[k], _slerp_wxyz(w0, w1, u[k]), u[k], s_abs[k]) for k in range(len(xyz))]
    if len(samples) < 2:
        samples = [
            _empty_sample(_as3(a.get("pos")), w0, 0.0, 0.0),
            _empty_sample(_as3(b.get("pos")), w1, 1.0, float(length)),
        ]
    _aim_samples_at_track(i, samples)
    return {
        "i0": i,
        "i1": i + 1,
        "status": "pending",
        "spacing": float(spacing),
        "sig": (_wp_sig(i), _wp_sig(i + 1)),
        "samples": samples,
        "fail_index": None,
        "speed": None,
        "accel": None,
        "dwell": False,
        "length": float(length),
        "transition": None,
    }


show_viewer_samples = True
show_tl_samples = True


def clear_bake():
    """Drop IK/speed bake and timeline sample nodes. Keep waypoints."""
    global baked_hops, baked_points, _speed_valid, segments, _tl_cache, _bake_generation
    baked_hops = []
    baked_points = []
    _speed_valid = False
    _tl_cache = None
    _bake_generation += 1
    _sync_segment_count()
    for i in range(len(segments)):
        segments[i] = _build_segment(i)
        segments[i]["status"] = "pending"
        segments[i]["timing"] = None
        segments[i]["micro"] = []
        segments[i]["fail_index"] = None
        for sm in segments[i].get("samples") or []:
            sm["q"] = None
            sm["t"] = None
            sm["timing"] = None
            sm["speed"] = None
    clear_sample_dots()
    if hasattr(wp, "draw_path"):
        wp.draw_path()
    _tl_cache = None
    return {"ok": True, "segments": len(segments), "bake": status_dict()}


# Dots are redrawn from the viewer thread (path window follow) and from
# request threads (bake, edits). Unlocked, two redraws could each clear the
# other's list and leave dots in the scene that no list owns.
_dots_lock = threading.RLock()
_dots_high = 0


def clear_sample_dots():
    global sample_handles
    with _dots_lock:
        for h in sample_handles:
            try:
                h.remove()
            except Exception:
                pass
        sample_handles = []


def _draw_sample_dots():
    global _dots_high
    with _dots_lock:
        _draw_sample_dots_locked()
        n = len(sample_handles)
        srv = getattr(wp, "server", None)
        if srv is not None:
            # Names below n were just redrawn (a name replaces its node); drop
            # any higher name an earlier draw left behind.
            for k in range(n, _dots_high):
                try:
                    srv.scene.remove_by_name(f"/bake_pt/{k}")
                except Exception:
                    pass
        _dots_high = n


def _draw_sample_dots_locked():
    """Dots only on calculated segments: green if solved, red if IK failed.
    Pending (not yet baked) segments stay empty. Need 2+ waypoints."""
    clear_sample_dots()
    if not show_viewer_samples:
        return
    if wp.server is None or len(getattr(wp, "waypoints", [])) < 2:
        return
    k = 0
    for seg in segments:
        st = seg.get("status")
        if st not in ("ok", "fail"):
            continue
        samples = seg.get("samples") or []
        fail_i = seg.get("fail_index")
        for si, sm in enumerate(samples):
            if si == 0 or si == len(samples) - 1:
                continue
            if wp.in_path_window(sm.get("t")):
                continue
            if fail_i is not None and si == fail_i:
                continue
            if st == "ok" or sm.get("q") is not None:
                color = COLOR_OK
            else:
                color = COLOR_FAIL
            try:
                h = wp.server.scene.add_icosphere(
                    name=f"/bake_pt/{k}",
                    radius=0.006,
                    color=color,
                    position=tuple(float(v) for v in sm["pos"]),
                )
                sample_handles.append(h)
                k += 1
            except Exception:
                pass
        if st == "fail" and fail_i is not None and 0 <= fail_i < len(samples):
            sm = samples[fail_i]
            try:
                h = wp.server.scene.add_icosphere(
                    name=f"/bake_pt/{k}",
                    radius=0.009,
                    color=COLOR_FAIL,
                    position=tuple(float(v) for v in sm["pos"]),
                )
                sample_handles.append(h)
                k += 1
            except Exception:
                pass


def _solve_force_fk_segment(i, seg, seed=None):
    """IK only the two diamonds. Skip interior sample generation.

    Start seed: previous knot/sample q.
    End seed: start q, then home, then sim pose at bake time.
    Already-tested endpoints (q present, not dirty) are not re-solved.
    """
    seg["force_fk"] = True
    if i + 1 >= len(wp.waypoints):
        seg["status"] = "fail"
        return None
    a = wp.waypoints[i]
    b = wp.waypoints[i + 1]
    sms = seg.get("samples") or []
    if len(sms) != 2:
        sms = [
            _empty_sample(_as3(a.get("pos")), _as4(a.get("wxyz")), 0.0, 0.0),
            _empty_sample(_as3(b.get("pos")), _as4(b.get("wxyz")), 1.0, 0.0),
        ]
        seg["samples"] = sms
    else:
        sms[0]["pos"] = _as3(a.get("pos"))
        sms[0]["wxyz"] = _as4(a.get("wxyz"))
        sms[-1]["pos"] = _as3(b.get("pos"))
        sms[-1]["wxyz"] = _as4(b.get("wxyz"))
    qa = None if a.get("q_dirty") else (sms[0].get("q") if sms[0].get("q") is not None else _usable_wp_q(a))
    if qa is None:
        qa = _ik_sample(sms[0], _prev_knot_q(i, seed), allow_jump=True)
    if qa is None:
        seg["status"] = "fail"
        seg["fail_index"] = 0
        return None
    qa = np.asarray(qa, dtype=float).reshape(-1)[:6]
    sms[0]["q"] = qa
    a["q"] = qa
    a["q_dirty"] = False
    qb = None if b.get("q_dirty") else (sms[-1].get("q") if sms[-1].get("q") is not None else _usable_wp_q(b))
    if qb is None:
        sim_q = None
        if seed is not None:
            sim_q = np.asarray(seed, dtype=float).reshape(-1)[:6]
        elif rr.robot_sim is not None:
            try:
                sim_q = np.asarray(rr.robot_sim.angle_vector(), dtype=float).reshape(-1)[:6]
            except Exception:
                sim_q = None
        for s in (qa, _home_q(), sim_q):
            if s is None:
                continue
            q = _ik_sample(sms[-1], s, allow_jump=True)
            if q is not None:
                qb = q
                break
    if qb is None:
        seg["status"] = "fail"
        seg["fail_index"] = 1
        return None
    qb = np.asarray(qb, dtype=float).reshape(-1)[:6]
    sms[-1]["q"] = qb
    b["q"] = qb
    b["q_dirty"] = False
    seg["status"] = "ok"
    seg["fail_index"] = None
    return qb


def segment_color(i):
    if _seg_force_fk(i):
        if i < len(segments) and segments[i].get("status") == "fail":
            return COLOR_FAIL
        if i < len(segments) and segments[i].get("timing") == "too_fast":
            return COLOR_PENDING
        return COLOR_FK
    if i >= len(segments):
        return COLOR_PENDING
    st = segments[i]["status"]
    if st == "ok":
        return COLOR_OK
    if st == "fail":
        return COLOR_FAIL
    return COLOR_PENDING


def set_spacing(value):
    global spacing
    try:
        value = float(value)
    except Exception:
        value = SPACING_DEFAULT
    value = max(0.005, min(1.0, value))
    if abs(value - spacing) > 1e-9:
        spacing = value
        invalidate_from(0, drop_all=True, keep_times=True)
    return spacing


def clear_sample_time_locks(from_seg=0):
    for j, seg in enumerate(segments):
        if j < from_seg:
            continue
        for sm in seg.get("samples") or []:
            sm.pop("t_lock", None)


selected_sample = None


def _segment_sig_matches(j):
    if j < 0 or j >= len(segments) or j + 1 >= len(wp.waypoints):
        return False
    return segments[j].get("sig") == (_wp_sig(j), _wp_sig(j + 1))


def invalidate_from(idx, drop_all=False, keep_times=None):
    """idx = waypoint index that changed. Dirties segment idx-1 and all after.

    keep_times (default: not drop_all) carries existing sample frames into the
    rebuilt spans. New-document resets pass drop_all and start clean.
    """
    if keep_times is None:
        keep_times = not drop_all
    global baked_hops, baked_points, _bake_generation, _tl_cache, _speed_valid
    start = 0 if drop_all else max(0, idx - 1)
    if not drop_all and segments:
        changed = False
        n = min(len(segments), max(0, len(wp.waypoints) - 1))
        if n != len(segments):
            changed = True
        else:
            for j in range(start, n):
                if not _segment_sig_matches(j):
                    changed = True
                    break
        if not changed:
            return
    _bake_generation += 1
    _tl_cache = None
    if not segments:
        _sync_segment_count()
        _draw_sample_dots()
        if hasattr(wp, "draw_path"):
            wp.draw_path()
        return
    baked_hops = []
    baked_points = []
    _speed_valid = False
    old_segs = {j: dict(segments[j]) for j in range(start, len(segments))} if keep_times else {}
    for j in range(start, len(segments)):
        segments[j]["status"] = "pending"
        segments[j]["samples"] = []
        segments[j]["fail_index"] = None
        segments[j]["sig"] = None
        segments[j]["micro"] = []
        segments[j]["timing"] = None
    _sync_segment_count()
    for j in range(start, len(segments)):
        segments[j] = _build_segment(j)
        if j in old_segs:
            _carry_sample_times(old_segs[j], segments[j], j)
    _draw_sample_dots()
    if hasattr(wp, "draw_path"):
        wp.draw_path()


def on_waypoint_moved(idx):
    if idx is None:
        return
    try:
        idx = int(idx)
    except Exception:
        return
    invalidate_from(idx)


def _sync_segment_count():
    global segments
    n = max(0, len(wp.waypoints) - 1)
    if len(segments) > n:
        segments = segments[:n]
    while len(segments) < n:
        segments.append(_build_segment(len(segments)))


def rebuild_preview():
    global _bake_generation, _tl_cache
    _bake_generation += 1
    _tl_cache = None
    _sync_segment_count()
    for i in range(len(segments)):
        sig = (_wp_sig(i), _wp_sig(i + 1))
        if segments[i].get("sig") != sig or not segments[i].get("samples") or segments[i].get("spacing") != spacing:
            old = segments[i]
            old_tr = old.get("transition")
            segments[i] = _build_segment(i)
            if old_tr:
                segments[i]["transition"] = old_tr
            _carry_sample_times(old, segments[i], i)
    _draw_sample_dots()
    if hasattr(wp, "draw_path"):
        wp.draw_path()


def _carry_sample_times(old, new, i):
    """Keep the sample frames a span already had.

    Same sample count: frames are matched by index and kept (with any t_lock)
    while they stay in order inside the span. Different count (the span got
    longer or shorter): the old frames no longer line up by index, so the old
    timing shape is resampled at the new samples' index fractions. Evenly
    spaced frames stay evenly spaced; a hand-shaped span keeps its shape.
    """
    olds = (old or {}).get("samples") or []
    news = new.get("samples") or []
    if len(olds) < 3 or len(news) < 3 or i + 1 >= len(wp.waypoints):
        return
    t0 = float(wp.waypoints[i].get("t") or 0.0)
    t1 = float(wp.waypoints[i + 1].get("t") or t0)
    # Same span only: after an insert or delete, index i is a different span.
    ta, tb = olds[0].get("t"), olds[-1].get("t")
    if ta is None or tb is None or abs(float(ta) - t0) > 1e-6 or abs(float(tb) - t1) > 1e-6:
        return
    if len(olds) != len(news):
        ts = [sm.get("t") for sm in olds]
        if any(t is None for t in ts):
            return
        ts = [float(t) for t in ts]
        if any(b <= a for a, b in zip(ts, ts[1:])):
            return
        fo = np.linspace(0.0, 1.0, len(ts))
        fn = np.linspace(0.0, 1.0, len(news))
        tn = np.interp(fn, fo, ts)
        for k in range(1, len(news) - 1):
            news[k]["t"] = float(tn[k])
        return
    last = t0
    for k in range(1, len(news) - 1):
        t = olds[k].get("t")
        if t is None or not (last < float(t) < t1):
            break
        news[k]["t"] = float(t)
        if olds[k].get("t_lock") is not None:
            news[k]["t_lock"] = olds[k]["t_lock"]
        last = float(t)


def _fill_missing_times(samples):
    """Equal spacing for interior samples with no time, between their timed neighbours."""
    k = 1
    n = len(samples)
    while k < n - 1:
        if samples[k].get("t") is not None:
            k += 1
            continue
        a = k - 1
        b = k
        while b < n - 1 and samples[b].get("t") is None:
            b += 1
        ta = float(samples[a]["t"])
        tb = float(samples[b]["t"])
        for j in range(a + 1, b):
            samples[j]["t"] = ta + (tb - ta) * float(j - a) / float(b - a)
        k = b


def bake_seed_q():
    return np.deg2rad(np.array(BAKE_SEED_DEG, dtype=float))


def _unwrap_toward(q, ref):
    q = np.asarray(q, dtype=float).reshape(-1)[:6]
    if ref is None:
        return q
    ref = np.asarray(ref, dtype=float).reshape(-1)[:6]
    out = q.copy()
    for j in range(6):
        d = out[j] - ref[j]
        while d > np.pi:
            out[j] -= 2.0 * np.pi
            d = out[j] - ref[j]
        while d < -np.pi:
            out[j] += 2.0 * np.pi
            d = out[j] - ref[j]
    return out


def _lerp_q(qa, qb, u):
    qa = np.asarray(qa, dtype=float).reshape(-1)[:6]
    qb = _unwrap_toward(qb, qa)
    u = max(0.0, min(1.0, float(u)))
    return qa * (1.0 - u) + qb * u


def _wrap_pi(q):
    q = np.asarray(q, dtype=float).reshape(-1)[:6]
    return ((q + np.pi) % (2.0 * np.pi)) - np.pi


def _ik_score(q):
    q = np.asarray(q, dtype=float).reshape(-1)[:6]
    in_pi = int(np.sum(np.abs(q) <= np.pi + 1e-6))
    return (in_pi, -float(np.sum(np.abs(q))))


def _fk_pos(q):
    q = np.asarray(q, dtype=float).reshape(-1)[:6]
    try:
        model = getattr(rr, "robot_sim", None)
        if model is None:
            return None
        hold = np.asarray(model.angle_vector(), dtype=float).reshape(-1)[:6]
        model.angle_vector(q)
        if hasattr(rr, "_cursor_pose_from_ee"):
            p, _ = rr._cursor_pose_from_ee(model, "sim")
        else:
            ee = rr._ee_link(model)
            p = np.asarray(ee.worldpos(), dtype=float).reshape(3)
        model.angle_vector(hold)
        return np.asarray(p, dtype=float).reshape(3)
    except Exception:
        return None


def _q_matches_pos(q, pos, tol=0.025):
    if q is None or pos is None:
        return False
    fk = _fk_pos(q)
    if fk is None:
        return False
    return float(np.linalg.norm(fk - np.asarray(pos, dtype=float).reshape(3))) <= float(tol)


def _q_same(a, b, tol=0.012):
    if a is None or b is None:
        return False
    a = np.asarray(a, dtype=float).reshape(-1)[:6]
    b = np.asarray(b, dtype=float).reshape(-1)[:6]
    return float(np.max(np.abs(_unwrap_toward(a, b) - b))) <= tol


def _ik_try(sm, seed=None):
    pos = sm["pos"]
    wxyz = sm["wxyz"]
    if hasattr(rr, "_ee_from_cursor"):
        pos, wxyz = rr._ee_from_cursor(pos, wxyz, "sim")

    def _solve(sd):
        q = None
        ik = sm.get("ik")
        if ik is not None and hasattr(rr, "_ik_to_xyzrpy"):
            q = rr._ik_to_xyzrpy(ik, seed=sd)
        if q is None:
            q = rr._ik_to_pose(pos, wxyz, seed=sd) if sd is not None else rr._ik_to_pose(pos, wxyz)
        if q is None and hasattr(rr, "sim_ik"):
            try:
                q = rr.sim_ik(pos, wxyz, seed=sd)
            except Exception:
                q = None
        return None if q is None else np.asarray(q, dtype=float).reshape(-1)[:6]

    q = _solve(seed)
    if q is not None and seed is not None and _q_same(q, seed):
        q2 = _solve(None)
        if q2 is not None and not _q_same(q2, seed):
            q = q2
        else:
            q = None
    return q


def _ik_sample(sm, seed, allow_jump=False, first=False, fresh=False):
    """Chain IK from previous q. Original unseeded solve is fallback.
    fresh=True: path start or first diamond after a force-FK span."""
    seed_q = None
    if seed is not None:
        seed_q = np.asarray(seed, dtype=float).reshape(-1)[:6]
        if seed_q.size < 6:
            seed_q = None

    def _ok(q):
        if q is None:
            return None
        q = np.asarray(q, dtype=float).reshape(-1)[:6]
        if seed_q is not None:
            q = _unwrap_toward(q, seed_q)
            if not allow_jump:
                jump = float(np.max(np.abs(q - seed_q[:6])))
                if jump > JOINT_JUMP_MAX:
                    print("bake jump", jump)
                    return None
        return q

    seed_for_ik = _wrap_pi(seed_q) if seed_q is not None else None
    if not fresh and seed_for_ik is not None:
        q = _ok(_ik_try(sm, seed_for_ik))
        if q is not None:
            return q
    q = _ok(_ik_try(sm, None))
    if q is not None:
        return q
    if seed_for_ik is not None:
        q = _ok(_ik_try(sm, seed_for_ik))
        if q is not None:
            return q
    return None


_baking = False


def is_baking():
    return bool(_baking)


def run_bake(spacing_m=None):
    global _speed_valid, _baking
    if _baking:
        return {"ok": False, "error": "bake in progress", "baked": 0, "total": len(segments)}
    _baking = True
    try:
        return _run_bake_body(spacing_m)
    finally:
        _baking = False


def _run_bake_body(spacing_m=None):
    global _speed_valid
    if spacing_m is not None:
        set_spacing(spacing_m)
    if len(wp.waypoints) < 2:
        return {"ok": False, "error": "need at least 2 waypoints", "baked": 0, "total": 0}

    rebuild_preview()
    seed = None
    if rr.robot_sim is not None:
        try:
            seed = np.array(rr.robot_sim.angle_vector(), dtype=float).reshape(-1)[:6]
        except Exception:
            seed = None
    if getattr(rr, "_last_good_q", None) is not None:
        seed = np.array(rr._last_good_q, dtype=float).reshape(-1)[:6]
    baked = 0
    failed_at = None
    prev_force_fk = False
    # Re-bake from the first hole so a pending first span cannot sit
    # in front of leftover "ok" spans from an older waypoint order.
    hole = 0
    for i, seg in enumerate(segments):
        sig = (_wp_sig(i), _wp_sig(i + 1))
        qs = [sm for sm in (seg.get("samples") or []) if sm.get("q") is not None]
        good = (
            seg.get("status") == "ok"
            and seg.get("sig") == sig
            and seg.get("spacing") == spacing
            and len(qs) >= 2
        )
        if not good:
            hole = i
            break
    else:
        hole = len(segments)
    for i, seg in enumerate(segments):
        if i < hole:
            if seg["samples"] and seg["samples"][-1].get("q") is not None:
                seed = seg["samples"][-1]["q"]
            baked += 1
            continue
        if failed_at is not None:
            break
        ok = True
        if _seg_force_fk(i) or seg.get("force_fk"):
            q_end = _solve_force_fk_segment(i, seg, seed)
            if q_end is not None:
                seed = q_end
                baked += 1
                prev_force_fk = True
                continue
            failed_at = i
            print("bake fail fk seg", i)
            break
        if seg.get("dwell") and (seg.get("samples") or []):
            qa = wp.waypoints[i].get("q")
            if qa is None:
                qa = seed
            if qa is None:
                qa = wp.waypoints[i + 1].get("q")
            q_hold = None if qa is None else np.asarray(qa, dtype=float).reshape(-1)[:6]
            if q_hold is not None:
                for sm in seg["samples"]:
                    sm["q"] = q_hold.copy()
                try:
                    wp.waypoints[i + 1]["q"] = q_hold.copy()
                    wp.waypoints[i + 1]["q_dirty"] = False
                except Exception:
                    pass
            if seg["samples"][0].get("q") is not None and seg["samples"][-1].get("q") is not None:
                seed = seg["samples"][-1]["q"]
                seg["status"] = "ok"
                baked += 1
                continue
        sms = seg.get("samples") or []
        qa_wp = wp.waypoints[i].get("q") if i < len(wp.waypoints) else None
        qb_wp = wp.waypoints[i + 1].get("q") if i + 1 < len(wp.waypoints) else None
        if qa_wp is not None and wp.waypoints[i].get("q_dirty"):
            qa_wp = None
        if qb_wp is not None and wp.waypoints[i + 1].get("q_dirty"):
            qb_wp = None
        if qa_wp is not None and not _q_matches_pos(qa_wp, wp.waypoints[i].get("pos")):
            qa_wp = None
        if qb_wp is not None and not _q_matches_pos(qb_wp, wp.waypoints[i + 1].get("pos")):
            qb_wp = None
        if qa_wp is not None and qb_wp is not None and _q_same(qa_wp, qb_wp):
            qb_wp = None
        nsm = len(sms)
        for k, sm in enumerate(sms):
            fresh = bool(k == 0 and (i == 0 or prev_force_fk))
            q = None
            try:
                u = float(sm["u"])
            except (KeyError, TypeError, ValueError):
                u = 0.0 if k == 0 else (1.0 if k == nsm - 1 else (k / max(1, nsm - 1)))
            if k == 0 and qa_wp is not None:
                q = _unwrap_toward(qa_wp, seed)
            elif k == nsm - 1 and qb_wp is not None:
                q = _unwrap_toward(qb_wp, seed if seed is not None else qa_wp)
            elif qa_wp is not None and qb_wp is not None:
                q_lerp = _lerp_q(qa_wp, qb_wp, u)
                q_ik = _ik_sample(sm, q_lerp, allow_jump=False, fresh=False)
                if q_ik is not None and float(np.max(np.abs(_unwrap_toward(q_ik, q_lerp) - q_lerp))) <= JOINT_JUMP_MAX:
                    q = q_ik
                else:
                    q = _ik_sample(sm, seed, allow_jump=False, fresh=False)
                    if q is None:
                        print("bake fail no IK (refused joint lerp) seg", i, "sample", k)
            else:
                q = _ik_sample(sm, seed, allow_jump=fresh, fresh=fresh)
            if q is None:
                ok = False
                seg["fail_index"] = k
                break
            if k > 0:
                prev_sm = sms[k - 1]
                try:
                    p0 = np.asarray(prev_sm.get("pos"), dtype=float).reshape(3)
                    p1 = np.asarray(sm.get("pos"), dtype=float).reshape(3)
                    moved = float(np.linalg.norm(p1 - p0)) > 0.008
                except Exception:
                    moved = True
                if moved and _q_same(q, prev_sm.get("q")):
                    print("bake fail same-q different pose seg", i, "sample", k)
                    ok = False
                    seg["fail_index"] = k
                    break
            sm["q"] = q
            seed = q
            if k == 0 and i < len(wp.waypoints):
                wp.waypoints[i]["q"] = _wrap_pi(q)
                wp.waypoints[i]["q_dirty"] = False
            if k == nsm - 1 and i + 1 < len(wp.waypoints):
                wp.waypoints[i + 1]["q"] = _wrap_pi(q)
                wp.waypoints[i + 1]["q_dirty"] = False
        prev_force_fk = False
        if ok:
            seg["status"] = "ok"
            baked += 1
            # Speed is scored once after every segment has IK.
        else:
            seg["status"] = "fail"
            failed_at = i
            print("bake fail seg", i, "sample", seg.get("fail_index"))
            break

    if failed_at is not None:
        for j in range(failed_at + 1, len(segments)):
            segments[j]["status"] = "pending"
            segments[j]["timing"] = None
            segments[j]["micro"] = []

    if hasattr(wp, "draw_path"):
        wp.draw_path()
    assign_sample_times()
    evaluate_timing(force=True)
    _speed_valid = True
    store_bake_motion()
    update_look_bars()
    print(
        "bake done",
        "ok", baked, "/", len(segments),
        "samples", sum(len(s.get("samples") or []) for s in segments),
        "hops", len(baked_hops),
        "timing", [s.get("timing") for s in segments],
    )
    try:
        write_overspeed_report()
    except Exception as e:
        print("overspeed report failed:", e)
    _draw_sample_dots()

    return {
        "ok": failed_at is None,
        "baked": baked,
        "total": len(segments),
        "failed_at": failed_at,
        "spacing": spacing,
    }


def _bake_seed():
    seed = None
    if rr.robot_sim is not None:
        try:
            seed = np.array(rr.robot_sim.angle_vector(), dtype=float).reshape(-1)[:6]
        except Exception:
            seed = None
    if getattr(rr, "_last_good_q", None) is not None:
        seed = np.array(rr._last_good_q, dtype=float).reshape(-1)[:6]
    return seed


def _ik_segment(seg, seed):
    ok = True
    for k, sm in enumerate(seg.get("samples") or []):
        q = _ik_sample(sm, seed, allow_jump=(k == 0))
        if q is None:
            seg["fail_index"] = k
            return False, seed
        sm["q"] = q
        seed = q
    if ok:
        seg["status"] = "ok"
        seg["fail_index"] = None
    return True, seed


def _finish_bake(baked, failed_at):
    global _speed_valid
    if hasattr(wp, "draw_path"):
        wp.draw_path()
    assign_sample_times()
    evaluate_timing(force=True)
    _speed_valid = True
    store_bake_motion()
    update_look_bars()
    print(
        "bake done",
        "ok", baked, "/", len(segments),
        "samples", sum(len(s.get("samples") or []) for s in segments),
        "hops", len(baked_hops),
        "timing", [s.get("timing") for s in segments],
    )
    try:
        write_overspeed_report()
    except Exception as e:
        print("overspeed report failed:", e)
    _draw_sample_dots()
    return {
        "ok": failed_at is None,
        "baked": baked,
        "total": len(segments),
        "failed_at": failed_at,
        "spacing": spacing,
    }


def run_optomise(spacing_m=None):
    """Bake each span with the fewest samples that still IK.

    Start from curvature / RDP minimum. If IK fails, add one sample and
    retry until the count matches the spacing-box preset, then fail.
    """
    global _baking
    if _baking:
        return {"ok": False, "error": "bake in progress", "baked": 0, "total": len(segments)}
    _baking = True
    try:
        if import_raw_json:
            return _run_bake_body(spacing_m)
        return _run_optomise_body(spacing_m)
    finally:
        _baking = False


def _run_optomise_body(spacing_m=None):
    if spacing_m is not None:
        set_spacing(spacing_m)
    if len(wp.waypoints) < 2:
        return {"ok": False, "error": "need at least 2 waypoints", "baked": 0, "total": 0}
    rebuild_preview()
    seed = _bake_seed()
    baked = 0
    failed_at = None
    used = []
    for i, seg in enumerate(segments):
        if failed_at is not None:
            seg["status"] = "pending"
            seg["timing"] = None
            seg["micro"] = []
            continue
        if seg.get("dwell"):
            ok, seed = True, seed
            qa = wp.waypoints[i].get("q")
            if qa is None:
                qa = seed
            if qa is None:
                qa = wp.waypoints[i + 1].get("q")
            q_hold = None if qa is None else np.asarray(qa, dtype=float).reshape(-1)[:6]
            if q_hold is not None:
                for sm in (seg.get("samples") or []):
                    sm["q"] = q_hold.copy()
                try:
                    wp.waypoints[i + 1]["q"] = q_hold.copy()
                    wp.waypoints[i + 1]["q_dirty"] = False
                except Exception:
                    pass
            if (seg.get("samples") or []) and seg["samples"][0].get("q") is not None and seg["samples"][-1].get("q") is not None:
                seed = seg["samples"][-1]["q"]
                seg["status"] = "ok"
                baked += 1
                used.append({"seg": i, "n": 2, "n_max": 2, "n_min": 2})
                continue
        pts = _segment_curve_pts(i)
        n_max = max(2, len(_polyline_samples(pts, spacing)[0]))
        n_min = _curvature_min_count(pts, spacing)
        n_min = max(2, min(n_min, n_max))
        ok = False
        n_used = n_min
        for n in range(n_min, n_max + 1):
            _fill_segment_n(i, n)
            seg = segments[i]
            ok, seed_try = _ik_segment(seg, seed)
            if ok:
                seed = seed_try
                n_used = n
                baked += 1
                break
        used.append({"seg": i, "n": n_used, "n_min": n_min, "n_max": n_max, "ok": ok})
        if not ok:
            segments[i]["status"] = "fail"
            failed_at = i
            print("optomise fail seg", i, "n", n_max)
            break
        else:
            print("optomise seg", i, "n", n_used, "of", n_max, "min", n_min)
    result = _finish_bake(baked, failed_at)
    result["optomise"] = used
    return result


bake = run_bake


def _sample_xyzrpy(sm):
    pos = sm.get("pos")
    wxyz = sm.get("wxyz")
    if pos is None or wxyz is None:
        return None
    if hasattr(rr, "_ee_from_cursor"):
        pos, wxyz = rr._ee_from_cursor(pos, wxyz, "sim")
    if hasattr(rr, "aubo_pose_from_viewer"):
        xyzrpy = rr.aubo_pose_from_viewer(pos, wxyz)
    else:
        p = np.asarray(pos, dtype=float).reshape(-1)[:3]
        xyzrpy = [float(p[0]), float(p[1]), float(p[2]), 0.0, 0.0, 0.0]
    return [float(v) for v in xyzrpy[:6]]


def solved_pose_arrays():
    """Waypoint + sample poses from fully solved (ok) segments only."""
    out = {"x": [], "y": [], "z": [], "rx": [], "ry": [], "rz": []}
    last = None
    for seg in segments:
        if seg.get("status") != "ok":
            break
        for sm in seg.get("samples") or []:
            if sm.get("q") is None:
                continue
            pose = _sample_xyzrpy(sm)
            if pose is None:
                continue
            if last is not None and all(abs(a - b) < 1e-6 for a, b in zip(pose, last)):
                continue
            out["x"].append(pose[0])
            out["y"].append(pose[1])
            out["z"].append(pose[2])
            out["rx"].append(pose[3])
            out["ry"].append(pose[4])
            out["rz"].append(pose[5])
            last = pose
    return out


def _iter_motion_samples():
    """Ordered baked samples (and waypoint ends) with t, q, pos, waypoint flag."""
    assign_sample_times()
    out = []
    wps = wp.waypoints
    for i, seg in enumerate(segments):
        if seg.get("status") != "ok":
            break
        sms = seg.get("samples") or []
        for k, sm in enumerate(sms):
            q = sm.get("q")
            if q is None:
                continue
            is_wp = (k == 0) or (k == len(sms) - 1)
            rec = {
                "t": float(sm.get("t") or 0.0),
                "q": [float(v) for v in np.asarray(q, dtype=float).reshape(-1)[:6]],
                "pos": [float(v) for v in np.asarray(sm.get("pos", [0, 0, 0])).reshape(-1)[:3]],
                "wxyz": [float(v) for v in np.asarray(sm.get("wxyz", [1, 0, 0, 0])).reshape(-1)[:4]],
                "is_waypoint": bool(is_wp),
                "wp_index": i if k == 0 else (i + 1 if k == len(sms) - 1 else None),
            }
            same_t = out and abs(out[-1]["t"] - rec["t"]) < 1e-6
            if same_t:
                out[-1]["is_waypoint"] = bool(out[-1]["is_waypoint"] or rec["is_waypoint"])
                if rec.get("wp_index") is not None:
                    out[-1]["wp_index"] = rec["wp_index"]
                out[-1]["q"] = rec["q"]
                out[-1]["pos"] = rec["pos"]
                out[-1]["wxyz"] = rec["wxyz"]
            else:
                out.append(rec)
    if not out and wps:
        for i, w in enumerate(wps):
            q = w.get("q")
            if q is None:
                continue
            out.append({
                "t": float(w.get("t") or 0.0),
                "q": [float(v) for v in np.asarray(q).reshape(-1)[:6]],
                "pos": [float(v) for v in np.asarray(w.get("pos", [0, 0, 0])).reshape(-1)[:3]],
                "wxyz": [float(v) for v in np.asarray(w.get("wxyz", [1, 0, 0, 0])).reshape(-1)[:4]],
                "is_waypoint": True,
                "wp_index": i,
            })
    return out


def _fit_cubic_controls(p0, p3, mids):
    """Least-squares cubic Bézier handles so the curve passes near mids [(u,xyz),...]."""
    p0 = np.asarray(p0, dtype=float).reshape(3)
    p3 = np.asarray(p3, dtype=float).reshape(3)
    if not mids:
        d = p3 - p0
        return p0 + d / 3.0, p0 + 2.0 * d / 3.0
    A = []
    B = []
    for u, p in mids:
        u = min(0.999, max(0.001, float(u)))
        p = np.asarray(p, dtype=float).reshape(3)
        a1 = 3.0 * (1 - u) ** 2 * u
        a2 = 3.0 * (1 - u) * u ** 2
        rhs = p - ((1 - u) ** 3 * p0 + u ** 3 * p3)
        A.append([a1, a2])
        B.append(rhs)
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    try:
        x, _, _, _ = np.linalg.lstsq(A, B, rcond=None)
        c1, c2 = x[0], x[1]
    except Exception:
        d = p3 - p0
        c1, c2 = p0 + d / 3.0, p0 + 2.0 * d / 3.0
    return c1.reshape(3), c2.reshape(3)


def _as6(x):
    if x is None:
        return None
    try:
        a = [float(v) for v in np.asarray(x, dtype=float).reshape(-1)]
    except Exception:
        return None
    if len(a) < 6:
        return None
    return a[:6]


def _ik_from_pos_wxyz(pos, wxyz):
    try:
        if hasattr(rr, "aubo_pose_from_viewer") and pos is not None and wxyz is not None:
            return [round(float(v), 6) for v in rr.aubo_pose_from_viewer(pos, wxyz)]
    except Exception:
        pass
    try:
        p = [float(v) for v in np.asarray(pos, dtype=float).reshape(-1)[:3]]
        return p + [0.0, 0.0, 0.0]
    except Exception:
        return None


def _pos_wxyz_from_ik(ik):
    ik = _as6(ik)
    if ik is None:
        return None, None
    if hasattr(rr, "viewer_pose_from_aubo"):
        try:
            return rr.viewer_pose_from_aubo(ik)
        except Exception:
            pass
    pos = np.array(ik[:3], dtype=float)
    try:
        quat_xyzw = R.from_euler("xyz", ik[3:6], degrees=False).as_quat()
        wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]], dtype=float)
        return pos, wxyz
    except Exception:
        return pos, np.array([1.0, 0.0, 0.0, 0.0], dtype=float)


def _pos_wxyz_from_pose_q(q):
    q = _as6(q)
    if q is None:
        return None, None
    try:
        if hasattr(wp, "_pose_from_q"):
            return wp._pose_from_q(q)
    except Exception:
        pass
    return None, None


def write_export_json(*a, **k):
    import file_manage as fm
    return fm.write_export_json(*a, **k)


def write_save_file(*a, **k):
    import file_manage as fm
    return fm.write_save_file(*a, **k)


def apply_loaded_doc(*a, **k):
    import file_manage as fm
    return fm.apply_loaded_doc(*a, **k)


def joint_path():
    """Ordered unique joint samples from green segments, stop at first non-ok."""
    qs = []
    for seg in segments:
        if seg.get("status") != "ok":
            break
        for sm in seg.get("samples") or []:
            q = sm.get("q")
            if q is None:
                continue
            q = np.asarray(q, dtype=float).reshape(-1)[:6]
            if q.size < 6:
                continue
            if qs and np.allclose(qs[-1], q, atol=1e-5):
                continue
            qs.append(q)
    return qs


def joint_path_all():
    """Every solved sample, including after a pending hole (status / timeline)."""
    qs = []
    for seg in segments:
        if seg.get("status") != "ok":
            continue
        for sm in seg.get("samples") or []:
            q = sm.get("q")
            if q is None:
                continue
            q = np.asarray(q, dtype=float).reshape(-1)[:6]
            if q.size < 6:
                continue
            qs.append(q)
    return qs


def _sample_step_m(a, b):
    try:
        pa = np.asarray(a.get("pos"), dtype=float).reshape(-1)[:3]
        pb = np.asarray(b.get("pos"), dtype=float).reshape(-1)[:3]
        return float(np.linalg.norm(pb - pa))
    except Exception:
        return 0.0


def _eat_segment_remainders(seg):
    """Drop a short first/last interior sample so the leftover scrap is not a hop."""
    if seg.get("dwell") or seg.get("force_fk"):
        return
    sms = seg.get("samples") or []
    if len(sms) < 4:
        return
    steps = [_sample_step_m(sms[k], sms[k + 1]) for k in range(len(sms) - 1)]
    typ = float(np.median([s for s in steps if s > 1e-6]) or 0.0)
    if typ < 1e-6:
        return
    changed = False
    if steps[0] < 0.75 * typ:
        sms.pop(1)
        changed = True
        steps = [_sample_step_m(sms[k], sms[k + 1]) for k in range(len(sms) - 1)]
        typ = float(np.median([s for s in steps if s > 1e-6]) or typ)
    if len(sms) >= 4 and steps[-1] < 0.75 * typ:
        sms.pop(-2)
        changed = True
    if changed:
        n = max(1, len(sms) - 1)
        s0 = float(sms[0].get("s") or 0.0)
        s1 = float(sms[-1].get("s") or s0)
        slen = max(1e-9, s1 - s0)
        t0 = float(sms[0].get("t") or 0.0)
        t1 = float(sms[-1].get("t") or t0)
        span = max(0.0, t1 - t0)
        for k in range(1, len(sms) - 1):
            s = sms[k].get("s")
            frac = (float(s) - s0) / slen if s is not None else float(k) / float(n)
            sms[k]["t"] = t0 + span * max(0.0, min(1.0, frac))


def _seg_length_m(sms, seg=None):
    length = 0.0
    if sms and len(sms) >= 2:
        length = sum(_sample_step_m(sms[k], sms[k + 1]) for k in range(len(sms) - 1))
    if length < 1e-6 and seg is not None:
        length = float(seg.get("length") or 0.0)
    if length < 1e-6 and sms:
        s0 = float(sms[0].get("s") or 0.0)
        s1 = float(sms[-1].get("s") or s0)
        length = max(s1 - s0, 1e-6)
    return max(length, 1e-6)


def _match_end_hop_times(seg, t0, t1):
    """First/last interior get dt = Δs / v_span. 1e-4 is only a last-resort epsilon."""
    if seg.get("dwell") or seg.get("force_fk"):
        return
    sms = seg.get("samples") or []
    if len(sms) < 3:
        return
    t0 = float(t0)
    t1 = float(t1)
    span = max(1e-9, t1 - t0)
    v = _seg_length_m(sms, seg) / span
    if v < 1e-9:
        return
    d0 = _sample_step_m(sms[0], sms[1])
    d1 = _sample_step_m(sms[-2], sms[-1])
    t_first = t0 + max(d0 / v, 1e-3 if d0 > 1e-4 else 0.0)
    t_last = t1 - max(d1 / v, 1e-3 if d1 > 1e-4 else 0.0)
    if t_last <= t_first:
        mid = 0.5 * (t0 + t1)
        t_first = min(t_first, mid)
        t_last = max(t_last, mid)
    if len(sms) == 3:
        sms[1]["t"] = 0.5 * (t_first + t_last)
        return
    hi = float(sms[2].get("t") or t1)
    sms[1]["t"] = min(t_first, hi) if hi > t0 else t_first
    lo = float(sms[-3].get("t") or t0)
    sms[-2]["t"] = max(t_last, lo) if lo < t1 else t_last


def _arc_fracs(samples):
    """Distance-along-path fraction (0..1) for each sample, or None.

    Ease / custom profiles map progress to time; progress must be distance,
    not the Bezier parameter u, or a span can be fast mid-curve.
    """
    try:
        pts = [np.asarray(sm.get("pos"), dtype=float).reshape(-1)[:3] for sm in samples]
    except Exception:
        return None
    if len(pts) < 2 or any(p.shape != (3,) for p in pts):
        return None
    cum = [0.0]
    for a, b in zip(pts, pts[1:]):
        cum.append(cum[-1] + float(np.linalg.norm(b - a)))
    total = cum[-1]
    if total < 1e-6:
        return None
    return [c / total for c in cum]


def _apply_seg_time_profile(seg, t0, t1):
    samples = seg.get("samples") or []
    if len(samples) < 2:
        return
    t0 = float(t0)
    t1 = float(t1)
    span = max(0.0, t1 - t0)
    n = max(1, len(samples) - 1)
    raw_mode = seg.get("transition")
    mode = str(raw_mode or "linear")
    samples[0]["t"] = t0
    samples[-1]["t"] = t1
    for sm in samples:
        sm.pop("t_lock", None)
    if seg.get("dwell") or n < 2:
        return
    arc = _arc_fracs(samples)
    if mode == "custom":
        rec = timeline.get_custom_curve(int(seg.get("i0") or 0))
        c1, c2 = rec["c1"], rec["c2"]
        for k, sm in enumerate(samples):
            if k == 0 or k == len(samples) - 1:
                continue
            u = arc[k] if arc else sm.get("u")
            if u is None:
                u = float(k) / float(n)
            s = timeline._s_for_progress(c1, c2, float(u))
            tau = timeline._bez(0.0, c1[0], c2[0], 1.0, s)
            sm["t"] = t0 + max(0.0, min(1.0, tau)) * span
    elif mode in ("linear", "uniform") or raw_mode in (None, "", "uniform"):
        for k in range(1, len(samples) - 1):
            samples[k]["t"] = t0 + span * (float(k) / float(n))
    else:
        for k, sm in enumerate(samples):
            if k == 0 or k == len(samples) - 1:
                continue
            u = arc[k] if arc else sm.get("u")
            if u is None:
                u = float(k) / float(n)
            sm["t"] = t0 + span * timeline._ease_time_frac(mode, float(u))


def assign_sample_times():
    """Pin span endpoints to key times. Fill missing interior times once.

    Do not rewrite existing interior times — a transition or user drag owns those
    until the next explicit apply_transition / bake rebuild.
    """
    wps = wp.waypoints
    for i, seg in enumerate(segments):
        if i + 1 >= len(wps):
            break
        t0 = float(wps[i].get("t") or 0.0)
        t1 = float(wps[i + 1].get("t") or t0)
        samples = seg.get("samples") or []
        if len(samples) < 2:
            continue
        missing = [
            k for k in range(1, len(samples) - 1) if samples[k].get("t") is None
        ]
        if missing and len(missing) == len(samples) - 2:
            # A span with no frames yet takes its transition profile.
            _apply_seg_time_profile(seg, t0, t1)
        else:
            samples[0]["t"] = t0
            samples[-1]["t"] = t1
            if missing:
                # Keep the frames that exist; space only the new ones.
                _fill_missing_times(samples)


def path_distance():
    total = 0.0
    last = None
    for seg in segments:
        for sm in seg.get("samples") or []:
            p = sm.get("pos")
            if p is None:
                continue
            p = np.asarray(p, dtype=float).reshape(3)
            if last is not None:
                total += float(np.linalg.norm(p - last))
            last = p
    return total


def joint_angle_deltas(q0, q1):
    a = np.asarray(q0, dtype=float).reshape(-1)[:6]
    b = np.asarray(q1, dtype=float).reshape(-1)[:6]
    out = []
    for i in range(min(6, len(a), len(b))):
        d = float(b[i] - a[i])
        d = (d + math.pi) % (2.0 * math.pi) - math.pi
        out.append(abs(d))
    return out


ACC_WINDOW_S = 0.12
# Per-hop speed check margins (fraction of the joint limit).
HOP_VEL_SLOP = 1.10
HOP_ACC_SLOP = 1.50


def _unwrap_q_series(qs):
    out = []
    prev = None
    for q in qs:
        q = np.asarray(q, dtype=float).reshape(-1)[:6].copy()
        if prev is not None:
            d = q - prev
            d = ((d + np.pi) % (2.0 * np.pi)) - np.pi
            q = prev + d
        out.append(q)
        prev = q
    return out


def _joint_kin_at(pts, t_query, half=ACC_WINDOW_S):
    """ω, α at t_query from a local quadratic q(t)=a+bτ+cτ².

    Window is a fixed duration so dense samples do not inflate α.
    Two-point spans fall back to constant-ω (α=0) plus a from-rest
    triangle only when the query sits on the path start.
    """
    if not pts:
        return [0.0] * 6, [0.0] * 6
    ts = np.array([float(p["t"]) for p in pts], dtype=float)
    qs = _unwrap_q_series([p["q"] for p in pts])
    t_query = float(t_query)
    half = max(0.04, float(half))
    idx = np.where(np.abs(ts - t_query) <= half)[0]
    if len(idx) < 3:
        lo = int(np.searchsorted(ts, t_query - half))
        hi = int(np.searchsorted(ts, t_query + half))
        lo = max(0, lo - 1)
        hi = min(len(ts), hi + 2)
        idx = np.arange(lo, hi)
    if len(idx) < 2:
        idx = np.arange(len(ts))
    tsel = ts[idx]
    span = float(tsel.max() - tsel.min()) if len(tsel) else 0.0
    speeds = [0.0] * 6
    accs = [0.0] * 6
    if len(idx) < 3 or span < 0.04:
        if len(idx) >= 2:
            dt = max(1e-6, float(ts[idx[-1]] - ts[idx[0]]))
            dq = qs[idx[-1]] - qs[idx[0]]
            speeds = [float(abs(v) / dt) for v in dq]
            t0 = float(ts[0])
            if abs(t_query - t0) <= half and dt >= 0.04:
                accs = [float(2.0 * abs(v) / (dt * dt)) for v in dq]
        return speeds, accs
    tau = tsel - t_query
    A = np.column_stack([np.ones(len(tau)), tau, tau * tau])
    for j in range(6):
        y = np.array([qs[i][j] for i in idx], dtype=float)
        try:
            coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        except Exception:
            continue
        b, c = float(coef[1]), float(coef[2])
        speeds[j] = abs(b)
        accs[j] = abs(2.0 * c)
    return speeds, accs


def _joint_wmax():
    order = [
        "shoulder_joint", "upperArm_joint", "foreArm_joint",
        "wrist1_joint", "wrist2_joint", "wrist3_joint",
    ]
    vel = getattr(rr, "AUBO_I5_MAX_VEL_DEG", {})
    acc = getattr(rr, "AUBO_I5_MAX_ACCEL_DEG", {})
    w = [math.radians(float(vel.get(n, 180.0))) for n in order]
    a = [math.radians(float(acc.get(n, 400.0))) for n in order]
    return w, a


def min_duration_for_deltas(deltas):
    w, a = _joint_wmax()
    t_max = 0.0
    for i, dq in enumerate(deltas):
        dq = abs(float(dq))
        if dq < 1e-9:
            continue
        wi = w[i] if i < len(w) else w[-1]
        ai = a[i] if i < len(a) else a[-1]
        t_vel = dq / max(wi, 1e-6)
        # bang-bang / triangle or trapezoid
        dv = wi * wi / max(ai, 1e-6)
        if dq <= dv:
            t_acc = 2.0 * math.sqrt(dq / max(ai, 1e-6))
        else:
            t_acc = 2.0 * (wi / max(ai, 1e-6)) + (dq - dv) / max(wi, 1e-6)
        t_max = max(t_max, t_vel, t_acc)
    return t_max


_JOINT_NAMES = (
    "J1_shoulder", "J2_upperArm", "J3_foreArm",
    "J4_wrist1", "J5_wrist2", "J6_wrist3",
)


def _lin_dist(a, b):
    try:
        pa = np.asarray(a.get("pos"), dtype=float).reshape(-1)[:3]
        pb = np.asarray(b.get("pos"), dtype=float).reshape(-1)[:3]
        return float(np.linalg.norm(pb - pa))
    except Exception:
        return 0.0


def _hop_metrics(a, b, wmax, amax, prev_speeds=None, accs=None, kin_speeds=None):
    t0 = float(a.get("t") or 0.0)
    t1 = float(b.get("t") or t0)
    raw_dt = t1 - t0
    dt = raw_dt if raw_dt > 1e-6 else 1e-6
    deltas = joint_angle_deltas(a["q"], b["q"])
    if max(deltas) < 1e-5:
        return {
            "t0": t0, "t1": t1, "dt": float(max(0.0, raw_dt)),
            "lin_m": _lin_dist(a, b),
            "speed": 0.0, "limit": float(wmax[0]),
            "acc": 0.0, "acc_limit": float(amax[0]),
            "avail": float(max(0.0, raw_dt)), "need": 0.0,
            "over_vel": False, "over_acc": False, "over_time": False,
            "over": False, "status": "ok", "dwell": True,
            "deltas": deltas, "speeds": [0.0] * 6, "accs": [0.0] * 6,
            "fault_vel": [], "fault_acc": [],
        }
    speeds = [abs(d) / dt for d in deltas]
    if kin_speeds is not None and len(kin_speeds) == 6:
        speeds = [max(speeds[j], float(kin_speeds[j])) for j in range(6)]
    if accs is not None and len(accs) == 6:
        accs = [float(v) for v in accs]
    elif prev_speeds is not None and len(prev_speeds) == len(speeds):
        accs = [abs(speeds[j] - prev_speeds[j]) / max(dt, ACC_WINDOW_S) for j in range(len(speeds))]
    else:
        accs = [abs(d) * 2.0 / (max(dt, ACC_WINDOW_S) ** 2) for d in deltas]
    worst = int(max(
        range(len(speeds)),
        key=lambda j: speeds[j] / max(wmax[j] if j < len(wmax) else wmax[-1], 1e-6),
    )) if speeds else 0
    speed = speeds[worst] if speeds else 0.0
    acc_worst = int(max(
        range(len(accs)),
        key=lambda j: accs[j] / max(amax[j] if j < len(amax) else amax[-1], 1e-6),
    )) if accs else 0
    acc = accs[acc_worst] if accs else 0.0
    vlim = wmax[worst] if worst < len(wmax) else wmax[-1]
    alim = amax[acc_worst] if acc_worst < len(amax) else amax[-1]
    need = min_duration_for_deltas(deltas)
    over_vel = any(speeds[j] > (wmax[j] if j < len(wmax) else wmax[-1]) + 1e-9 for j in range(len(speeds)))
    over_acc = any(accs[j] > (amax[j] if j < len(amax) else amax[-1]) + 1e-9 for j in range(len(accs)))
    # Per-hop from-rest min-duration is only meaningful on a long hop.
    # Sub-samples of the same span would all fail 2*sqrt(dq/a) even when legal.
    over_time = raw_dt + 1e-6 < need and raw_dt >= 0.2
    fault_vel = [j for j in range(len(speeds)) if speeds[j] > (wmax[j] if j < len(wmax) else wmax[-1]) + 1e-9]
    fault_acc = [j for j in range(len(accs)) if accs[j] > (amax[j] if j < len(amax) else amax[-1]) + 1e-9]
    over = over_vel or over_acc or over_time
    return {
        "t0": t0,
        "t1": t1,
        "dt": float(max(0.0, raw_dt)),
        "lin_m": _lin_dist(a, b),
        "speed": float(speed),
        "limit": float(vlim),
        "acc": float(acc),
        "acc_limit": float(alim),
        "avail": float(max(0.0, raw_dt)),
        "need": float(need),
        "over_vel": bool(over_vel),
        "over_acc": bool(over_acc),
        "over_time": bool(over_time),
        "over": bool(over),
        "status": "too_fast" if over else "ok",
        "dwell": False,
        "deltas": deltas,
        "speeds": speeds,
        "accs": accs,
        "fault_vel": fault_vel,
        "fault_acc": fault_acc,
        "worst_vel": worst,
        "worst_acc": acc_worst,
    }


def _timing_points():
    pts = []
    last_t = None
    for seg in segments:
        if seg.get("status") != "ok":
            break
        for sm in seg.get("samples") or []:
            if sm.get("q") is None or sm.get("t") is None:
                continue
            t = float(sm["t"])
            if last_t is not None and abs(t - last_t) < 1e-9:
                pts[-1] = {"t": t, "q": sm["q"], "pos": sm.get("pos")}
                continue
            pts.append({"t": t, "q": sm["q"], "pos": sm.get("pos")})
            last_t = t
    return pts


def _signed_rates(a, b):
    """Signed joint rates (rad/s) over one hop and its dt."""
    dt = float(b.get("t") or 0.0) - float(a.get("t") or 0.0)
    qa = np.asarray(a["q"], dtype=float).reshape(-1)[:6]
    qb = np.asarray(b["q"], dtype=float).reshape(-1)[:6]
    d = (qb - qa + math.pi) % (2.0 * math.pi) - math.pi
    if dt <= 1e-6:
        return None, dt
    return d / dt, dt


def _hop_accels(wmax):
    """Joint acceleration at every sample, from the rate change between the
    hops on either side: |w_k - w_(k-1)| / mean(dt). Each sample's
    acceleration is blamed on the faster of its two hops, so a fault stays on
    the hop that causes it instead of leaking onto a neighbour.

    Returns {(seg, k): [acc per joint]} keyed by the hop's start sample index.
    """
    chain = []
    for i, seg in enumerate(segments):
        if seg.get("status") != "ok":
            break
        sms = [sm for sm in (seg.get("samples") or []) if sm.get("q") is not None]
        for k, (a, b) in enumerate(zip(sms, sms[1:])):
            w, dt = _signed_rates(a, b)
            if w is None:
                continue
            chain.append({"key": (i, k), "w": w, "dt": dt})
    out = {c["key"]: [0.0] * 6 for c in chain}
    zero = np.zeros(6)

    def _blame(acc, prev, cur):
        pick = cur
        if prev is not None and cur is not None:
            rp = max(abs(prev["w"][j]) / max(wmax[j], 1e-6) for j in range(6))
            rc = max(abs(cur["w"][j]) / max(wmax[j], 1e-6) for j in range(6))
            pick = prev if rp > rc else cur
        elif cur is None:
            pick = prev
        rec = out[pick["key"]]
        for j in range(6):
            rec[j] = max(rec[j], float(acc[j]))

    for n in range(len(chain) + 1):
        prev = chain[n - 1] if n > 0 else None
        cur = chain[n] if n < len(chain) else None
        if prev is None and cur is None:
            continue
        # The arm starts and ends the path at rest.
        w0 = prev["w"] if prev is not None else zero
        w1 = cur["w"] if cur is not None else zero
        if prev is not None and cur is not None:
            span = 0.5 * (prev["dt"] + cur["dt"])
        else:
            span = (cur or prev)["dt"]
        acc = np.abs(w1 - w0) / max(span, 1e-6)
        _blame(acc, prev, cur)
    return out


def evaluate_timing(force=False):
    """Score every hop and fail a span when any hop in it is over the limits.

    A hop fails on its own joint rate (dq/dt above limit * HOP_VEL_SLOP)
    or on the windowed joint acceleration (above limit * HOP_ACC_SLOP).
    A too-fast span does not block scoring of later spans.
    """
    if not force and not _speed_valid:
        return False
    timeline.pin_key_times()
    wmax, amax = _joint_wmax()
    any_fast = False
    prev_speeds = None
    series = _timing_points()
    acc_by_hop = _hop_accels(wmax)
    for i, seg in enumerate(segments):
        if seg.get("status") != "ok":
            if seg.get("status") == "fail":
                seg["timing"] = "fail"
            elif seg.get("timing") not in ("uncalc", "too_fast"):
                seg["timing"] = None
            seg["micro"] = []
            continue
        seg["micro"] = []
        samples = [sm for sm in (seg.get("samples") or []) if sm.get("q") is not None]
        if len(samples) < 2:
            seg["timing"] = "uncalc"
            continue
        for k, (a, b) in enumerate(zip(samples, samples[1:])):
            mid = 0.5 * (float(a.get("t") or 0.0) + float(b.get("t") or 0.0))
            kin_w, _ = _joint_kin_at(series, mid)
            # Per-sample rate change, not a windowed fit: the fit smoothed a
            # short fast hop away and smeared its fault onto the hop before.
            kin_a = acc_by_hop.get((i, k), [0.0] * 6)
            rec = _hop_metrics(
                a, b, wmax, amax,
                prev_speeds=prev_speeds,
                accs=kin_a,
                kin_speeds=kin_w,
            )
            prev_speeds = rec.get("speeds")
            # Hop pass/fail: raw dq/dt per joint (what the arm actually has to
            # do between these two samples) and the windowed acceleration fit.
            # Margins keep the 0.01 s time snap and fit noise from false-flagging.
            hop_dt = max(1e-6, float(b.get("t") or 0.0) - float(a.get("t") or 0.0))
            raw_w = [d / hop_dt for d in rec.get("deltas") or []]
            fault_vel = [j for j, v in enumerate(raw_w) if v > wmax[j] * HOP_VEL_SLOP]
            fault_acc = [] if rec.get("dwell") else [
                j for j, v in enumerate(kin_a or []) if v > amax[j] * HOP_ACC_SLOP
            ]
            rec["hop_w"] = raw_w
            rec["over_vel"] = bool(fault_vel)
            rec["over_acc"] = bool(fault_acc)
            rec["over_time"] = False
            rec["fault_vel"] = fault_vel
            rec["fault_acc"] = fault_acc
            rec["over"] = bool(fault_vel or fault_acc)
            rec["status"] = "too_fast" if rec["over"] else "ok"
            seg["micro"].append(rec)
            if a is samples[0]:
                a["timing"] = "ok"
            # Orange goes on the failing hop and the frame it arrives at only.
            b["timing"] = rec["status"]
            b["speed"] = rec["speed"]
            b["accel"] = rec["acc"]
            b["limit"] = rec["limit"]
        mid = 0.5 * (float(samples[0].get("t") or 0.0) + float(samples[-1].get("t") or 0.0))
        kin_w, kin_a = _joint_kin_at(series, mid)
        span = _hop_metrics(
            samples[0], samples[-1], wmax, amax,
            accs=kin_a,
            kin_speeds=kin_w,
        )
        t_span = float(samples[-1].get("t") or 0.0) - float(samples[0].get("t") or 0.0)
        deltas = joint_angle_deltas(samples[0]["q"], samples[-1]["q"])
        lin = _lin_dist(samples[0], samples[-1])
        # Keyframe span: mean joint rate vs limit only.
        # Accel / from-rest triangle on 0.1 s hops false-flags look-at paths.
        dt_use = max(t_span, 1e-3)
        mean_w = [abs(d) / dt_use for d in deltas]
        slop = 1.10
        over_vel = any(mean_w[j] > wmax[j] * slop for j in range(len(mean_w)))
        need_vel = 0.0
        for j, d in enumerate(deltas):
            need_vel = max(need_vel, abs(d) / max(wmax[j], 1e-6))
        span["need"] = need_vel
        span["over_vel"] = bool(over_vel)
        span["over_acc"] = False
        span["over_time"] = bool(t_span + 1e-6 < need_vel)
        if lin < 0.002 and max(deltas) < 0.02:
            span["over_vel"] = False
            span["over_time"] = False
        hop_vel = any(h.get("over_vel") for h in seg["micro"])
        hop_acc = any(h.get("over_acc") for h in seg["micro"])
        span["over_vel"] = bool(span["over_vel"] or hop_vel)
        span["over_acc"] = bool(hop_acc)
        span["over"] = bool(span["over_vel"] or span["over_acc"] or span["over_time"])
        if span.get("over"):
            seg["timing"] = "too_fast"
            any_fast = True
            # A hop fault already marked its own two samples above; only a
            # failed keyframe-to-keyframe mean marks the whole span.
            if not (hop_vel or hop_acc):
                for sm in samples:
                    sm["timing"] = "too_fast"
                for hop in seg["micro"]:
                    hop["over"] = True
                    hop["status"] = "too_fast"
                    hop["over_vel"] = span.get("over_vel")
                    hop["over_time"] = span.get("over_time")
        elif _seg_force_fk(i) or seg.get("force_fk"):
            span["status"] = "fk"
            for hop in seg["micro"]:
                hop["status"] = "fk"
            for sm in samples:
                sm["timing"] = "fk"
            seg["timing"] = "fk"
        else:
            seg["timing"] = "ok"
    return any_fast


def write_overspeed_report(path=None):
    """Write every overspeed/over-accel hop to CSV + xlsx under artifacts."""
    wmax, amax = _joint_wmax()
    names = _JOINT_NAMES
    rows = []
    # Same records evaluate_timing scored, so the report agrees with the
    # span colors (an all-ok bake writes 0 rows).
    for si, seg in enumerate(segments):
        for k, rec in enumerate(seg.get("micro") or []):
            if not rec.get("over"):
                continue
            fv = rec.get("fault_vel") or []
            fa = rec.get("fault_acc") or []
            rows.append({
                "seg": si,
                "hop": k,
                "t0_s": rec["t0"],
                "t1_s": rec["t1"],
                "dt_s": rec.get("dt"),
                "lin_m": rec.get("lin_m"),
                "set_duration_s": rec.get("avail"),
                "min_duration_s": rec.get("need"),
                "dwell": bool(rec.get("dwell") or seg.get("dwell")),
                "over_vel": rec.get("over_vel"),
                "over_acc": rec.get("over_acc"),
                "over_time": rec.get("over_time"),
                "fault_vel_joints": ",".join(names[j] for j in fv),
                "fault_vel_rad_s": ",".join("%.4f" % (rec.get("hop_w") or rec["speeds"])[j] for j in fv),
                "max_vel_rad_s": ",".join("%.4f" % (wmax[j] if j < len(wmax) else wmax[-1]) for j in fv),
                "fault_acc_joints": ",".join(names[j] for j in fa),
                "fault_acc_rad_s2": ",".join("%.4f" % rec["accs"][j] for j in fa),
                "max_acc_rad_s2": ",".join("%.4f" % (amax[j] if j < len(amax) else amax[-1]) for j in fa),
                "worst_vel": names[rec.get("worst_vel", 0)],
                "worst_acc": names[rec.get("worst_acc", 0)],
                "speed_rad_s": rec.get("speed"),
                "acc_rad_s2": rec.get("acc"),
            })
    out_dir = app_paths.ensure_dir(app_paths.EXPORTS_DIR)
    csv_path = Path(path) if path else out_dir / "bake_overspeed.csv"
    xlsx_path = csv_path.with_suffix(".xlsx")
    cols = [
        "seg", "hop", "t0_s", "t1_s", "dt_s", "lin_m", "set_duration_s",
        "min_duration_s", "dwell", "over_vel", "over_acc", "over_time",
        "fault_vel_joints", "fault_vel_rad_s", "max_vel_rad_s",
        "fault_acc_joints", "fault_acc_rad_s2", "max_acc_rad_s2",
        "worst_vel", "worst_acc", "speed_rad_s", "acc_rad_s2",
    ]
    try:
        import pandas as pd
        df = pd.DataFrame(rows, columns=cols)
        df.to_csv(csv_path, index=False)
        df.to_excel(xlsx_path, index=False)
    except Exception:
        import csv
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for r in rows:
                w.writerow(r)
        xlsx_path = None
    print("overspeed report", csv_path, "rows", len(rows))
    return {"ok": True, "csv": str(csv_path), "xlsx": str(xlsx_path) if xlsx_path else None, "rows": len(rows)}


def export_overspeed():
    """/api/speed_export: rewrite the report from the current scored bake."""
    return write_overspeed_report()


def _first_overspeed_t():
    hops = list(baked_hops) if baked_hops else []
    if not hops:
        for s in segments:
            hops.extend(s.get("micro") or [])
    for hop in hops:
        if hop.get("over") or hop.get("status") == "too_fast":
            return float(hop.get("t1") or hop.get("t0") or 0.0)
    return None


def store_bake_motion():
    """Freeze hop speeds at bake time for timeline, export, and robot run."""
    global baked_hops, baked_points, _tl_cache
    baked_hops = timeline._spans_from_timed()
    pts = timed_joint_samples()
    baked_points = []
    for i, p in enumerate(pts):
        hop = baked_hops[i - 1] if i and (i - 1) < len(baked_hops) else None
        baked_points.append({
            "t": float(p["t"]),
            "q": list(p["q"]),
            "speed": float(hop["speed"]) if hop else 0.0,
            "acc": float(hop["acc"]) if hop else float(DEFAULT_ACC),
            "limit": None if hop is None else float(hop["limit"]),
            "acc_limit": None if hop is None else float(hop.get("acc_limit") or 0.0),
            "over": bool(hop["over"]) if hop else False,
        })
    _tl_cache = None
    print("bake motion stored", "points", len(baked_points), "hops", len(baked_hops))
    return baked_hops


def timed_joint_samples():
    assign_sample_times()
    out = []
    last_t = None
    for seg in segments:
        if seg.get("status") != "ok":
            break
        for sm in seg.get("samples") or []:
            q = sm.get("q")
            if q is None:
                continue
            t = float(sm.get("t") or 0.0)
            q = [float(v) for v in np.asarray(q, dtype=float).reshape(-1)[:6]]
            rec = {
                "t": t,
                "q": q,
                "speed": sm.get("speed"),
                "acc": sm.get("accel") or sm.get("acc") or DEFAULT_ACC,
                "limit": sm.get("limit"),
                "over": bool(sm.get("timing") == "too_fast"),
            }
            if last_t is not None and abs(t - last_t) < 1e-6:
                out[-1] = rec
            else:
                out.append(rec)
            last_t = t
    return out


def _interior_wp_times():
    wps = wp.waypoints or []
    if len(wps) < 3:
        return []
    return [float(w.get("t") or 0.0) for w in wps[1:-1]]


def _is_interior_wp_t(t, tol=0.02):
    t = float(t or 0.0)
    return any(abs(t - wt) <= tol for wt in _interior_wp_times())


def _joint_reversal(qa, qb, qc):
    vin = np.asarray(qb, dtype=float).reshape(-1)[:6] - np.asarray(qa, dtype=float).reshape(-1)[:6]
    vout = np.asarray(qc, dtype=float).reshape(-1)[:6] - np.asarray(qb, dtype=float).reshape(-1)[:6]
    if float(np.max(np.abs(vin) + np.abs(vout))) < 0.04:
        return False
    return bool(np.any((vin * vout) < -0.002))


def _dense_polyline():
    return len(wp.waypoints or []) >= 16


def diamond_timed_joints():
    """One timed q per diamond (segment endpoints). Raw SDK-style knots."""
    out = []
    last_t = None
    for seg in segments:
        if seg.get("status") != "ok":
            break
        sms = [sm for sm in (seg.get("samples") or []) if sm.get("q") is not None and sm.get("t") is not None]
        if not sms:
            continue
        ends = [sms[0]] if len(sms) == 1 else [sms[0], sms[-1]]
        for sm in ends:
            t = float(sm["t"])
            q = [float(v) for v in np.asarray(sm["q"], dtype=float).reshape(-1)[:6]]
            if last_t is not None and abs(t - last_t) < 1e-6:
                out[-1] = {"t": t, "q": q}
            else:
                out.append({"t": t, "q": q})
            last_t = t
    return out


def _fill_joint_times(recs, dt=0.02):
    """Linear joint fill so type-2 cubic cannot cut a corner between sparse knots."""
    if len(recs) < 2:
        return recs
    out = [{"t": float(recs[0]["t"]), "q": list(recs[0]["q"])[:6]}]
    prev = out[0]["q"]
    for rec in recs[1:]:
        t1 = float(rec["t"])
        q1 = _unwrap_toward(rec["q"], prev)
        t0 = out[-1]["t"]
        q0 = out[-1]["q"]
        span = t1 - t0
        if span > dt * 1.5:
            n = max(1, int(round(span / dt)))
            for k in range(1, n):
                u = k / float(n)
                out.append({
                    "t": t0 + span * u,
                    "q": [(1.0 - u) * q0[j] + u * q1[j] for j in range(6)],
                })
        out.append({"t": t1, "q": list(q1)})
        prev = q1
    return out


def execution_joint_samples():
    """Sim/real buffer: baked samples only. No extra time-fill."""
    recs = []
    try:
        recs = [r for r in (_iter_motion_samples() or []) if r.get("q") is not None]
    except Exception:
        recs = timed_joint_samples()
    if len(recs) < 2:
        recs = timed_joint_samples()
    out = []
    for r in recs:
        t = float(r["t"])
        q = [float(v) for v in list(r["q"])[:6]]
        if out and abs(t - out[-1]["t"]) < 0.002:
            out[-1] = {"t": t, "q": q}
            continue
        out.append({"t": t, "q": q})
    return out


def _blend_interior_knots(samples):
    if len(samples) < 3:
        return samples
    out = []
    n = len(samples)
    for i, s in enumerate(samples):
        if 0 < i < n - 1 and _is_interior_wp_t(s.get("t"), tol=0.02):
            if _joint_reversal(samples[i - 1]["q"], s["q"], samples[i + 1]["q"]):
                continue
        out.append({"t": float(s["t"]), "q": list(s["q"])[:6]})
    return out


def _pad_holds(samples, dt=0.05):
    """Repeat identical q across a dwell so type-2 cubic cannot carry velocity."""
    if not samples or len(samples) < 2:
        return samples
    out = [{"t": float(samples[0]["t"]), "q": list(samples[0]["q"])[:6]}]
    for rec in samples[1:]:
        t1 = float(rec["t"])
        q1 = list(rec["q"])[:6]
        t0 = out[-1]["t"]
        q0 = out[-1]["q"]
        span = t1 - t0
        same = False
        try:
            same = float(np.max(np.abs(np.asarray(q0, dtype=float) - np.asarray(q1, dtype=float)))) < 1e-5
        except Exception:
            same = False
        if same and span > dt * 1.5:
            n = max(2, int(round(span / dt)))
            for k in range(1, n):
                out.append({"t": t0 + span * k / float(n), "q": list(q0)})
        out.append({"t": t1, "q": list(q1)})
    return out


def padded_timed_joints(samples=None):
    """Joint samples from t=0. Diamonds stay in the list."""
    if not samples:
        samples = execution_joint_samples()
    if not samples:
        return []
    q0 = list(samples[0]["q"])
    t0 = float(samples[0]["t"])
    out = []
    if t0 > 1e-6:
        out.append({"t": 0.0, "q": q0})
        out.append({"t": t0, "q": list(q0)})
        for sm in samples:
            if abs(float(sm["t"]) - t0) < 1e-9:
                continue
            out.append({"t": float(sm["t"]), "q": list(sm["q"])})
    else:
        out = [{"t": float(s["t"]), "q": list(s["q"])} for s in samples]
    return _pad_holds(out)


def sample_interval(samples=None):
    samples = samples if samples is not None else padded_timed_joints()
    dts = []
    for a, b in zip(samples, samples[1:]):
        dt = float(b["t"]) - float(a["t"])
        if dt > 1e-6:
            dts.append(dt)
    if not dts:
        return 0.01
    dts.sort()
    return max(0.005, float(dts[len(dts) // 2]))


def is_ik_playable():
    """IK-solved prefix long enough to animate in sim (timing may be yellow/orange)."""
    if not segments or len(wp.waypoints) < 2:
        return False
    n = 0
    for seg in segments:
        if seg.get("status") != "ok":
            break
        for sm in seg.get("samples") or []:
            if sm.get("q") is not None:
                n += 1
    return n >= 2


def is_runnable():
    """Real Run needs a finished bake: IK plus every span scored, including Force FK."""
    if not segments:
        return False
    if not is_ik_playable():
        return False
    if any(s.get("timing") == "too_fast" for s in segments):
        return False
    if any(s.get("status") != "ok" or s.get("timing") not in ("ok", "fk") for s in segments):
        return False
    return True


def status_dict():
    qs = joint_path_all()
    ik_pts = 0
    ik_total = 0
    speed_ok = 0
    speed_fail = 0
    speed_pending = 0
    for seg in segments:
        sms = list(seg.get("samples") or [])
        ik_total += len(sms)
        ik_pts += sum(1 for sm in sms if sm.get("q") is not None)
        hops = list(seg.get("micro") or [])
        if not hops and len(sms) >= 2:
            hops = sms[1:]
        timing = seg.get("timing")
        st = seg.get("status")
        n_h = max(0, len(hops))
        if st != "ok":
            speed_pending += n_h
        elif timing in ("ok", "fk"):
            speed_ok += n_h
        elif timing == "too_fast":
            speed_fail += n_h
        else:
            speed_pending += n_h
    hop_n = len(baked_hops or [])
    return {
        "spacing": spacing,
        "segments": len(segments),
        "ok": sum(1 for s in segments if s["status"] == "ok"),
        "fail": sum(1 for s in segments if s["status"] == "fail"),
        "pending": sum(1 for s in segments if s["status"] == "pending"),
        "samples": len(qs),
        "ik_pts": ik_pts,
        "ik_total": ik_total,
        "speed_ok": speed_ok,
        "speed_fail": speed_fail,
        "speed_pending": speed_pending,
        "speed_total": speed_ok + speed_fail + speed_pending,
        "hops": hop_n,
        "speed_valid": bool(_speed_valid),
        "baking": bool(globals().get("_baking")),
        "runnable": is_runnable(),
        "ik_ok": bool(segments) and all(s.get("status") == "ok" for s in segments),
        "path_distance": path_distance(),
        "show_viewer_samples": bool(show_viewer_samples),
        "show_tl_samples": bool(show_tl_samples),
    }


def _json_sample(sm):
    def arr(x, n):
        if x is None:
            return None
        return [float(v) for v in np.asarray(x, dtype=float).reshape(-1)[:n]]
    return {
        "t": None if sm.get("t") is None else float(sm["t"]),
        "u": sm.get("u"),
        "s": sm.get("s"),
        "pos": arr(sm.get("pos"), 3),
        "wxyz": arr(sm.get("wxyz"), 4),
        "q": arr(sm.get("q"), 6),
        "timing": sm.get("timing"),
        "speed": sm.get("speed"),
        "accel": sm.get("accel"),
        "limit": sm.get("limit"),
        "t_lock": sm.get("t_lock"),
    }


def export_bake_state():
    segs = []
    for seg in segments:
        segs.append({
            "status": seg.get("status"),
            "timing": seg.get("timing"),
            "fail_index": seg.get("fail_index"),
            "samples": [_json_sample(sm) for sm in (seg.get("samples") or [])],
            "micro": list(seg.get("micro") or []),
        })
    return {
        "spacing": float(spacing),
        "speed_valid": bool(_speed_valid),
        "segments": segs,
        "hops": list(baked_hops),
        "points": list(baked_points),
    }


def import_bake_state(data):
    global segments, baked_hops, baked_points, _speed_valid, spacing
    if not data:
        return False
    if data.get("spacing") is not None:
        try:
            spacing = float(data["spacing"])
        except Exception:
            pass
    _sync_segment_count()
    raw = data.get("segments") or []
    for i, rec in enumerate(raw):
        if i >= len(segments):
            break
        samples = []
        for sm in rec.get("samples") or []:
            item = dict(sm)
            if sm.get("pos") is not None:
                item["pos"] = np.asarray(sm["pos"], dtype=float).reshape(-1)[:3]
            if sm.get("wxyz") is not None:
                item["wxyz"] = np.asarray(sm["wxyz"], dtype=float).reshape(-1)[:4]
            if sm.get("q") is not None:
                item["q"] = np.asarray(sm["q"], dtype=float).reshape(-1)[:6]
            samples.append(item)
        segments[i]["status"] = rec.get("status") or "pending"
        segments[i]["timing"] = rec.get("timing")
        segments[i]["fail_index"] = rec.get("fail_index")
        segments[i]["samples"] = samples
        segments[i]["micro"] = list(rec.get("micro") or [])
    baked_hops = list(data.get("hops") or [])
    baked_points = list(data.get("points") or [])
    _speed_valid = bool(data.get("speed_valid"))
    if any((s.get("samples") or []) for s in segments):
        try:
            evaluate_timing(force=True)
        except Exception as e:
            print("import timing failed", e)
    _draw_sample_dots()
    if hasattr(wp, "draw_path"):
        wp.draw_path()
    return True


# Timeline control (diamond drag, sample times, transitions, clipboard) lives
# in timeline.py; importing it exposes those functions here as bake.<name>.
import timeline  # noqa: E402,F401
