"""Timeline control: keyframe (diamond) dragging and snapping, sample-time
edits, span transitions / custom curves, timeline payload for the UI, and
keyframe copy / cut / paste.

Bake state (segments, baked samples, selected_sample, clipboard, custom
curves) stays in path_bake.py and editor state stays in waypoints.py; this
module reads and writes them as bake.<name> / wp.<name>. Functions moved here
are re-exposed on their original module (end of file), so callers such as
bake.pin_key_times() and wp.set_keyframe_times(...) keep working unchanged.
"""
import math
import numpy as np

import path_bake as bake
import waypoints as wp


# ---- keyframe (diamond) time edits: moved from waypoints.py ---------------

def _segment_length_m(i):
    if i < 0 or i + 1 >= len(wp.waypoints):
        return 0.0
    try:
        import path_bake as bake
        segs = getattr(bake, "segments", None) or []
        if i < len(segs):
            L = float(segs[i].get("length") or 0.0)
            if L > 1e-9:
                return L
    except Exception:
        pass
    try:
        poly = wp._segment_polyline(i)
        pts = np.asarray(poly, dtype=float).reshape(-1, 3)
        if len(pts) < 2:
            raise ValueError("short")
        return float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))
    except Exception:
        a = np.asarray(wp.waypoints[i]["pos"], dtype=float).reshape(3)
        b = np.asarray(wp.waypoints[i + 1]["pos"], dtype=float).reshape(3)
        return float(np.linalg.norm(b - a))


def apply_linear_speed(indices, speed_mps):
    try:
        speed = float(speed_mps)
    except Exception:
        return {"ok": False, "error": "bad speed"}
    if speed <= 1e-9:
        return {"ok": False, "error": "speed must be > 0"}
    n = len(wp.waypoints)
    if n < 2:
        return {"ok": False, "error": "need two keyframes"}
    raw = [int(i) for i in (indices or [])]
    if not raw and wp.selected_idx is not None:
        raw = [int(wp.selected_idx)]
    idxs = []
    for i in raw:
        if 0 <= i < n - 1:
            idxs.append(i)
        elif i == n - 1:
            idxs.append(n - 2)
    idxs = sorted(set(idxs))
    if not idxs:
        return {"ok": False, "error": "select a diamond that starts a span"}
    times = [float(w.get("t") or 0.0) for w in wp.waypoints]
    for i in idxs:
        dist = _segment_length_m(i)
        dt = max(0.01, float(dist) / speed)
        new_end = times[i] + dt
        shift = new_end - times[i + 1]
        wp.waypoints[i]["lin_speed"] = float(speed)
        print("linear_speed span", i, "dist_m", dist, "dt", dt, "shift", shift)
        for k in range(i + 1, n):
            times[k] = max(0.0, times[k] + shift)
    times = [wp._snap_t(t) for t in times]
    set_keyframe_times([(k, times[k]) for k in range(n)], released=True, stretch=True)
    for k in range(n):
        wp.waypoints[k]["t"] = times[k]
    wp._bake_call("bump_timeline_cache")
    live = [None if w.get("t") is None else float(w["t"]) for w in wp.waypoints]
    print("linear_speed done", live)
    return {"ok": True, "speed": speed, "indices": idxs, "times": live}


def _first_compressed_span(old_times, new_times):
    n = min(len(old_times or []), len(new_times or []))
    for i in range(max(0, n - 1)):
        a0, a1 = old_times[i], old_times[i + 1]
        b0, b1 = new_times[i], new_times[i + 1]
        if a0 is None or a1 is None or b0 is None or b1 is None:
            continue
        if abs(float(b1) - float(b0)) + 1e-6 < abs(float(a1) - float(a0)):
            return i
    return None


def set_keyframe_times(moves, released=False, stretch=False):
    """Apply one or more (index, t) updates. On release, allow reorder by t,
    rebuild path from the first changed neighbor, and drop bake from there on
    (white / unbaked) including every later keyframe."""
    if not moves:
        return []
    cleaned = []
    for idx, t in moves:
        try:
            idx = int(idx)
            t = wp._snap_t(t)
        except Exception:
            continue
        if 0 <= idx < len(wp.waypoints):
            cleaned.append((idx, t))
    if not cleaned:
        return []
    old_times = [None if w.get("t") is None else float(w.get("t")) for w in wp.waypoints]
    if not released:
        if not wp._suspend_history and not wp._drag_checkpointed:
            wp._push_history()
            wp._drag_checkpointed = True
        for idx, t in cleaned:
            wp.waypoints[idx]["t"] = t
        wp._bake_call("pin_key_times")
        new_times = [None if w.get("t") is None else float(w.get("t")) for w in wp.waypoints]
        comp = _first_compressed_span(old_times, new_times)
        if comp is not None:
            wp._bake_call("invalidate_timing_span", comp)
        wp._bake_call("bump_timeline_cache")
        return [{"index": i, "t": t} for i, t in cleaned]

    if not wp._suspend_history and not wp._drag_checkpointed:
        wp._push_history()
    wp._drag_checkpointed = False
    wp._quit_sim()

    old_index_by_id = {id(w): i for i, w in enumerate(wp.waypoints)}
    moved_items = []
    moved_ids = set()
    any_change = False
    old_times = [None if w.get("t") is None else float(w.get("t")) for w in wp.waypoints]
    for idx, t in cleaned:
        item = wp.waypoints[idx]
        old_t = item.get("t")
        try:
            if old_t is None or abs(float(old_t) - float(t)) > 1e-6:
                any_change = True
        except Exception:
            any_change = True
        item["t"] = t
        moved_items.append(item)
        moved_ids.add(id(item))
    if not any_change:
        out = []
        for item in moved_items:
            j = wp._index_of(item)
            if j is not None:
                out.append({"index": j, "t": float(item.get("t") or 0.0)})
        return out

    reordered = wp.sort_waypoints_by_time()
    dirty_wp = wp._dirty_start_from_moves(old_index_by_id, moved_ids)
    if reordered:
        wp._rebind_after_reorder()
        n = len(wp.waypoints)
        for i in range(max(0, dirty_wp - 1), max(0, n - 1)):
            try:
                wp.place_segment_ctrls(i, i + 1, force=False)
            except Exception:
                pass

    if moved_items:
        sel = wp._index_of(moved_items[0])
        if sel is not None:
            wp.selected_idx = sel

    dirty_seg = max(0, dirty_wp - 1) if dirty_wp > 0 else 0
    wp.draw_path()
    new_times = [None if w.get("t") is None else float(w.get("t")) for w in wp.waypoints]
    comp = _first_compressed_span(old_times, new_times)
    if reordered:
        wp._bake_call("invalidate_from", dirty_seg)
    elif stretch:
        wp._bake_call("pin_key_times")
        wp._bake_call("invalidate_timing_from", dirty_seg)
    else:
        wp._bake_call("pin_key_times")
        if comp is not None:
            wp._bake_call("invalidate_timing_span", comp)
    wp._bake_call("bump_timeline_cache")
    out = []
    for item in moved_items:
        j = wp._index_of(item)
        if j is not None:
            out.append({"index": j, "t": float(item.get("t") or 0.0)})
    return out


def set_keyframe_time(idx, t, released=False):
    rows = set_keyframe_times([(idx, t)], released=released)
    if not rows:
        return None
    return float(rows[0]["t"])


# ---- sample timing, transitions, timeline payload, clipboard: moved from path_bake.py

TRANSITIONS = {
    "linear": lambda u: u,
    "ease_in": lambda u: u * u,
    "ease_out": lambda u: 1.0 - (1.0 - u) * (1.0 - u),
    "ease_in_out": lambda u: u * u * (3.0 - 2.0 * u),
}


def select_sample(seg, k):
    bake.selected_sample = (int(seg), int(k))
    if hasattr(wp, "deselect_all"):
        wp.deselect_all()
    elif hasattr(wp, "selected_idx"):
        wp.selected_idx = None


def get_selected_sample():
    if bake.selected_sample is None:
        return None
    seg_i, k = bake.selected_sample
    if seg_i < 0 or seg_i >= len(bake.segments):
        return None
    samples = bake.segments[seg_i].get("samples") or []
    if k < 0 or k >= len(samples):
        return None
    return seg_i, k, samples[k]


def invalidate_timing_span(seg_i):
    """Drop speed scores for one span. Keep IK and other spans."""
    bake._speed_valid = False
    bake._tl_cache = None
    if seg_i < 0 or seg_i >= len(bake.segments):
        return
    seg = bake.segments[seg_i]
    if seg.get("status") == "ok":
        seg["timing"] = "uncalc"
        seg["micro"] = []
    for sm in seg.get("samples") or []:
        if sm.get("q") is not None:
            sm["timing"] = "uncalc"
            sm["speed"] = None
            sm["accel"] = None
            sm["limit"] = None
    t0 = t1 = None
    try:
        t0 = float(wp.waypoints[seg_i].get("t") or 0.0)
        t1 = float(wp.waypoints[seg_i + 1].get("t") or t0)
    except Exception:
        t0 = t1 = None
    if t0 is not None and t1 is not None:
        lo, hi = (t0, t1) if t0 <= t1 else (t1, t0)
        bake.baked_hops = [
            h for h in bake.baked_hops
            if not (float(h.get("t0") or 0.0) >= lo - 1e-6 and float(h.get("t1") or 0.0) <= hi + 1e-6)
        ]


def invalidate_timing_from(seg_i):
    """Drop speed from seg_i onward. Keep earlier hop scores and all IK."""
    bake._speed_valid = False
    t_cut = None
    if 0 <= seg_i < len(wp.waypoints):
        t_cut = float(wp.waypoints[seg_i].get("t") or 0.0)
    for j, seg in enumerate(bake.segments):
        if j < max(0, int(seg_i)):
            continue
        if seg.get("status") == "ok":
            seg["timing"] = "uncalc"
            seg["micro"] = []
        for sm in seg.get("samples") or []:
            if sm.get("q") is not None:
                sm["timing"] = "uncalc"
                sm["speed"] = None
                sm["accel"] = None
                sm["limit"] = None
    if t_cut is None:
        bake.baked_hops = []
        bake.baked_points = []
    else:
        bake.baked_hops = [h for h in bake.baked_hops if float(h.get("t1") or 0.0) <= t_cut + 1e-9]
        bake.baked_points = [p for p in bake.baked_points if float(p.get("t") or 0.0) <= t_cut + 1e-9]


def _ease_time_frac(mode, u):
    u = min(1.0, max(0.0, float(u)))
    if mode == "ease_in":
        return math.sqrt(u)
    if mode == "ease_out":
        return 1.0 - math.sqrt(max(0.0, 1.0 - u))
    if mode == "ease_in_out":
        x = min(1.0 - 1e-9, max(1e-9, u))
        return math.acos(1.0 - 2.0 * x) / math.pi
    if mode == "sine_in":
        return 1.0 - math.cos(0.5 * math.pi * u)
    if mode == "sine_out":
        return math.sin(0.5 * math.pi * u)
    if mode == "sine_in_out":
        return 0.5 - 0.5 * math.cos(math.pi * u)
    if mode == "cubic_in":
        return u ** (1.0 / 3.0)
    if mode == "cubic_out":
        return 1.0 - (1.0 - u) ** (1.0 / 3.0)
    if mode == "cubic_in_out":
        if u < 0.5:
            return 0.5 * (2.0 * u) ** (1.0 / 3.0)
        return 1.0 - 0.5 * (2.0 * (1.0 - u)) ** (1.0 / 3.0)
    return u


def remap_segment_times(seg_i, old_t0, old_t1, new_t0, new_t1):
    if seg_i < 0 or seg_i >= len(bake.segments):
        return
    samples = bake.segments[seg_i].get("samples") or []
    span0 = float(old_t1) - float(old_t0)
    span1 = float(new_t1) - float(new_t0)
    for k, sm in enumerate(samples):
        if k == 0:
            sm["t"] = float(new_t0)
            continue
        if k == len(samples) - 1:
            sm["t"] = float(new_t1)
            continue
        old = float(sm.get("t") if sm.get("t") is not None else old_t0)
        u = 0.5 if abs(span0) < 1e-9 else (old - float(old_t0)) / span0
        u = min(1.0, max(0.0, u))
        t = float(new_t0) + u * span1
        t = max(0.0, round(t / 0.01) * 0.01)
        sm["t"] = t
        sm.pop("t_lock", None)


def _bez(p0, p1, p2, p3, s):
    s = min(1.0, max(0.0, float(s)))
    o = 1.0 - s
    return o*o*o*p0 + 3*o*o*s*p1 + 3*o*s*s*p2 + s*s*s*p3


def _s_for_progress(c1, c2, y, steps=48):
    lo, hi = 0.0, 1.0
    for _ in range(steps):
        mid = 0.5 * (lo + hi)
        if _bez(0.0, c1[1], c2[1], 1.0, mid) < y:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def apply_custom_transition(idx, c1, c2):
    c1 = [min(1.0, max(0.0, float(c1[0]))), min(1.0, max(0.0, float(c1[1])))]
    c2 = [min(1.0, max(0.0, float(c2[0]))), min(1.0, max(0.0, float(c2[1])))]
    if c2[0] < c1[0]:
        c1[0], c2[0] = c2[0], c1[0]
    if idx < 0 or idx >= len(bake.segments):
        return False
    samples = bake.segments[idx].get("samples") or []
    if len(samples) < 2:
        return False
    t0 = float(samples[0].get("t") or 0.0)
    t1 = float(samples[-1].get("t") or t0)
    n = max(1, len(samples) - 1)
    arc = bake._arc_fracs(samples)
    for k, sm in enumerate(samples):
        if k == 0 or k == len(samples) - 1:
            continue
        u = arc[k] if arc else sm.get("u")
        if u is None:
            u = float(k) / float(n)
        s = _s_for_progress(c1, c2, float(u))
        tau = _bez(0.0, c1[0], c2[0], 1.0, s)
        t = t0 + max(0.0, min(1.0, tau)) * (t1 - t0)
        t = max(0.0, round(t / 0.01) * 0.01)
        sm["t"] = t
        sm.pop("t_lock", None)
    for k in range(1, len(samples) - 1):
        lo = float(samples[k - 1]["t"]) + 0.01
        hi = float(samples[k + 1]["t"]) - 0.01
        if hi >= lo:
            samples[k]["t"] = min(max(float(samples[k]["t"]), lo), hi)
            samples[k].pop("t_lock", None)
    bake.custom_curves[int(idx)] = [list(c1), list(c2)]
    bake.segments[idx]["transition"] = "custom"
    try:
        wp.waypoints[idx]["transition"] = "custom"
        wp.waypoints[idx]["trans_c1"] = list(c1)
        wp.waypoints[idx]["trans_c2"] = list(c2)
    except Exception:
        pass
    invalidate_timing_from(idx)
    return True


def get_custom_curve(idx):
    rec = bake.custom_curves.get(int(idx))
    if rec:
        return {"c1": rec[0], "c2": rec[1]}
    return {"c1": list(bake.LINEAR_C1), "c2": list(bake.LINEAR_C2)}


def apply_transition(idx, mode, c1=None, c2=None):
    mode = str(mode or "linear")
    if mode == "custom":
        rec = get_custom_curve(idx)
        return apply_custom_transition(idx, c1 or rec["c1"], c2 or rec["c2"])
    if idx < 0 or idx >= len(bake.segments):
        return False
    samples = bake.segments[idx].get("samples") or []
    if len(samples) < 2:
        return False
    for sm in samples:
        sm.pop("t_lock", None)
        if sm is not samples[0] and sm is not samples[-1]:
            sm["t"] = None
    bake.segments[idx]["transition"] = mode
    try:
        wp.waypoints[idx]["transition"] = mode
    except Exception:
        pass
    t0 = float(wp.waypoints[idx].get("t") or samples[0].get("t") or 0.0)
    t1 = float(wp.waypoints[idx + 1].get("t") or samples[-1].get("t") or t0)
    bake._apply_seg_time_profile(bake.segments[idx], t0, t1)
    bake._tl_cache = None
    invalidate_timing_from(idx)
    bump_timeline_cache()
    return True


def set_transition(idx, mode, c1=None, c2=None):
    """UI name. Places sample times once. Later diamond drags do not reapply this."""
    try:
        idx = int(idx)
    except Exception:
        return False
    return apply_transition(idx, mode, c1=c1, c2=c2)


def invalidate_timing():
    """Keep IK (q) but drop all speed scores. Nodes go yellow."""
    invalidate_timing_from(0)


def set_sample_time(seg_i, k, t):
    if seg_i < 0 or seg_i >= len(bake.segments):
        return None
    samples = bake.segments[seg_i].get("samples") or []
    if k <= 0 or k >= len(samples) - 1:
        return None
    prev_t = float(samples[k - 1].get("t") or 0.0)
    next_t = float(samples[k + 1].get("t") or prev_t)
    lo = prev_t + 0.01
    hi = next_t - 0.01
    if hi < lo:
        t = 0.5 * (prev_t + next_t)
    else:
        t = min(max(float(t), lo), hi)
    t = max(0.0, round(t / 0.01) * 0.01)
    samples[k]["t"] = t
    samples[k]["t_lock"] = t
    invalidate_timing_from(seg_i)
    bump_timeline_cache()
    return t


def set_sample_times(moves):
    """Batch-move interior samples. Returns applied [{seg, index, t}]."""
    parsed = []
    for rec in moves or []:
        try:
            if isinstance(rec, dict):
                seg_i, k, t = int(rec.get("seg")), int(rec.get("index")), float(rec.get("t"))
            else:
                seg_i, k, t = int(rec[0]), int(rec[1]), float(rec[2])
        except Exception:
            continue
        if seg_i < 0 or seg_i >= len(bake.segments):
            continue
        samples = bake.segments[seg_i].get("samples") or []
        if k <= 0 or k >= len(samples) - 1:
            continue
        parsed.append((seg_i, k, t))
    for seg_i, k, t in parsed:
        sm = bake.segments[seg_i]["samples"][k]
        sm["t"] = float(t)
        sm["t_lock"] = float(t)
    applied = []
    dirty = None
    for seg_i, k, _t in sorted(parsed, key=lambda r: (r[0], r[1])):
        nt = set_sample_time(seg_i, k, bake.segments[seg_i]["samples"][k]["t"])
        if nt is None:
            continue
        applied.append({"seg": seg_i, "index": k, "t": nt})
        dirty = seg_i if dirty is None else min(dirty, seg_i)
    return applied


def pin_key_times():
    """Move diamond endpoints only. Interior samples stay put unless a diamond crosses them."""
    wps = wp.waypoints
    for i, seg in enumerate(bake.segments):
        if i + 1 >= len(wps):
            break
        t0 = float(wps[i].get("t") or 0.0)
        t1 = float(wps[i + 1].get("t") or t0)
        if t1 < t0:
            t0, t1 = t1, t0
        samples = seg.get("samples") or []
        if len(samples) < 2:
            continue
        samples[0]["t"] = t0
        samples[-1]["t"] = t1
        micros = seg.get("micro") or []
        if micros:
            if micros[0].get("t0") is not None:
                micros[0]["t0"] = t0
            if micros[-1].get("t1") is not None:
                micros[-1]["t1"] = t1


def _blend_knot_times():
    """Average the two span speeds at each interior diamond so the last hop of A
    and first hop of B share one v."""
    for i in range(len(bake.segments) - 1):
        a = bake.segments[i]
        b = bake.segments[i + 1]
        if a.get("dwell") or b.get("dwell") or a.get("force_fk") or b.get("force_fk"):
            continue
        if a.get("status") != "ok" or b.get("status") != "ok":
            continue
        sa = a.get("samples") or []
        sb = b.get("samples") or []
        if len(sa) < 3 or len(sb) < 3:
            continue
        t_w = float(sa[-1].get("t") or sb[0].get("t") or 0.0)
        t_a0 = float(sa[0].get("t") or 0.0)
        t_b1 = float(sb[-1].get("t") or t_w)
        la = sum(bake._sample_step_m(sa[k], sa[k + 1]) for k in range(len(sa) - 1))
        lb = sum(bake._sample_step_m(sb[k], sb[k + 1]) for k in range(len(sb) - 1))
        va = la / max(1e-9, t_w - t_a0)
        vb = lb / max(1e-9, t_b1 - t_w)
        v = 0.5 * (va + vb)
        if v < 1e-9:
            continue
        d_a = bake._sample_step_m(sa[-2], sa[-1])
        d_b = bake._sample_step_m(sb[0], sb[1])
        ta = t_w - (d_a / v if d_a > 1e-4 else 0.0)
        tb = t_w + (d_b / v if d_b > 1e-4 else 0.0)
        lo = float(sa[-3].get("t") or t_a0) if len(sa) >= 3 else t_a0
        sa[-2]["t"] = min(max(ta, lo), t_w)
        hi = float(sb[2].get("t") or t_b1) if len(sb) >= 3 else t_b1
        sb[1]["t"] = max(min(tb, hi), t_w)


def _map_span_t(t, t0o, t1o, t0n, t1n):
    try:
        t = float(t)
    except Exception:
        t = t0n
    span_o = float(t1o) - float(t0o)
    span_n = float(t1n) - float(t0n)
    if abs(span_o) < 1e-9:
        return float(t0n)
    u = (t - float(t0o)) / span_o
    if u < 0.0:
        u = 0.0
    elif u > 1.0:
        u = 1.0
    return float(t0n) + u * span_n


def stretch_sample_times(old_times, new_times=None):
    """Scale each span's sample / hop times by the keyframe time change.

    Fraction along the old [t_i, t_{i+1}] is kept on the new interval.
    IK is left intact; speed scores on touched spans are dropped.
    """
    wps = wp.waypoints
    n = len(wps)
    if n < 2:
        return False
    old = []
    for i in range(n):
        if old_times is not None and i < len(old_times) and old_times[i] is not None:
            old.append(float(old_times[i]))
        else:
            old.append(float(wps[i].get("t") or 0.0))
    new = []
    src = new_times if new_times is not None else None
    for i in range(n):
        if src is not None and i < len(src) and src[i] is not None:
            new.append(float(src[i]))
        else:
            new.append(float(wps[i].get("t") or 0.0))
    touched = None
    for i, seg in enumerate(bake.segments):
        if i + 1 >= n:
            break
        t0o, t1o = old[i], old[i + 1]
        t0n, t1n = new[i], new[i + 1]
        if abs(t0o - t0n) < 1e-9 and abs(t1o - t1n) < 1e-9:
            continue
        if touched is None:
            touched = i
        for sm in seg.get("samples") or []:
            if sm.get("t") is not None:
                sm["t"] = _map_span_t(sm["t"], t0o, t1o, t0n, t1n)
            if sm.get("t_lock") is not None:
                sm["t_lock"] = _map_span_t(sm["t_lock"], t0o, t1o, t0n, t1n)
        for hop in seg.get("micro") or []:
            if hop.get("t0") is not None:
                hop["t0"] = _map_span_t(hop["t0"], t0o, t1o, t0n, t1n)
            if hop.get("t1") is not None:
                hop["t1"] = _map_span_t(hop["t1"], t0o, t1o, t0n, t1n)
    if touched is None:
        return False
    def _hop_in_old(h):
        try:
            t0 = float(h.get("t0") or 0.0)
            t1 = float(h.get("t1") or t0)
        except Exception:
            return False
        mid = 0.5 * (t0 + t1)
        for i in range(n - 1):
            lo, hi = (old[i], old[i + 1]) if old[i] <= old[i + 1] else (old[i + 1], old[i])
            if lo - 1e-6 <= mid <= hi + 1e-6:
                h["t0"] = _map_span_t(t0, old[i], old[i + 1], new[i], new[i + 1])
                h["t1"] = _map_span_t(t1, old[i], old[i + 1], new[i], new[i + 1])
                return True
        return True
    bake.baked_hops = [h for h in (bake.baked_hops or []) if _hop_in_old(h) or True]
    for p in bake.baked_points or []:
        try:
            t = float(p.get("t") or 0.0)
        except Exception:
            continue
        for i in range(n - 1):
            lo, hi = (old[i], old[i + 1]) if old[i] <= old[i + 1] else (old[i + 1], old[i])
            if lo - 1e-6 <= t <= hi + 1e-6:
                p["t"] = _map_span_t(t, old[i], old[i + 1], new[i], new[i + 1])
                break
    bake._tl_cache = None
    return True


def _seg_ui_state(seg):
    """white unbaked / red fail / grey IK only / yellow KF-span over limit / green ok."""
    if seg is None:
        return None
    if seg.get("status") == "fail":
        return "fail"
    has_q = any(sm.get("q") is not None for sm in (seg.get("samples") or []))
    if seg.get("status") != "ok" and not has_q:
        return "unbaked"
    if seg.get("timing") == "too_fast":
        return "too_fast"
    if seg.get("force_fk") or bake._seg_force_fk(seg.get("i0") if seg.get("i0") is not None else -1):
        if seg.get("status") == "ok" and seg.get("timing") in ("ok", "fk"):
            return "force_fk"
        return "unbaked"
    if seg.get("timing") == "ok":
        return "ok"
    if has_q or seg.get("status") == "ok":
        return "uncalc"
    return "unbaked"


def _iter_scored_hops():
    """All scored hops. A yellow span does not hide later spans."""
    hops = []
    for seg in bake.segments:
        if seg.get("status") != "ok":
            continue
        hops.extend(list(seg.get("micro") or []))
    if hops:
        return hops
    return list(bake.baked_hops or [])


def _node_color_map():
    """
    White = no IK. Yellow = IK, speed not checked yet.
    Orange (too_fast) = this hop failed speed. Green = speed passed.
    Hop color matches the hop, not a default.
    """
    key_st = {}
    sample_st = {}
    nodes = []
    n = len(wp.waypoints)
    for i, w in enumerate(wp.waypoints):
        left = bake.segments[i - 1] if i > 0 and i - 1 < len(bake.segments) else None
        right = bake.segments[i] if i < len(bake.segments) else None
        if (left and left.get("status") == "fail") or (right and right.get("status") == "fail"):
            key_st[i] = "fail"
        elif (left and left.get("status") == "ok") or (right and right.get("status") == "ok"):
            key_st[i] = "uncalc"
        else:
            key_st[i] = "unbaked"
        try:
            nodes.append((float(w.get("t") or 0.0), 0, "key", i))
        except Exception:
            pass
    for i, seg in enumerate(bake.segments):
        sms = seg.get("samples") or []
        for k, sm in enumerate(sms):
            if k == 0 or k == len(sms) - 1:
                continue
            if sm.get("q") is None and seg.get("status") != "ok":
                continue
            sample_st[(i, k)] = "fail" if seg.get("status") == "fail" else (
                "uncalc" if (sm.get("q") is not None or seg.get("status") == "ok") else "unbaked"
            )
            try:
                nodes.append((float(sm.get("t") or 0.0), 1, "sample", (i, k)))
            except Exception:
                pass
    nodes.sort()

    def _set(kind, ref, st):
        if kind == "key":
            if key_st.get(ref) == "fail":
                return
            key_st[ref] = st
        else:
            if sample_st.get(ref) == "fail":
                return
            sample_st[ref] = st

    # Each span keeps its own state: a too-fast span does not repaint later
    # spans yellow (yellow means "not checked yet").
    for i, seg in enumerate(bake.segments):
        st = _seg_ui_state(seg) or "unbaked"
        sms = seg.get("samples") or []
        for k, sm in enumerate(sms):
            if k == 0 or k == len(sms) - 1:
                continue
            if (i, k) in sample_st:
                sample_st[(i, k)] = st
        if key_st.get(i) != "fail":
            if st == "too_fast":
                key_st[i] = "too_fast"
            elif st in ("fk", "force_fk"):
                key_st[i] = "fk"
            elif st == "ok" and key_st.get(i) not in ("too_fast", "fk"):
                key_st[i] = "ok"
            elif st == "uncalc":
                key_st[i] = "uncalc"
        if key_st.get(i + 1) != "fail":
            if st == "too_fast":
                key_st[i + 1] = "too_fast"
            elif st in ("fk", "force_fk"):
                key_st[i + 1] = "fk"
            elif st == "ok" and key_st.get(i + 1) not in ("too_fast", "fk"):
                key_st[i + 1] = "ok"
            elif st == "uncalc":
                key_st[i + 1] = "uncalc"
    return key_st, sample_st


def keyframe_status(i):
    if not wp.waypoints:
        return "empty"
    key_st, _ = _node_color_map()
    return key_st.get(i, "unbaked")


def _spans_from_timed():
    """Hops for the IK-ok prefix only. Do not jump over a pending first span.

    Speed hops skip the diamond itself: last interior of A → first interior of B.
    """
    bake.assign_sample_times()
    wmax, amax = bake._joint_wmax()
    pts = []
    for seg in bake.segments:
        if seg.get("status") != "ok":
            break
        sms = [sm for sm in (seg.get("samples") or []) if sm.get("q") is not None]
        for k, sm in enumerate(sms):
            pts.append({
                "t": float(sm.get("t") or 0.0),
                "q": [float(v) for v in np.asarray(sm["q"], dtype=float).reshape(-1)[:6]],
                "pos": sm.get("pos"),
                "is_wp": (k == 0 or k == len(sms) - 1),
            })
    out = []
    i = 0
    while i < len(pts) - 1:
        a = pts[i]
        j = i + 1
        while j < len(pts):
            b = pts[j]
            dt = float(b["t"]) - float(a["t"])
            same_q = max(abs(a["q"][k] - b["q"][k]) for k in range(6)) < 1e-4
            if dt >= 0.002 and not (same_q and dt < 0.02):
                break
            j += 1
        if j >= len(pts):
            break
        b = pts[j]
        rec = bake._hop_metrics(a, b, wmax, amax)
        if rec.get("dt", 0) < 0.002:
            rec["over"] = False
            rec["over_vel"] = False
            rec["status"] = "ok"
            rec["speed"] = 0.0
        elif rec.get("over_vel") and rec.get("dt", 0) < 0.05 and rec.get("lin_m", 0) > 0.002:
            rec["status"] = "too_fast"
            rec["over"] = True
        else:
            rec["over"] = False
            rec["status"] = "ok"
        out.append(rec)
        i = j
    return out


def bump_timeline_cache():
    bake._tl_cache = None


def timeline_payload():
    try:
        return _timeline_payload_body()
    except Exception as e:
        print("timeline_payload failed", e)
        kfs = []
        for i, w in enumerate(wp.waypoints or []):
            kfs.append({
                "index": i,
                "track": "arm",
                "kind": "key",
                "t": None if w.get("t") is None else float(w.get("t") or 0.0),
                "status": "unbaked",
                "clip_id": w.get("clip_id"),
                "clip_name": w.get("clip_name"),
                "clip_locked": bool(w.get("clip_locked")),
            })
        return {"keyframes": kfs, "samples": [], "spans": [], "runnable": False, "sim_runnable": False}


def _timeline_payload_body():
    cache_key = (
        bake._bake_generation,
        len(wp.waypoints),
        bool(bake._speed_valid),
        len(bake.baked_hops),
        tuple(s.get("status") for s in bake.segments),
        tuple(s.get("timing") for s in bake.segments),
        tuple(len(s.get("samples") or []) for s in bake.segments),
        tuple(round(float(w.get("t") or 0.0), 3) for w in wp.waypoints),
        tuple(
            tuple(round(float(sm.get("t") or 0.0), 3) for sm in (s.get("samples") or [])[:24])
            for s in bake.segments
        ),
    )
    if bake._tl_cache and bake._tl_cache.get("key") == cache_key:
        data = bake._tl_cache["data"]
        kfs = data.get("keyframes") or []
        for i, kf in enumerate(kfs):
            if i < len(wp.waypoints):
                tw = wp.waypoints[i].get("t")
                kf["t"] = None if tw is None else float(tw)
        return data
    kfs = []
    key_st, sample_st = _node_color_map()
    for i, w in enumerate(wp.waypoints):
        st = key_st.get(i)
        if st is None:
            st = "unbaked"
        kfs.append({
            "index": i,
            "track": "arm",
            "kind": "key",
            "t": None if w.get("t") is None else float(w["t"]),
            "pos": [float(v) for v in np.asarray(w.get("pos", [0, 0, 0])).reshape(-1)[:3]],
            "wxyz": [float(v) for v in np.asarray(w.get("wxyz", [1, 0, 0, 0])).reshape(-1)[:4]],
            "q": None if w.get("q") is None else [float(v) for v in np.asarray(w["q"]).reshape(-1)[:6]],
            "status": st,
            "mode": w.get("mode", "arc"),
            "force_fk": bool(w.get("force_fk")),
            "clip_id": w.get("clip_id"),
            "clip_name": w.get("clip_name"),
            "clip_locked": bool(w.get("clip_locked")),
        })
        if w.get("clip_id"):
            left = bake.segments[i - 1] if i > 0 and i - 1 < len(bake.segments) else None
            right = bake.segments[i] if i < len(bake.segments) else None
            baked = False
            for seg in (left, right):
                if seg and seg.get("status") == "ok" and seg.get("timing") in ("ok", "fk", "too_fast"):
                    baked = True
            if not baked:
                kfs[-1]["status"] = "unbaked"
    samples = []
    spans = []
    for seg in bake.segments:
        if seg.get("status") != "ok":
            continue
        for hop in (seg.get("micro") or []):
            spans.append(dict(hop))
    if not spans:
        spans = list(bake.baked_hops) if bake.baked_hops else []
    key_times = set()
    for rec in kfs:
        if rec.get("t") is not None:
            key_times.add(round(float(rec["t"]), 3))
    for i, seg in enumerate(bake.segments):
        sms = seg.get("samples") or []
        micros = seg.get("micro") or []
        seg_ui = _seg_ui_state(seg)
        for k, sm in enumerate(sms):
            if sm.get("q") is None and seg.get("status") != "ok":
                continue
            t = float(sm.get("t") or 0.0)
            is_end = (k == 0 or k == len(sms) - 1)
            if is_end:
                continue
            if any(abs(t - kt) < 0.012 for kt in key_times):
                continue
            u = sm.get("u")
            if u is not None and (float(u) < 0.02 or float(u) > 0.98):
                continue
            hop = None
            if not is_end and (k - 1) < len(micros):
                hop = micros[k - 1]
            elif micros:
                hop = micros[min(k, len(micros) - 1)]
            st = sample_st.get((i, k))
            if st is None:
                if seg_ui == "ok":
                    st = "ok"
                elif seg.get("status") == "fail":
                    st = "fail"
                elif sm.get("q") is not None or seg.get("status") == "ok":
                    st = "uncalc"
                else:
                    st = "unbaked"
            samples.append({
                "seg": i,
                "index": k,
                "kind": "sample",
                "t": t,
                "status": st,
                "speed": None if hop is None else hop.get("speed"),
                "limit": None if hop is None else hop.get("limit"),
            })
    if not samples and bake.baked_points:
        for i, p in enumerate(bake.baked_points):
            t = float(p.get("t") or 0.0)
            if round(t, 3) in key_times:
                continue
            samples.append({
                "seg": 0,
                "index": i,
                "kind": "sample",
                "t": t,
                "status": "too_fast" if p.get("over") else "ok",
                "speed": p.get("speed"),
                "limit": p.get("limit"),
            })
    ok_all = bool(bake.segments) and all(s.get("status") == "ok" for s in bake.segments)
    time_ok = bake.is_runnable()
    norm_spans = []
    for hop in spans:
        rec = dict(hop)
        mid = 0.5 * (float(rec.get("t0") or 0.0) + float(rec.get("t1") or 0.0))
        st = "ok" if bake._speed_valid else "uncalc"
        for seg in bake.segments:
            sms = [sm for sm in (seg.get("samples") or []) if sm.get("t") is not None]
            if len(sms) < 2:
                continue
            t0 = float(sms[0]["t"])
            t1 = float(sms[-1]["t"])
            lo, hi = (t0, t1) if t0 <= t1 else (t1, t0)
            if lo - 1e-6 <= mid <= hi + 1e-6:
                st = _seg_ui_state(seg) or st
                break
        rec["status"] = st
        dt = float(rec.get("dt") or 0.0)
        if rec.get("lin_mps") is None:
            lin_m = rec.get("lin_m")
            if lin_m is not None and dt > 1e-9:
                rec["lin_mps"] = float(lin_m) / dt
            else:
                rec["lin_mps"] = None
        norm_spans.append(rec)
    spans = norm_spans
    data = {
        "keyframes": kfs,
        "samples": samples,
        "spans": spans,
        "runnable": time_ok,
        "sim_runnable": bake.is_ik_playable(),
        "ik_ok": ok_all,
        "path_distance": bake.path_distance(),
        "t_end": max([kf["t"] or 0 for kf in kfs] + [0.0]),
    }
    bake._tl_cache = {"key": cache_key, "data": data}
    return data


def _kf_chain(sel_kf):
    idxs = sorted(set(int(i) for i in (sel_kf or []) if 0 <= int(i) < len(wp.waypoints)))
    if not idxs:
        return []
    chains = []
    cur = [idxs[0]]
    for i in idxs[1:]:
        if i == cur[-1] + 1:
            cur.append(i)
        else:
            chains.append(cur)
            cur = [i]
    chains.append(cur)
    chains.sort(key=lambda c: (-len(c), c[0]))
    return chains[0]


def _ctrl_list(w, which):
    p = w.get(f"{which}_ctrl_pos")
    if p is None:
        return None
    return [float(v) for v in np.asarray(p, dtype=float).reshape(-1)[:3]]


def copy_selection(sel_kf, sel_sm=None):
    chain = _kf_chain(sel_kf)
    extra = []
    for i in list(chain):
        cid = str(wp.waypoints[i].get("clip_id") or "")
        if not cid:
            continue
        extra.extend([j for j, w in enumerate(wp.waypoints) if str(w.get("clip_id") or "") == cid])
    if extra:
        chain = sorted(set(list(chain) + extra))
    if not chain:
        bake.clipboard = None
        return {"ok": False, "error": "no keyframes"}
    t0 = float(wp.waypoints[chain[0]].get("t") or 0.0)
    t1 = float(wp.waypoints[chain[-1]].get("t") or t0)
    keys = []
    for i in chain:
        w = wp.waypoints[i]
        keys.append({
            "rel_t": float(w.get("t") or 0.0) - t0,
            "pos": [float(v) for v in np.asarray(w.get("pos", [0, 0, 0])).reshape(-1)[:3]],
            "wxyz": [float(v) for v in np.asarray(w.get("wxyz", [1, 0, 0, 0])).reshape(-1)[:4]],
            "mode": w.get("mode", "arc"),
            "in_ctrl": _ctrl_list(w, "in"),
            "out_ctrl": _ctrl_list(w, "out"),
            "clip_id": w.get("clip_id"),
            "clip_name": w.get("clip_name"),
            "clip_path": w.get("clip_path"),
            "clip_locked": bool(w.get("clip_locked")),
        })
    bake.clipboard = {"t0": t0, "span": max(0.0, t1 - t0), "keys": keys}
    return {"ok": True, "span": bake.clipboard["span"], "keys": len(keys), "segments": max(0, len(keys) - 1)}


def cut_selection(sel_kf, sel_sm=None):
    info = copy_selection(sel_kf, sel_sm)
    if not info.get("ok") or not bake.clipboard:
        return info
    chain = _kf_chain(sel_kf)
    for i in reversed(chain):
        if 0 <= i < len(wp.waypoints):
            try:
                wp.selected_idx = i
                wp.delete_selected()
            except Exception:
                pass
    return info


def _occupied(t0, t1):
    for w in wp.waypoints:
        tw = w.get("t")
        if tw is None:
            continue
        if t0 - 1e-6 <= float(tw) <= t1 + 1e-6:
            return True
    for seg in bake.segments:
        for sm in seg.get("samples") or []:
            tw = sm.get("t")
            if tw is None:
                continue
            if t0 - 1e-6 <= float(tw) <= t1 + 1e-6:
                return True
    return False


def can_paste(t0):
    if not bake.clipboard:
        return False
    t0 = max(0.0, float(t0))
    t1 = t0 + float(bake.clipboard["span"])
    return not _occupied(t0, t1)


def paste_clipboard(t0):
    if not bake.clipboard:
        return {"ok": False, "error": "clipboard empty"}
    t0 = max(0.0, round(float(t0) / 0.01) * 0.01)
    if not can_paste(t0):
        return {"ok": False, "error": "target range occupied"}
    keys = bake.clipboard["keys"]
    created_t = []
    for k in keys:
        t = t0 + float(k["rel_t"])
        wp.add_waypoint(
            k["pos"], mode=k.get("mode") or "arc", wxyz=k.get("wxyz"),
            t=t, q=None, replace=False, skip_ctrls=True,
        )
        created_t.append(t)
    wp.waypoints.sort(key=lambda w: float(w.get("t") or 0.0))
    start = None
    for i, w in enumerate(wp.waypoints):
        if created_t and abs(float(w.get("t") or -1) - created_t[0]) < 1e-4:
            start = i
            break
    if start is None:
        return {"ok": True, "t": t0, "n": len(keys)}
    clip_src = next((k.get("clip_id") for k in keys if k.get("clip_id")), None)
    clip_id = None
    if clip_src:
        if not hasattr(wp, "place_saved_path"):
            clip_id = "clip_paste"
        else:
            seq = int(getattr(wp.place_saved_path, "_clip_seq", 0) or 0) + 1
            wp.place_saved_path._clip_seq = seq
            clip_id = "clip_%d" % seq
    for j, k in enumerate(keys):
        idx = start + j
        if idx >= len(wp.waypoints):
            break
        if clip_id:
            wp.waypoints[idx]["clip_id"] = clip_id
            wp.waypoints[idx]["clip_name"] = k.get("clip_name")
            wp.waypoints[idx]["clip_path"] = k.get("clip_path")
            wp.waypoints[idx]["clip_locked"] = True
        if k.get("in_ctrl") is not None:
            wp.set_ctrl_pos(idx, "in", k["in_ctrl"])
        if k.get("out_ctrl") is not None:
            wp.set_ctrl_pos(idx, "out", k["out_ctrl"])
    end = start + len(keys) - 1
    if start > 0 and wp.waypoints[start].get("in_ctrl_pos") is None:
        wp.place_segment_ctrls(start - 1, start)
    if end + 1 < len(wp.waypoints) and wp.waypoints[end].get("out_ctrl_pos") is None:
        wp.place_segment_ctrls(end, end + 1)
    if hasattr(wp, "draw_path"):
        wp.draw_path()
    bake.invalidate_from(start)
    return {"ok": True, "t": t0, "n": len(keys), "start": start}


def clipboard_info(t0=0.0):
    return {
        "has": bool(bake.clipboard),
        "span": None if not bake.clipboard else float(bake.clipboard["span"]),
        "can_paste": can_paste(t0) if bake.clipboard else False,
        "curve": None,
    }



# Re-expose on the original modules so existing wp.* / bake.* callers keep working.
_WAYPOINT_API = (
    "_segment_length_m",
    "apply_linear_speed",
    "_first_compressed_span",
    "set_keyframe_times",
    "set_keyframe_time",
)
_BAKE_API = (
    "select_sample",
    "get_selected_sample",
    "invalidate_timing_span",
    "invalidate_timing_from",
    "_ease_time_frac",
    "remap_segment_times",
    "_bez",
    "_s_for_progress",
    "apply_custom_transition",
    "get_custom_curve",
    "apply_transition",
    "set_transition",
    "invalidate_timing",
    "set_sample_time",
    "set_sample_times",
    "pin_key_times",
    "_blend_knot_times",
    "_map_span_t",
    "stretch_sample_times",
    "_seg_ui_state",
    "_iter_scored_hops",
    "_node_color_map",
    "keyframe_status",
    "_spans_from_timed",
    "bump_timeline_cache",
    "timeline_payload",
    "_timeline_payload_body",
    "_kf_chain",
    "_ctrl_list",
    "copy_selection",
    "cut_selection",
    "_occupied",
    "can_paste",
    "paste_clipboard",
    "clipboard_info",
)
for _name in _WAYPOINT_API:
    setattr(wp, _name, globals()[_name])
for _name in _BAKE_API:
    setattr(bake, _name, globals()[_name])
