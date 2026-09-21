# version: beta.0904
"""Save / open / export for .arcspath and export.json."""
import json
import math
import time
from pathlib import Path
import numpy as np

import waypoints as wp
import path_bake as bake
import robot_runtime as rr

EXPORT_NAME = "export.json"
SAVE_NAME = "path.arcspath"


def _dump_track(tr):
    rec = {}
    if tr.get("id"):
        rec["id"] = str(tr.get("id"))
    rec["name"] = str(tr.get("name") or rec.get("id") or "Track")
    keys = []
    for k in list(tr.get("keys") or []):
        if not isinstance(k, dict):
            continue
        item = {}
        if k.get("t") is not None:
            try:
                item["t"] = float(k.get("t"))
            except Exception:
                continue
        if "rec" in k or "record" in k or "v" in k:
            item["rec"] = bool(k.get("rec") if "rec" in k else (k.get("record") if "record" in k else k.get("v")))
        for extra in ("kind", "name", "on"):
            if k.get(extra) is not None:
                item[extra] = k.get(extra)
        if item:
            keys.append(item)
    rec["keys"] = keys
    return rec
DEFAULT_ACC = 0.2
DEFAULT_BLEND = 0.0
DEFAULT_SPEED = 0.05
FORMAT_VERSION = "beta.0904"


def _dump_sim_poses():
    """Custom Sim Go Pose entries only (Real/Home are rebuilt on load)."""
    out = []
    for p in list(getattr(wp, "sim_poses", None) or []):
        if not isinstance(p, dict):
            continue
        pid = str(p.get("id") or "")
        if p.get("builtin") or pid in ("real", "home"):
            continue
        if str(p.get("store") or "local") == "global":
            continue
        rec = {
            "id": pid or ("pose_%d" % (len(out) + 1)),
            "name": str(p.get("name") or ""),
            "builtin": False,
            "store": "local",
            "added_at": float(p.get("added_at") or 0.0),
        }
        ik = _as6(p.get("ik"))
        q = _as6(p.get("q"))
        q_deg = _as6(p.get("q_deg"))
        if ik:
            rec["ik"] = [round(float(v), 6) for v in ik]
        if q:
            rec["q"] = [round(float(v), 6) for v in q]
        if q_deg:
            rec["q_deg"] = [round(float(v), 6) for v in q_deg]
        out.append(rec)
    return out


def _load_sim_poses(raw):
    loaded = []
    max_seq = 0
    if not isinstance(raw, list):
        raw = []
    for i, p in enumerate(raw):
        if not isinstance(p, dict):
            continue
        pid = str(p.get("id") or ("pose_%d" % (i + 1)))
        if pid in ("real", "home") or p.get("builtin"):
            continue
        rec = {
            "id": pid,
            "name": str(p.get("name") or ""),
            "builtin": False,
            "store": str(p.get("store") or "local"),
            "added_at": float(p.get("added_at") or 0.0),
        }
        ik = _as6(p.get("ik")) or [0.0] * 6
        q = _as6(p.get("q"))
        q_deg = _as6(p.get("q_deg"))
        rec["ik"] = ik
        if q:
            rec["q"] = q
        elif q_deg:
            rec["q"] = [float(v) * math.pi / 180.0 for v in q_deg]
        else:
            rec["q"] = [0.0] * 6
        if q_deg:
            rec["q_deg"] = q_deg
        loaded.append(rec)
        if pid.startswith("pose_"):
            try:
                max_seq = max(max_seq, int(pid.split("_", 1)[1]))
            except Exception:
                pass
    wp.sim_poses = loaded
    if hasattr(wp, "_sim_pose_seq"):
        wp._sim_pose_seq = max(int(getattr(wp, "_sim_pose_seq", 0) or 0), max_seq)
    if hasattr(wp, "ensure_builtin_sim_poses"):
        wp.ensure_builtin_sim_poses()
    if hasattr(wp, "refresh_pose_markers"):
        wp.refresh_pose_markers()
    try:
        import webui
        wp.load_global_stored_points((webui.load_settings() or {}).get("stored_points") or [])
    except Exception:
        pass
    return loaded


def _dump_scene_objects():
    out = []
    for obj in list(getattr(wp, "scene_objects", None) or []):
        if not isinstance(obj, dict):
            continue
        out.append({
            "id": obj.get("id"),
            "kind": obj.get("kind") or "empty",
            "name": obj.get("name") or "",
            "pos": [float(v) for v in np.asarray(obj.get("pos", [0, 0, 0])).reshape(-1)[:3]],
            "wxyz": [float(v) for v in np.asarray(obj.get("wxyz", [1, 0, 0, 0])).reshape(-1)[:4]],
            "scale": [float(v) for v in np.asarray(obj.get("scale", [1, 1, 1])).reshape(-1)[:3]],
            "parent_id": obj.get("parent_id"),
            "space": obj.get("space") or "world",
            "local_pos": [float(v) for v in np.asarray(obj.get("local_pos", obj.get("pos", [0, 0, 0]))).reshape(-1)[:3]],
            "local_wxyz": [float(v) for v in np.asarray(obj.get("local_wxyz", obj.get("wxyz", [1, 0, 0, 0]))).reshape(-1)[:4]],
            "camera": dict(obj.get("camera") or {}) if obj.get("kind") == "camera" else None,
        })
    return out


def _load_scene_objects(raw):
    if hasattr(wp, "clear_scene_objects"):
        wp.clear_scene_objects()
    if not isinstance(raw, list):
        return []
    loaded = []
    for rec in raw:
        if not isinstance(rec, dict):
            continue
        if hasattr(wp, "restore_scene_object"):
            obj = wp.restore_scene_object(rec)
        else:
            obj = wp.add_scene_object(rec.get("kind") or "empty")
            wp.update_scene_object(obj.get("id"), name=rec.get("name"), pos=rec.get("pos"), wxyz=rec.get("wxyz"), scale=rec.get("scale"))
        if obj:
            loaded.append(obj)
    return loaded


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


def _wrap_pi(a):
    """Map angle to (-pi, pi]. Accepts degrees if |a| > pi and |a| <= 180.5."""
    try:
        a = float(a)
    except Exception:
        return 0.0
    if abs(a) > math.pi + 1e-6 and abs(a) <= 180.5:
        a = math.radians(a)
    a = (a + math.pi) % (2.0 * math.pi) - math.pi
    if a <= -math.pi:
        a += 2.0 * math.pi
    return a


def _wrap_angles6(vals, cartesian=False):
    vals = _as6(vals)
    if vals is None:
        return None
    if cartesian:
        return [vals[0], vals[1], vals[2], _wrap_pi(vals[3]), _wrap_pi(vals[4]), _wrap_pi(vals[5])]
    return [_wrap_pi(v) for v in vals]


def _truthy_wp(val):
    if val is True:
        return True
    if val is False or val is None:
        return False
    return str(val).strip().lower() in ("1", "true", "yes")


def _joint_move_duration(dq, speed, acc):
    dq = abs(float(dq))
    v = max(1e-9, float(speed))
    a = max(1e-9, float(acc))
    if dq <= 1e-12:
        return 0.0
    if dq <= (v * v) / a:
        return 2.0 * math.sqrt(dq / a)
    return (v / a) + (dq / v)


def _speeds_from_samples(recs):
    speeds = []
    for i, rec in enumerate(recs):
        if i + 1 < len(recs):
            q0 = np.asarray(rec.get("q") or [0] * 6, dtype=float).reshape(-1)[:6]
            q1 = np.asarray(recs[i + 1].get("q") or q0, dtype=float).reshape(-1)[:6]
            dt = max(1e-6, float(recs[i + 1]["t"] - rec["t"]))
            dq = float(np.max(np.abs(q1 - q0))) if q0.size and q1.size else 0.0
            speeds.append(dq / dt)
        else:
            speeds.append(speeds[-1] if speeds else DEFAULT_SPEED)
    return speeds


def build_motion_doc(include_waypoint_flag=False, tracks=None):
    recs = bake._iter_motion_samples()
    stored = getattr(bake, "baked_points", None) or []
    speeds = []
    if stored and len(stored) >= len(recs):
        speeds = [float(p.get("speed") or DEFAULT_SPEED) for p in stored]
    else:
        speeds = _speeds_from_samples(recs) if recs else []
    t0 = recs[0]["t"] if recs else 0.0
    t1 = recs[-1]["t"] if recs else 0.0
    points = []
    t_prev_out = t0
    for i, rec in enumerate(recs):
        tt = float(rec["t"])
        item = {
            "name": "p%03d" % i,
            "joints": [round(float(v), 6) for v in rec["q"]],
            "speed": round(float(speeds[i] if i < len(speeds) else DEFAULT_SPEED), 6),
            "acc": round(float((stored[i].get("acc") if i < len(stored) else None) or DEFAULT_ACC), 6),
            "blend": DEFAULT_BLEND,
            "dur": round(max(0.0, tt - t_prev_out), 6),
        }
        if include_waypoint_flag:
            item["is_waypoint"] = bool(rec["is_waypoint"])
            item["t"] = rec["t"]
            item["ik"] = bake._ik_from_pos_wxyz(rec["pos"], rec["wxyz"])
        points.append(item)
        t_prev_out = tt
    doc = {
        "total_pts": len(points),
        "estimated_seconds_per_lap": round(float(max(0.0, t1 - t0)), 4),
        "points": points,
    }
    if include_waypoint_flag:
        doc["format"] = "arcspath"
        doc["version"] = FORMAT_VERSION
        doc["saved_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        doc["spacing"] = bake.spacing
        doc["cursor"] = _sim_cursor_record()
        src_tracks = tracks
        if src_tracks is None:
            if hasattr(wp, "sync_camera_tracks"):
                try:
                    wp.sync_camera_tracks()
                except Exception:
                    pass
            src_tracks = getattr(wp, "extra_tracks", None) or []
        doc["tracks"] = [_dump_track(tr) for tr in (src_tracks or []) if isinstance(tr, dict)]
        doc["sim_poses"] = _dump_sim_poses()
        doc["objects"] = _dump_scene_objects()
        doc["waypoints"] = []
        nwp = len(wp.waypoints)
        t_prev_wp = None
        for i, w in enumerate(wp.waypoints):
            out_p = None if i == nwp - 1 or w.get("out_ctrl_pos") is None else [float(v) for v in np.asarray(w["out_ctrl_pos"]).reshape(-1)[:3]]
            in_p = None if i == 0 or w.get("in_ctrl_pos") is None else [float(v) for v in np.asarray(w["in_ctrl_pos"]).reshape(-1)[:3]]
            tw = None if w.get("t") is None else float(w["t"])
            if tw is None:
                dur_w = None
            elif t_prev_wp is None:
                dur_w = 0.0
            else:
                dur_w = max(0.0, tw - t_prev_wp)
            if tw is not None:
                t_prev_wp = tw
            doc["waypoints"].append({
                "name": str(w.get("name") or ("Waypoint %d" % i)),
                "added_at": float(w.get("added_at") or 0.0),
                "t": tw,
                "dur": None if dur_w is None else round(float(dur_w), 6),
                "ik": bake._ik_from_pos_wxyz(w.get("pos"), w.get("wxyz")),
                "joints": None if w.get("q") is None else [float(v) for v in np.asarray(w["q"]).reshape(-1)[:6]],
                "mode": w.get("mode", "arc"),
                "pos": None if w.get("pos") is None else [float(v) for v in np.asarray(w["pos"]).reshape(-1)[:3]],
                "wxyz": None if w.get("wxyz") is None else [float(v) for v in np.asarray(w["wxyz"]).reshape(-1)[:4]],
                "out_ctrl_pos": out_p,
                "in_ctrl_pos": in_p,
                "force_fk": bool(w.get("force_fk")),
                "q_dirty": bool(w.get("q_dirty")),
                "added_from": w.get("added_from"),
                "track_object": None if not w.get("track_object") else dict(w.get("track_object")),
                "is_waypoint": True,
            })
        if hasattr(bake, "export_bake_state"):
            doc["bake"] = bake.export_bake_state()
    return doc


def _sim_cursor_record():
    rec = {"ik": None, "joints": None}
    try:
        pos, wxyz = rr.get_sim_cursor_pose()
        rec["ik"] = bake._ik_from_pos_wxyz(pos, wxyz)
    except Exception:
        pass
    try:
        if rr.robot_sim is not None:
            rec["joints"] = [float(v) for v in np.asarray(rr.robot_sim.angle_vector()).reshape(-1)[:6]]
    except Exception:
        pass
    return rec


def _safe_stem(name, fallback):
    s = str(name or "").strip()
    s = "".join(ch if ch not in '\\/:*?"<>|' else " " for ch in s)
    s = " ".join(s.split())
    for ext in (".arcspath", ".json"):
        if s.lower().endswith(ext):
            s = s[: -len(ext)]
    return s or fallback


def write_export_json(path=None, title=None):
    data = build_motion_doc(include_waypoint_flag=False)
    data["sim_poses"] = _dump_sim_poses()
    data["objects"] = _dump_scene_objects()
    stem = _safe_stem(title or data.get("title"), "export")
    data["title"] = stem
    dest = Path(path) if path else Path(__file__).resolve().parent / (stem + ".json")
    dest.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print("exported", dest, "pts", data["total_pts"])
    return {"ok": True, "path": str(dest), "count": data["total_pts"], "doc": data, "title": stem}


def write_save_file(path=None, title=None, tracks=None):
    data = build_motion_doc(include_waypoint_flag=True, tracks=tracks)
    stem = _safe_stem(title or data.get("title"), "path")
    data["title"] = stem
    dest = Path(path) if path else Path(__file__).resolve().parent / (stem + ".arcspath")
    dest.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print("saved", dest, "pts", data["total_pts"])
    return {"ok": True, "path": str(dest), "count": data["total_pts"], "doc": data, "title": stem}


def resolve_point_fields(raw, index=0, t_prev=0.0, q_prev=None, pos_prev=None):
    p = raw if isinstance(raw, dict) else {}
    name = p.get("name") or ("p%03d" % index)
    blend = p.get("blend")
    try:
        blend = 0.0 if blend is None else float(blend)
    except Exception:
        blend = 0.0
    is_wp = _truthy_wp(p.get("is_waypoint"))

    ik = _wrap_angles6(p.get("ik"), cartesian=True)
    if ik is None:
        ik = _wrap_angles6(p.get("pose"), cartesian=True)
    if ik is None and p.get("pos") is not None:
        ik = _wrap_angles6(bake._ik_from_pos_wxyz(p.get("pos"), p.get("wxyz")), cartesian=True)
    joints = _wrap_angles6(p.get("joints") if p.get("joints") is not None else p.get("q"))

    pos = wxyz = q = None
    raw_pos = p.get("pos")
    raw_wxyz = p.get("wxyz")
    if raw_pos is not None:
        try:
            pos = np.asarray(raw_pos, dtype=float).reshape(-1)[:3]
        except Exception:
            pos = None
    if raw_wxyz is not None:
        try:
            wxyz = np.asarray(raw_wxyz, dtype=float).reshape(-1)[:4]
        except Exception:
            wxyz = None
    if pos is None and ik is not None:
        pos, wxyz_ik = bake._pos_wxyz_from_ik(ik)
        if wxyz is None:
            wxyz = wxyz_ik
    if pos is None and joints is not None:
        pos, wxyz_q = bake._pos_wxyz_from_pose_q(joints)
        if wxyz is None:
            wxyz = wxyz_q
        q = joints
    if q is None:
        q = joints

    t = p.get("t", p.get("time", None))
    try:
        t = None if t is None else float(t)
    except Exception:
        t = None
    if t is None:
        dur = p.get("dur", p.get("duration"))
        try:
            dur = None if dur is None else float(dur)
        except Exception:
            dur = None
        if dur is not None:
            t = float(t_prev) + max(0.0, dur)
    if t is None:
        speed = p.get("speed")
        acc = p.get("acc")
        if speed is None:
            speed = DEFAULT_SPEED
        if acc is None:
            acc = DEFAULT_ACC
        try:
            speed = float(speed)
        except Exception:
            speed = DEFAULT_SPEED
        try:
            acc = float(acc)
        except Exception:
            acc = DEFAULT_ACC
        dt = 0.0
        if q is not None and q_prev is not None:
            dq = float(np.max(np.abs(
                np.asarray(q, dtype=float) - np.asarray(q_prev, dtype=float)
            )))
            dt = _joint_move_duration(dq, speed, acc)
        elif pos is not None and pos_prev is not None:
            dist = float(np.linalg.norm(
                np.asarray(pos, dtype=float).reshape(3) - np.asarray(pos_prev, dtype=float).reshape(3)
            ))
            dt = dist / max(1e-9, speed)
        t = float(t_prev) + dt

    return {
        "name": str(name),
        "t": float(t),
        "ik": ik,
        "joints": q,
        "q": q,
        "pos": None if pos is None else np.asarray(pos, dtype=float).reshape(3),
        "wxyz": None if wxyz is None else np.asarray(wxyz, dtype=float).reshape(4),
        "speed": p.get("speed", DEFAULT_SPEED),
        "acc": p.get("acc", DEFAULT_ACC),
        "blend": blend,
        "dur": p.get("dur", p.get("duration")),
        "is_waypoint": is_wp,
        "mode": p.get("mode", "arc"),
        "out_ctrl_pos": p.get("out_ctrl_pos"),
        "in_ctrl_pos": p.get("in_ctrl_pos"),
        "force_fk": bool(p.get("force_fk")),
        "q_dirty": bool(p.get("q_dirty")),
        "added_from": p.get("added_from"),
        "track_object": None if not p.get("track_object") else dict(p.get("track_object")),
    }


def normalize_point_list(pts, lap_seconds=None):
    out = []
    t_prev = 0.0
    q_prev = None
    pos_prev = None
    for i, raw in enumerate(pts or []):
        rec = resolve_point_fields(
            raw, index=i, t_prev=t_prev, q_prev=q_prev, pos_prev=pos_prev
        )
        out.append(rec)
        t_prev = rec["t"]
        if rec.get("q") is not None:
            q_prev = rec["q"]
        if rec.get("pos") is not None:
            pos_prev = rec["pos"]
    if out and lap_seconds not in (None, ""):
        try:
            lap = float(lap_seconds)
        except Exception:
            lap = None
        if lap is not None and lap > 0 and out[-1]["t"] > 1e-9:
            scale = lap / out[-1]["t"]
            for rec in out:
                rec["t"] = rec["t"] * scale
        elif lap is not None and lap > 0 and len(out) > 1:
            for i, rec in enumerate(out):
                rec["t"] = lap * i / (len(out) - 1)
    return out


def parse_project_doc(data):
    if isinstance(data, str):
        data = json.loads(data)
    if not isinstance(data, dict):
        raise ValueError("not an object")
    return data


def apply_loaded_doc(data):
    data = parse_project_doc(data)
    fmt = str(data.get("format") or "").lower()
    bake.import_raw_json = fmt not in ("arcspath", "arcs")
    raw_pts = data.get("points") or []
    pts = normalize_point_list(raw_pts, lap_seconds=data.get("estimated_seconds_per_lap"))
    editor_wps = data.get("waypoints")
    if editor_wps:
        wp.load_editor_waypoints(normalize_point_list(editor_wps))
    else:
        tagged = any(p.get("is_waypoint") for p in pts)
        if tagged:
            wps = [p for p in pts if p.get("is_waypoint")]
            samples = [p for p in pts if not p.get("is_waypoint")]
        else:
            # Untagged pose lists (e.g. eight_points.json): every point is a waypoint.
            wps = pts
            samples = []
        if not wps and pts:
            wps = [pts[0], pts[-1]] if len(pts) >= 2 else list(pts)
            samples = pts[1:-1] if len(pts) > 2 else []
        wp.load_from_motion_points(wps, samples)
    raw_tracks = data.get("tracks")
    loaded_tracks = []
    if isinstance(raw_tracks, list):
        for i, tr in enumerate(raw_tracks):
            if isinstance(tr, dict):
                rec = {"name": str(tr.get("name") or ("Track %d" % (i + 1)))}
                if tr.get("id"):
                    rec["id"] = str(tr.get("id"))
                if isinstance(tr.get("keys"), list):
                    rec["keys"] = tr.get("keys")
                loaded_tracks.append(rec)
            else:
                loaded_tracks.append({"name": str(tr)})
    wp.extra_tracks = loaded_tracks
    _load_sim_poses(data.get("sim_poses"))
    _load_scene_objects(data.get("objects"))
    if hasattr(wp, "sync_camera_tracks"):
        wp.sync_camera_tracks()
    if bake.import_raw_json:
        for w in getattr(wp, "waypoints", []) or []:
            w["force_fk"] = False
        try:
            import webui
            webui.save_settings({"snap_to_grid": False})
        except Exception as e:
            print("json import snap off failed", e)
    if hasattr(wp, "draw_path"):
        wp.draw_path()
    baked = data.get("bake")
    has_real_bez = False
    stub = getattr(wp, "_ctrl_is_stub", None)
    for w in getattr(wp, "waypoints", []) or []:
        if stub is None:
            break
        if not stub(w.get("pos"), w.get("out_ctrl_pos")) or not stub(w.get("pos"), w.get("in_ctrl_pos")):
            has_real_bez = True
            break
    if baked and has_real_bez and hasattr(bake, "import_bake_state"):
        bake.import_bake_state(baked)
    else:
        bake.invalidate_from(0, drop_all=True)
    if hasattr(wp, "draw_path"):
        wp.draw_path()
    if hasattr(wp, "deselect_all"):
        wp.deselect_all()
    if hasattr(wp, "hide_unselected_handles"):
        wp.hide_unselected_handles()
    cur = data.get("cursor")
    if cur:
        try:
            rr.restore_sim_cursor(cur)
        except Exception as e:
            print("restore cursor failed:", e)
    return {"ok": True, "waypoints": len(wp.waypoints), "points": len(pts)}
