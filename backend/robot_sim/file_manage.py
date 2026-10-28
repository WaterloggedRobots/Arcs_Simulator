# version: beta.0904
"""Save / open / export for .arcspath and export.json."""
import json
import math
import time
from pathlib import Path
import numpy as np

import app_paths
import arms
import waypoints as wp
import path_bake as bake
import robot_runtime as rr

EXPORT_NAME = "export.json"
SAVE_NAME = "path.arcspath"
TEMP_NAME = "temp.json"

SESSION = {
    "path": None,
    "title": "Path editor",
    "last_save_ts": 0.0,
    "last_autosave_ts": 0.0,
    "last_hash": None,
    "recovered": False,
    "source": None,
}


def session_info():
    return dict(SESSION)


def autosave_path():
    return app_paths.ensure_dir(app_paths.STATE_DIR) / TEMP_NAME


def _stable_doc(data):
    skip = {"saved_at", "last_autosave_ts", "last_save_ts"}
    if not isinstance(data, dict):
        return data
    return {k: v for k, v in data.items() if k not in skip}


def _doc_hash(data):
    try:
        raw = json.dumps(_stable_doc(data), sort_keys=True, default=str)
    except Exception:
        raw = str(data)
    return str(hash(raw))


def _doc_has_work(data):
    if not isinstance(data, dict):
        return False
    if data.get("sim_camera") or data.get("sim_offset") or data.get("sim_collider"):
        return True
    for key in ("waypoints", "objects", "sim_poses", "tracks", "points", "arms"):
        val = data.get(key)
        if isinstance(val, list) and len(val) > 0:
            return True
    return False


def scene_is_empty():
    if len(arms.each()) > 1:
        return False
    n_wp = len(getattr(wp, "waypoints", None) or [])
    n_obj = len(getattr(wp, "scene_objects", None) or [])
    n_pose = len([p for p in (getattr(wp, "sim_poses", None) or []) if not p.get("builtin")])
    cam = getattr(wp, "sim_camera", None) or getattr(wp, "sim_offset", None) or getattr(wp, "sim_collider", None)
    tracks = [t for t in (getattr(wp, "extra_tracks", None) or []) if t]
    return n_wp == 0 and n_obj == 0 and n_pose == 0 and not cam and not tracks


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
        if p.get("pos") is not None:
            try:
                rec["pos"] = [round(float(v), 6) for v in np.asarray(p.get("pos"), dtype=float).reshape(3)]
            except Exception:
                pass
        if p.get("wxyz") is not None:
            try:
                rec["wxyz"] = [round(float(v), 6) for v in np.asarray(p.get("wxyz"), dtype=float).reshape(4)]
            except Exception:
                pass
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
        if p.get("pos") is not None:
            try:
                rec["pos"] = [float(v) for v in np.asarray(p.get("pos"), dtype=float).reshape(3)]
            except Exception:
                pass
        if p.get("wxyz") is not None:
            try:
                rec["wxyz"] = [float(v) for v in np.asarray(p.get("wxyz"), dtype=float).reshape(4)]
            except Exception:
                pass
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
            "camera": dict(obj.get("camera") or {}) if obj.get("camera") else None,
            "offset": wp.offsets.clean_offset(obj.get("offset")),
            "collider": wp.colliders.clean_collider(obj.get("collider")),
            "comp_order": list(obj.get("comp_order") or []),
            "mesh_path": obj.get("mesh_path") or None,
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


# Per-arm parts of a project file; the default arm's sit at the top level,
# each added arm's in doc["arms"]. Objects, tracks and saved paths are shared.
ARM_DOC_KEYS = (
    "points", "total_pts", "estimated_seconds_per_lap", "robot_type", "cursor",
    "sim_track", "arm_base", "sim_poses", "sim_camera", "sim_offset",
    "sim_collider", "sim_comp_order", "look_bars", "waypoints", "bake", "format",
)


def build_motion_doc(include_waypoint_flag=False, tracks=None):
    """Export (no flag): the active arm's motion. Project file: every arm."""
    if not include_waypoint_flag:
        return _build_doc_one(False, tracks)
    with arms.using(arms.DEFAULT_ID):
        doc = _build_doc_one(True, tracks)
    extra = []
    for aid in arms.each():
        if aid == arms.DEFAULT_ID:
            continue
        with arms.using(aid):
            one = _build_doc_one(True, tracks)
            rec = {k: one[k] for k in ARM_DOC_KEYS if k in one}
            rec["id"] = aid
            rec["name"] = arms.name_of(aid)
            rec["robot_ip"] = rr._robot_addr.get("ip")
            rec["robot_port"] = rr._robot_addr.get("port")
            extra.append(rec)
    if extra:
        doc["arms"] = extra
        doc["active_arm"] = arms.active()
    return doc


def _build_doc_one(include_waypoint_flag=False, tracks=None):
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
    robot_type = "aubo_i5"
    try:
        import robot_runtime as _rr
        robot_type = str(getattr(_rr, "_MODEL_ID", None) or "aubo_i5")
    except Exception:
        robot_type = "aubo_i5"
    doc = {
        "total_pts": len(points),
        "estimated_seconds_per_lap": round(float(max(0.0, t1 - t0)), 4),
        "points": points,
        "robot_type": robot_type,
    }
    if include_waypoint_flag:
        doc["format"] = "arcspath"
        doc["version"] = FORMAT_VERSION
        doc["saved_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        doc["spacing"] = bake.spacing
        doc["cursor"] = _sim_cursor_record()
        doc["sim_track"] = _sim_track_record()
        doc["arm_base"] = _arm_base_record()
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
        doc["saved_paths"] = list(getattr(wp, "saved_paths", None) or [])
        doc["objects"] = _dump_scene_objects()
        cam = getattr(wp, "sim_camera", None)
        doc["sim_camera"] = dict(cam) if cam else None
        doc["sim_offset"] = wp.offsets.clean_offset(getattr(wp, "sim_offset", None))
        doc["sim_collider"] = wp.colliders.clean_sim_collider(getattr(wp, "sim_collider", None))
        doc["sim_comp_order"] = list(getattr(wp, "sim_comp_order", None) or [])
        doc["look_bars"] = [dict(b) for b in (getattr(wp, "look_bars", None) or [])]
        doc["last_save_ts"] = float(SESSION.get("last_save_ts") or 0.0)
        doc["last_autosave_ts"] = float(SESSION.get("last_autosave_ts") or 0.0)
        if SESSION.get("path"):
            doc["source_path"] = str(SESSION.get("path"))
        try:
            import robot_runtime as _rr
            doc["robot_type"] = str(getattr(_rr, "_MODEL_ID", None) or "aubo_i5")
        except Exception:
            doc["robot_type"] = "aubo_i5"
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
                "ik_stale": bool(w.get("ik_stale")),
                "added_from": w.get("added_from"),
                "track_object": None if not w.get("track_object") else dict(w.get("track_object")),
                "is_waypoint": True,
                "clip_id": w.get("clip_id"),
                "clip_path": w.get("clip_path"),
                "clip_name": w.get("clip_name"),
                "clip_locked": bool(w.get("clip_locked")),
                "clip_head": bool(w.get("clip_head")),
                "transition": w.get("transition"),
                "lin_speed": w.get("lin_speed"),
                "trans_c1": w.get("trans_c1"),
                "trans_c2": w.get("trans_c2"),
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
    dest = Path(path) if path else app_paths.ensure_dir(app_paths.EXPORTS_DIR) / (stem + ".json")
    dest.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print("exported", dest, "pts", data["total_pts"])
    return {"ok": True, "path": str(dest), "count": data["total_pts"], "doc": data, "title": stem}


def write_save_file(path=None, title=None, tracks=None, save_as=False, pick=False):
    dest = Path(path) if path else (None if save_as or pick else (Path(SESSION["path"]) if SESSION.get("path") else None))
    if dest is None:
        picked = pick_save_path(title=title)
        if not picked:
            return {"ok": False, "cancelled": True, "error": "cancelled"}
        dest = Path(picked)
    dest.parent.mkdir(parents=True, exist_ok=True)
    data = build_motion_doc(include_waypoint_flag=True, tracks=tracks)
    stem = _safe_stem(dest.stem or title or data.get("title"), "path")
    data["title"] = stem
    now = time.time()
    data["last_save_ts"] = now
    data["source_path"] = str(dest)
    dest.write_text(json.dumps(data, indent=2), encoding="utf-8")
    SESSION["path"] = str(dest)
    SESSION["title"] = stem
    SESSION["last_save_ts"] = now
    SESSION["last_hash"] = _doc_hash(data)
    SESSION["recovered"] = False
    SESSION["source"] = "manual"
    print("saved", dest, "pts", data["total_pts"])
    return {"ok": True, "path": str(dest), "count": data["total_pts"], "doc": data, "title": stem, "last_save_ts": now}


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
        "ik_stale": bool(p.get("ik_stale")),
        "added_from": p.get("added_from"),
        "track_object": None if not p.get("track_object") else dict(p.get("track_object")),
        "clip_id": p.get("clip_id"),
        "clip_path": p.get("clip_path"),
        "clip_name": p.get("clip_name"),
        "clip_locked": bool(p.get("clip_locked")) if p.get("clip_id") else bool(p.get("clip_locked") or False),
        "clip_head": bool(p.get("clip_head")),
        "transition": p.get("transition"),
        "lin_speed": p.get("lin_speed"),
        "trans_c1": p.get("trans_c1"),
        "trans_c2": p.get("trans_c2"),
        "out_ctrl_rel": p.get("out_ctrl_rel"),
        "in_ctrl_rel": p.get("in_ctrl_rel"),
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
    inner = None
    for key in ("doc", "document", "project"):
        block = data.get(key)
        if isinstance(block, dict) and (
            "points" in block or "waypoints" in block or "format" in block
        ):
            inner = block
            break
    if inner is not None:
        data = inner
    return data


def apply_loaded_doc(data):
    data = parse_project_doc(data)
    # Added arms first, so objects parented to their cursors find them.
    arms.activate(arms.DEFAULT_ID)
    arms.remove_extra_arms()
    recs = [r for r in (data.get("arms") or []) if isinstance(r, dict)]
    for r in recs:
        base = r.get("arm_base") or {}
        r["id"] = arms.add_arm(
            name=r.get("name"), aid=r.get("id"), pos=base.get("pos"), wxyz=base.get("wxyz"),
            model=r.get("robot_type"), ip=r.get("robot_ip"), port=r.get("robot_port"),
            activate_it=False)
    out = _apply_doc_one(data)
    for r in recs:
        with arms.using(r["id"]):
            try:
                _apply_arm_points(r)
                _apply_arm_comps(r)
                _apply_arm_rest(r)
            except Exception as e:
                import traceback
                traceback.print_exc()
                print("load arm failed", r.get("name"), e)
    if data.get("active_arm") and arms.exists(data.get("active_arm")):
        arms.activate(data.get("active_arm"))
    return out


def _apply_arm_points(data):
    """One arm's base, waypoints / motion points and saved points (as that arm)."""
    # Before waypoints: IK values in the file are relative to the arm base.
    _restore_arm_base(data.get("arm_base"))
    fmt = str(data.get("format") or "").lower()
    bake.import_raw_json = fmt not in ("arcspath", "arcs")
    raw_pts = data.get("points") or []
    pts = normalize_point_list(raw_pts, lap_seconds=data.get("estimated_seconds_per_lap"))
    editor_wps = data.get("waypoints")
    print(
        "load doc format=%s points=%s editor_wps=%s"
        % (fmt or "?", len(pts), 0 if not editor_wps else len(editor_wps))
    )
    if editor_wps:
        wp.load_editor_waypoints(normalize_point_list(editor_wps))
        if hasattr(wp, "recover_wrapped_clips"):
            wp.recover_wrapped_clips()
        if not (getattr(wp, "waypoints", None) or []) and pts:
            wp.load_from_motion_points(pts, [])
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
    _load_sim_poses(data.get("sim_poses"))
    return pts


def _apply_doc_one(data):
    pts = _apply_arm_points(data)
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
    if isinstance(data.get("saved_paths"), list):
        wp.saved_paths = [dict(p) for p in data.get("saved_paths") if isinstance(p, dict)]
        seq = 0
        for p in wp.saved_paths:
            pid = str(p.get("id") or "")
            if pid.startswith("path_"):
                try:
                    seq = max(seq, int(pid.split("_")[-1]))
                except Exception:
                    pass
        wp._saved_path_seq = max(int(getattr(wp, "_saved_path_seq", 0) or 0), seq)
    try:
        import webui
        if hasattr(wp, "load_global_saved_paths"):
            wp.load_global_saved_paths((webui.load_settings() or {}).get("stored_paths") or [])
    except Exception:
        pass
    _load_scene_objects(data.get("objects"))
    _apply_arm_comps(data)
    # Set tracks only now: restoring objects re-syncs camera tracks, and a sync
    # that runs before the camera objects exist drops their cam_<id> keys.
    wp.extra_tracks = loaded_tracks
    if hasattr(wp, "sync_camera_tracks"):
        wp.sync_camera_tracks()
    if bake.import_raw_json:
        # Raw JSON times are off-grid: turn snap off for this session only.
        # Do not write it to settings.json, which is the user's preference.
        wp.snap_to_grid = False
    _apply_arm_rest(data)
    return {"ok": True, "waypoints": len(wp.waypoints), "points": len(pts)}


def _apply_arm_comps(data):
    """The arm's sim cursor components."""
    if data.get("sim_camera"):
        wp.sim_camera = dict(data.get("sim_camera") or {})
        if hasattr(wp, "refresh_sim_camera_viz"):
            try:
                wp.refresh_sim_camera_viz()
            except Exception:
                pass
    else:
        wp.sim_camera = None
    wp.sim_offset = wp.offsets.clean_offset(data.get("sim_offset"))
    wp.sim_collider = wp.colliders.clean_sim_collider(data.get("sim_collider"))
    wp.sim_comp_order = [str(k) for k in (data.get("sim_comp_order") or [])]
    wp.look_bars = [dict(b) for b in (data.get("look_bars") or []) if isinstance(b, dict)]


def _apply_arm_rest(data):
    """The arm's bake, cursor and Look At, after its waypoints and the objects."""
    if bake.import_raw_json:
        for w in getattr(wp, "waypoints", []) or []:
            w["force_fk"] = False
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
    if hasattr(bake, "bump_timeline_cache"):
        bake.bump_timeline_cache()
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
    # After objects and cursor: the Track target must exist and the pin is the cursor.
    _restore_sim_track(data.get("sim_track"))
    for oid in (arms.cid("sim_bot"), arms.cid("sim_cursor")):
        try:
            wp._propagate_children(oid)
        except Exception:
            pass


def _arm_base_record():
    """Aubo Arm (robot base) pose in the world."""
    try:
        pos, wxyz = rr.arm_base_pose()
    except Exception:
        return None
    return {"pos": [float(v) for v in pos], "wxyz": [float(v) for v in wxyz]}


def _restore_arm_base(rec):
    pos, wxyz = [0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]
    if isinstance(rec, dict):
        try:
            pos = [float(v) for v in list(rec.get("pos"))[:3]]
            wxyz = [float(v) for v in list(rec.get("wxyz"))[:4]]
        except Exception:
            pos, wxyz = [0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]
    try:
        if hasattr(wp, "set_sim_bot_pose"):
            wp.set_sim_bot_pose(pos, wxyz)
        else:
            rr.set_arm_base(pos, wxyz)
    except Exception as e:
        print("restore arm base failed:", e)


def _sim_track_record():
    """The sim cursor's Track component; waypoints keep their own track_object."""
    tr = getattr(rr, "_track", None) or {}
    if not tr.get("added") and not tr.get("on"):
        return None
    return {
        "on": bool(tr.get("on")),
        "name": str(tr.get("name") or ""),
        "added": bool(tr.get("added")),
        "offset": [float(v) for v in list(tr.get("offset") or [0.0, 0.0, 0.0])[:3]],
        "pos_offset": [float(v) for v in list(tr.get("pos_offset") or [0.0, 0.0, 0.0])[:3]],
        "tool": str(tr.get("tool") or ""),
    }


def _restore_sim_track(rec):
    try:
        if isinstance(rec, dict):
            rr.set_track_object(
                on=bool(rec.get("on")),
                name=str(rec.get("name") or ""),
                pick=False,
                added=bool(rec.get("added", True)),
                offset=rec.get("offset"),
                pos_offset=rec.get("pos_offset") or [0.0, 0.0, 0.0],
            )
            rr._track["tool"] = str(rec.get("tool") or "")
        else:
            rr.set_track_object(on=False, name="", pick=False, added=False, offset=[0.0, 0.0, 0.0], pos_offset=[0.0, 0.0, 0.0])
    except Exception as e:
        print("restore track failed:", e)


def pick_save_path(title=None, default=None):
    start = default or SESSION.get("path") or str(app_paths.ensure_dir(app_paths.PATHS_DIR) / ((title or "path") + ".arcspath"))
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        path = filedialog.asksaveasfilename(
            title="Save project",
            defaultextension=".arcspath",
            initialfile=Path(start).name,
            initialdir=str(Path(start).parent),
            filetypes=[("Arcs path", "*.arcspath"), ("JSON", "*.json"), ("All", "*.*")],
        )
        root.destroy()
        return path or None
    except Exception as e:
        print("save picker failed", e)
        return None


def pick_open_path():
    start = SESSION.get("path") or str(app_paths.ensure_dir(app_paths.PATHS_DIR))
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        path = filedialog.askopenfilename(
            title="Open project",
            initialdir=str(Path(start).parent if Path(start).suffix else start),
            filetypes=[("Arcs path", "*.arcspath"), ("JSON", "*.json"), ("All", "*.*")],
        )
        root.destroy()
        return path or None
    except Exception as e:
        print("open picker failed", e)
        return None


def note_opened(path, data=None):
    p = Path(path) if path else None
    SESSION["path"] = str(p) if p else None
    SESSION["title"] = _safe_stem((data or {}).get("title") if data else None, p.stem if p else "path")
    SESSION["last_save_ts"] = float((data or {}).get("last_save_ts") or (p.stat().st_mtime if p and p.exists() else 0.0))
    SESSION["last_autosave_ts"] = float((data or {}).get("last_autosave_ts") or 0.0)
    SESSION["last_hash"] = _doc_hash(data) if data else None
    SESSION["recovered"] = False
    SESSION["source"] = "open"
    return session_info()


def write_autosave():
    data = build_motion_doc(include_waypoint_flag=True)
    h = _doc_hash(data)
    if h == SESSION.get("last_hash"):
        return {"ok": True, "skipped": True, "last_autosave_ts": SESSION.get("last_autosave_ts")}
    man_ts = SESSION.get("last_save_ts")
    try:
        man_ts = 0.0 if man_ts is None else float(man_ts)
    except Exception:
        man_ts = 0.0
    prev_auto = float(SESSION.get("last_autosave_ts") or 0.0)
    if prev_auto > man_ts and prev_auto > 0:
        auto_ts = prev_auto
    else:
        auto_ts = time.time()
    data["title"] = SESSION.get("title") or data.get("title")
    data["last_save_ts"] = man_ts if man_ts > 0 else 0.0
    data["last_autosave_ts"] = auto_ts
    if SESSION.get("path"):
        data["source_path"] = SESSION["path"]
    dest = autosave_path()
    dest.write_text(json.dumps(data, indent=2), encoding="utf-8")
    SESSION["last_autosave_ts"] = auto_ts
    SESSION["last_hash"] = h
    print("autosave", dest, "save_ts", data["last_save_ts"], "auto_ts", auto_ts)
    return {"ok": True, "path": str(dest), "last_autosave_ts": auto_ts, "skipped": False}


def autosave_offer():
    dest = autosave_path()
    if not dest.exists():
        return {"offer": False}
    try:
        data = json.loads(dest.read_text(encoding="utf-8"))
    except Exception:
        return {"offer": False}
    if not _doc_has_work(data):
        return {"offer": False}
    auto_raw = data.get("last_autosave_ts")
    try:
        auto_ts = float(auto_raw) if auto_raw is not None else float(dest.stat().st_mtime)
    except Exception:
        auto_ts = float(dest.stat().st_mtime)
    man_raw = data.get("last_save_ts")
    try:
        man_ts = 0.0 if man_raw is None else float(man_raw)
    except Exception:
        man_ts = 0.0
    never_saved = man_ts <= 0.0
    if (not never_saved) and auto_ts <= man_ts + 0.5:
        return {"offer": False}
    return {
        "offer": True,
        "never_saved": never_saved,
        "auto_path": str(dest),
        "source_path": data.get("source_path") or SESSION.get("path"),
        "title": data.get("title") or "autosave",
        "last_autosave_ts": auto_ts,
        "last_save_ts": man_ts,
    }


def _read_json_file(path):
    p = Path(path)
    if not p.exists() or not p.is_file():
        raise FileNotFoundError("not a file: %s" % p)
    raw = p.read_bytes()
    if not raw.strip():
        raise ValueError("empty file (%d bytes): %s" % (len(raw), p))
    text = None
    for enc in ("utf-8-sig", "utf-8", "utf-16"):
        try:
            text = raw.decode(enc)
            break
        except Exception:
            continue
    if text is None:
        text = raw.decode("utf-8", errors="replace")
    text = text.strip()
    if not text:
        raise ValueError("empty file (%d bytes): %s" % (len(raw), p))
    return json.loads(text)


def load_from_path(path, recovered=False):
    p = Path(path)
    try:
        data = _read_json_file(p)
    except Exception as e:
        print("load_from_path failed", p, e)
        return {"ok": False, "error": str(e), "path": str(p), "waypoints": 0, "points": 0}
    if not isinstance(data, dict):
        return {"ok": False, "error": "file is not a project object", "path": str(p), "waypoints": 0, "points": 0}
    out = apply_loaded_doc(data)
    note_opened(data.get("source_path") if recovered else p, data)
    SESSION["recovered"] = bool(recovered)
    if recovered and data.get("source_path"):
        SESSION["path"] = str(data.get("source_path"))
    return {**out, "title": SESSION.get("title"), "path": SESSION.get("path")}


def new_project():
    # One full reset, on this thread only. /api/file/new used to also queue
    # reset_document on the viewer thread, so both cleared the same lists at
    # once and a swallowed error could leave objects in the scene.
    arms.activate(arms.DEFAULT_ID)
    arms.remove_extra_arms()
    _restore_arm_base(None)
    wp.sim_camera = None
    wp.sim_offset = None
    wp.sim_collider = None
    wp.sim_comp_order = []
    wp.look_bars = []
    if hasattr(wp, "reset_document"):
        wp.reset_document()
    else:
        if hasattr(wp, "clear_waypoints"):
            wp.clear_waypoints()
        if hasattr(wp, "clear_scene_objects"):
            wp.clear_scene_objects()
    try:
        import robot_runtime as rr
        if hasattr(rr, "set_track_object"):
            rr.set_track_object(on=False, name="", pick=False, added=False, offset=[0.0, 0.0, 0.0], pos_offset=[0.0, 0.0, 0.0])
    except Exception as e:
        print("new: track reset failed", e)
    if hasattr(wp, "refresh_sim_camera_viz"):
        try:
            wp.refresh_sim_camera_viz()
        except Exception:
            pass
    wp.extra_tracks = []
    if hasattr(bake, "invalidate_from"):
        bake.invalidate_from(0, drop_all=True)
    SESSION["path"] = None
    SESSION["title"] = "Path editor"
    SESSION["last_save_ts"] = 0.0
    SESSION["last_autosave_ts"] = 0.0
    SESSION["last_hash"] = None
    SESSION["recovered"] = False
    SESSION["source"] = "new"
    return {"ok": True}
