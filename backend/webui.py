# version: v_beta10.06
"""Parent Web UI. Native source only — do not exec a .pyc."""
from __future__ import annotations

import json
import logging
import math
import sys
import threading
import time
import traceback
import webbrowser
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
_PARENT = _ROOT.parent
_SIM = _ROOT / "robot_sim"
_CAMERAS = _ROOT / "cameras"
for _p in (_CAMERAS, _SIM, _ROOT, _PARENT):
    s = str(_p)
    if _p.is_dir() and s not in sys.path:
        sys.path.insert(0, s)

# Started as `python backend/webui.py`, this module is __main__. Register it
# as "webui" too, so `import webui` elsewhere (file open, New, settings)
# reuses the running module instead of loading and running webui.py again.
if __name__ == "__main__":
    sys.modules.setdefault("webui", sys.modules[__name__])

from flask import Flask, Response, jsonify, render_template, request, send_file
from werkzeug.serving import WSGIRequestHandler

import app_paths
import file_manage as fman
import path_bake as bake
import robot_runtime as rr
import robot_sim_run as viewer_main
import simulatorview as sv
import waypoints as wp

ROOT = Path(__file__).resolve().parent
SETTINGS_PATH = app_paths.SETTINGS_PATH
SETTINGS_DEFAULT = {
    "joint_vel_deg": [223.0, 223.0, 223.0, 267.0, 267.0, 267.0],
    "joint_acc_deg": [496.0, 496.0, 496.0, 594.0, 594.0, 594.0],
    "spacing": 0.05,
    "path_window_radius": 1.0,
    "path_window_on": False,
    "camera_urdf": "",
    "robot_model": "aubo_i5",
    "dev_mode": False,
    "hold_lin_mps": 0.05,
    "ik_pref": "backup",
    "robot_models": [
        {"id": "aubo_i5", "name": "aubo i5"},
        {"id": "aubo_i7", "name": "aubo i7"},
    ],
}

app = Flask(__name__, template_folder=str(ROOT / "templates"), static_folder=None)
logging.getLogger("werkzeug").setLevel(logging.ERROR)


class QuietHandler(WSGIRequestHandler):
    def log_request(self, code="-", size="-"):
        try:
            c = int(code)
        except Exception:
            c = 0
        if c >= 400:
            WSGIRequestHandler.log_request(self, code, size)


_status_lock = threading.Lock()
_status_heavy = {}
_status_worker_started = False
_viewer_thread = None
_console_lines = []
_console_i = 0


def _console_add(msg):
    global _console_i
    _console_i += 1
    _console_lines.append({"i": _console_i, "t": time.time(), "line": str(msg)})
    if len(_console_lines) > 400:
        del _console_lines[: len(_console_lines) - 400]


class _ConsoleStream:
    def __init__(self, raw):
        self.raw = raw
        self.buf = ""

    def write(self, s):
        try:
            self.raw.write(s)
        except Exception:
            pass
        self.buf += str(s)
        while "\n" in self.buf:
            line, self.buf = self.buf.split("\n", 1)
            if line.strip():
                _console_add(line)

    def flush(self):
        try:
            self.raw.flush()
        except Exception:
            pass


sys.stdout = _ConsoleStream(sys.stdout)
sys.stderr = _ConsoleStream(sys.stderr)
_console_add("console ready")


def load_settings():
    data = dict(SETTINGS_DEFAULT)
    try:
        if SETTINGS_PATH.is_file():
            raw = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                data.update(raw)
    except Exception as e:
        print("load settings failed", e)
    data["robot_model"] = str(data.get("robot_model") or "aubo_i5")
    return data


def save_settings(data):
    cur = load_settings()
    cur.update(data or {})
    app_paths.ensure_dir(SETTINGS_PATH.parent)
    SETTINGS_PATH.write_text(json.dumps(cur, indent=2), encoding="utf-8")
    return cur


def apply_settings(data):
    data = data or {}
    if "spacing" in data and hasattr(bake, "set_spacing"):
        try:
            bake.set_spacing(float(data["spacing"]))
        except Exception:
            pass
    if hasattr(rr, "set_joint_limits"):
        try:
            if rr.set_joint_limits(data.get("joint_vel_deg"), data.get("joint_acc_deg")):
                # New limits: re-score the bake (IK stays) so colours match them.
                bake.evaluate_timing(force=True)
                if hasattr(bake, "bump_timeline_cache"):
                    bake.bump_timeline_cache()
        except Exception as e:
            print("joint limits failed", e)
    if "hold_lin_mps" in data and hasattr(rr, "set_hold_lin_mps"):
        try:
            rr.set_hold_lin_mps(data.get("hold_lin_mps"))
        except Exception:
            pass
    if "ik_pref" in data and hasattr(rr, "set_ik_pref"):
        try:
            rr.set_ik_pref(data.get("ik_pref"))
        except Exception:
            pass
    if "snap_to_grid" in data:
        try:
            wp.snap_to_grid = bool(data.get("snap_to_grid"))
        except Exception:
            pass
    if "grid_spacing" in data:
        try:
            wp.grid_spacing = max(0.001, float(data.get("grid_spacing")))
        except Exception:
            pass
    if hasattr(wp, "load_global_saved_paths"):
        try:
            wp.load_global_saved_paths(data.get("stored_paths") or [])
        except Exception:
            pass
    mid = data.get("robot_model") or "aubo_i5"
    try:
        if hasattr(rr, "set_robot_model"):
            rr.set_robot_model(mid)
        if hasattr(rr, "apply_aubo_i5_joint_limits"):
            if getattr(rr, "robot_actual", None) is not None:
                rr.apply_aubo_i5_joint_limits(rr.robot_actual)
            if getattr(rr, "robot_sim", None) is not None:
                rr.apply_aubo_i5_joint_limits(rr.robot_sim)
    except Exception as e:
        print("apply robot model failed", e)


def start_viewer_thread():
    global _viewer_thread
    if _viewer_thread is not None and _viewer_thread.is_alive():
        return
    t = threading.Thread(target=viewer_main.start_viewer, daemon=True)
    t.start()
    _viewer_thread = t


def _edits_locked():
    try:
        sim = rr.sim_status() if hasattr(rr, "sim_status") else {}
        real = rr.real_path_status() if hasattr(rr, "real_path_status") else {}
        return bool(sim.get("active") or real.get("active"))
    except Exception:
        return False


def _scene_objects_payload():
    rows = []
    for obj in list(getattr(wp, "scene_objects", None) or []):
        if not isinstance(obj, dict):
            continue
        pos = obj.get("pos")
        wxyz = obj.get("wxyz")
        sc = obj.get("scale")
        rows.append({
            "id": obj.get("id"),
            "kind": obj.get("kind"),
            "name": obj.get("name"),
            "parent_id": obj.get("parent_id"),
            "pos": [float(v) for v in (pos if pos is not None else (0, 0, 0))][:3],
            "wxyz": [float(v) for v in (wxyz if wxyz is not None else (1, 0, 0, 0))][:4],
            "scale": [float(v) for v in (sc if sc is not None else (1, 1, 1))][:3],
            "space": obj.get("space") or "world",
            "mesh_path": obj.get("mesh_path"),
            "mesh_error": obj.get("mesh_error"),
            "camera": obj.get("camera"),
            "cam_type": str(((obj.get("camera") or {}).get("type") or "sim")),
            "has_frame": bool(obj.get("id") in getattr(sv, "_cam_frame_jpeg", {})),
            "cam_error": obj.get("cam_error") or ((obj.get("camera") or {}).get("error")),
        })
    return rows


def _cam_view_payload():
    try:
        if hasattr(rr, "camera_view_angles"):
            return rr.camera_view_angles()
    except Exception:
        pass
    return {}


def _sim_poses_payload():
    try:
        if hasattr(wp, "list_sim_poses"):
            return wp.list_sim_poses()
    except Exception:
        pass
    return list(getattr(wp, "sim_poses", None) or [])


def clipboard_info():
    try:
        if hasattr(wp, "clipboard_info"):
            return wp.clipboard_info()
    except Exception:
        pass
    return {"can_paste": False}


def _build_status_heavy():
    st = {}
    try:
        st = bake.status_dict() if hasattr(bake, "status_dict") else {}
    except Exception:
        traceback.print_exc()
    try:
        tl = bake.timeline_payload() if hasattr(bake, "timeline_payload") else {}
    except Exception:
        tl = {}
    st.update({
        "keyframes": tl.get("keyframes") or [
            {
                "index": i,
                "t": None if w.get("t") is None else float(w.get("t")),
                "status": "unbaked",
                "clip_id": w.get("clip_id"),
                "clip_name": w.get("clip_name"),
                "clip_locked": bool(w.get("clip_locked")),
            }
            for i, w in enumerate(getattr(wp, "waypoints", []) or [])
        ],
        "samples": tl.get("samples") or [],
        "spans": tl.get("spans") or [],
        "runnable": bool(tl.get("runnable") or st.get("runnable")),
        "sim_runnable": bool(tl.get("sim_runnable") or st.get("sim_runnable") or st.get("runnable")),
    })
    try:
        st["object"] = rr.get_selected_props() if hasattr(rr, "get_selected_props") else {}
    except Exception:
        st["object"] = {}
    try:
        st["move"] = rr.hold_move_status() if hasattr(rr, "hold_move_status") else {}
    except Exception:
        st["move"] = {}
    st["clipboard"] = clipboard_info()
    try:
        st["playhead_t"] = float(getattr(bake, "playhead_t", 0.0) or 0.0)
    except Exception:
        st["playhead_t"] = 0.0
    st["scene_objects"] = _scene_objects_payload()
    st["sim_camera"] = getattr(wp, "sim_camera", None)
    st["cam_view"] = _cam_view_payload()
    st["selected_obj_ids"] = list(getattr(wp, "selected_obj_ids", None) or [])
    st["selected_wp_idxs"] = list(getattr(wp, "selected_wp_idxs", None) or [])
    st["selected_pose_ids"] = list(getattr(wp, "selected_pose_ids", None) or [])
    try:
        if hasattr(wp, "measure_selected_objects"):
            st["measure"] = wp.measure_selected_objects()
    except Exception:
        pass
    st["pose_markers"] = _sim_poses_payload()
    st["saved_paths"] = list(getattr(wp, "saved_paths", None) or [])
    st["selected_path_id"] = getattr(wp, "selected_path_id", None)
    st["show_pose_markers"] = bool(getattr(wp, "show_pose_markers", True))
    st["tracks"] = list(getattr(wp, "extra_tracks", None) or [])
    return st


_status_last_req = 0.0
# Set by timeline edits so the next status build runs at once, not on the
# 0.25 s cadence; a stale build made dropped frames jump back.
_status_dirty = threading.Event()
# The page polls /api/status every 1.2 s; rebuilding the heavy payload
# (is_runnable, timeline, padded samples) 8x a second re-scored the same bake.
STATUS_REBUILD_S = 0.25


def _status_loop():
    global _status_heavy
    last_build = 0.0
    while True:
        try:
            if hasattr(wp, "flush_idle_path_rebuild"):
                wp.flush_idle_path_rebuild()
            if hasattr(wp, "snap_global_gizmos_idle"):
                wp.snap_global_gizmos_idle()
            if hasattr(wp, "tick_cameras"):
                wp.tick_cameras()
            now = time.time()
            polled = now - _status_last_req < 3.0
            dirty = _status_dirty.is_set()
            if dirty or ((polled or not _status_heavy) and now - last_build >= STATUS_REBUILD_S):
                _status_dirty.clear()
                heavy = _build_status_heavy()
                last_build = time.time()
                with _status_lock:
                    _status_heavy = heavy
        except Exception:
            traceback.print_exc()
        _status_dirty.wait(0.12)


def _ensure_status_worker():
    global _status_worker_started
    if _status_worker_started:
        return
    _status_worker_started = True
    threading.Thread(target=_status_loop, daemon=True).start()


@app.route("/")
def index():
    port = getattr(viewer_main, "VISER_PORT", None) or 8080
    return render_template("index.html", viser_url="http://127.0.0.1:%s" % port)


@app.route("/api/status")
def api_status():
    global _status_last_req
    _status_last_req = time.time()
    _ensure_status_worker()
    with _status_lock:
        heavy = dict(_status_heavy or {})
    bake_info = {}
    try:
        st = bake.status_dict() if hasattr(bake, "status_dict") else {}
        bake_info = st.get("bake") if isinstance(st.get("bake"), dict) else st
    except Exception:
        pass
    return jsonify({
        "ok": True,
        "viewer_ready": bool(getattr(viewer_main, "VISER_PORT", None)),
        "viser_port": getattr(viewer_main, "VISER_PORT", None),
        "waypoint_count": len(getattr(wp, "waypoints", []) or []),
        "selected": getattr(wp, "selected_idx", None),
        "selected_sample": getattr(bake, "selected_sample", None),
        "mode": wp.get_mode() if hasattr(wp, "get_mode") else "arc",
        "bake": bake_info,
        "keyframes": [] if not getattr(wp, "waypoints", None) else (heavy.get("keyframes") or []),
        "samples": [] if not getattr(wp, "waypoints", None) else (heavy.get("samples") or []),
        "spans": [] if not getattr(wp, "waypoints", None) else (heavy.get("spans") or []),
        "runnable": bool(heavy.get("runnable")),
        "sim_runnable": bool(heavy.get("sim_runnable") or heavy.get("runnable")),
        "sim": rr.sim_status() if hasattr(rr, "sim_status") else {},
        "real": rr.real_path_status() if hasattr(rr, "real_path_status") else {},
        "object": heavy.get("object") or {},
        "move": heavy.get("move") or {},
        "clipboard": heavy.get("clipboard") or {"can_paste": False},
        "playhead": heavy.get("playhead_t"),
        "scene_objects": _scene_objects_payload(),
        "sim_camera": getattr(wp, "sim_camera", None),
        "cam_view": heavy.get("cam_view") or {},
        "selected_obj_ids": heavy.get("selected_obj_ids") or [],
        "selected_wp_idxs": heavy.get("selected_wp_idxs") or [],
        "selected_pose_ids": heavy.get("selected_pose_ids") or [],
        "measure": heavy.get("measure"),
        "pose_markers": heavy.get("pose_markers") or [],
        "sim_poses": _sim_poses_payload(),
        "saved_paths": list(getattr(wp, "saved_paths", None) or []),
        "selected_path_id": getattr(wp, "selected_path_id", None),
        "show_pose_markers": bool(heavy.get("show_pose_markers", True)),
        "tracks": heavy.get("tracks") or list(getattr(wp, "extra_tracks", None) or []),
        "snap_to_grid": bool(getattr(wp, "snap_to_grid", False)),
        "item_vis": getattr(wp, "item_vis", {}) or {},
        "list_vis": getattr(wp, "list_vis", {}) or {},
        "merged_points": wp.list_merged_points() if hasattr(wp, "list_merged_points") else [],
        "gizmo_space": getattr(wp, "gizmo_space", "local"),
        "hide_robot": rr.hide_real_robot_status() if hasattr(rr, "hide_real_robot_status") else {},
        "show_viewer_samples": bool(getattr(bake, "show_viewer_samples", True)),
        "show_tl_samples": bool(getattr(bake, "show_tl_samples", True)),
    })


@app.route("/api/clock")
def api_clock():
    t = None
    try:
        if hasattr(rr, "playhead_now"):
            t = rr.playhead_now()
    except Exception:
        t = None
    # The speed graph plots these; without them both lines sat at 0.
    try:
        lin_sim = float(rr._motion["sim"]["lin_mps"] or 0.0)
        lin_real = float(rr._motion["real"]["lin_mps"] or 0.0)
    except Exception:
        lin_sim = lin_real = 0.0
    return jsonify({
        "ok": True,
        "t": t,
        "playhead": t,
        "lin_sim": lin_sim,
        "lin_real": lin_real,
        "sim": rr.sim_status() if hasattr(rr, "sim_status") else {},
        "real": rr.real_path_status() if hasattr(rr, "real_path_status") else {},
    })


@app.route("/api/settings", methods=["GET", "POST"])
def api_settings():
    if request.method == "GET":
        st = load_settings()
        if not st.get("robot_models"):
            st["robot_models"] = list(SETTINGS_DEFAULT.get("robot_models") or [])
        st["robot_model"] = getattr(rr, "_MODEL_ID", None) or st.get("robot_model") or "aubo_i5"
        return jsonify({"ok": True, "settings": st})
    data = request.get_json(silent=True) or {}
    st = save_settings(data)
    apply_settings(st)
    return jsonify({"ok": True, "settings": st})


@app.route("/api/import_mesh", methods=["POST"])
def api_import_mesh():
    dest_dir = app_paths.ensure_dir(app_paths.MESHES_DIR)
    path = None
    name = None
    if request.files.get("file"):
        f = request.files.get("file")
        if not f.filename:
            return jsonify({"ok": False, "error": "no file"}), 400
        raw_name = f.filename
        safe = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in raw_name)
        if not safe.lower().endswith((".stl", ".fbx")):
            return jsonify({"ok": False, "error": "need .stl or .fbx"}), 400
        dest = dest_dir / safe
        n = 1
        stem, suf = dest.stem, dest.suffix
        while dest.exists():
            dest = dest_dir / ("%s_%d%s" % (stem, n, suf))
            n += 1
        f.save(str(dest))
        path = str(dest)
        name = Path(raw_name).stem
    else:
        data = request.get_json(silent=True) or {}
        src = data.get("path") or data.get("mesh_path")
        if not src:
            return jsonify({"ok": False, "error": "no file"}), 400
        src_p = Path(str(src))
        if not src_p.is_file():
            return jsonify({"ok": False, "error": "path not found"}), 400
        dest = dest_dir / src_p.name
        if src_p.resolve() != dest.resolve():
            n = 1
            stem, suf = dest.stem, dest.suffix
            while dest.exists():
                dest = dest_dir / ("%s_%d%s" % (stem, n, suf))
                n += 1
            dest.write_bytes(src_p.read_bytes())
        path = str(dest)
        name = src_p.stem
    try:
        obj = wp.add_scene_object("mesh", mesh_path=path, name=name)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"ok": False, "error": str(e)}), 500
    if not obj:
        return jsonify({"ok": False, "error": "add failed"}), 500
    return jsonify({
        "ok": True,
        "id": obj.get("id"),
        "kind": obj.get("kind"),
        "name": obj.get("name"),
        "mesh_path": obj.get("mesh_path") or path,
        "error": obj.get("mesh_error"),
    })


@app.route("/api/add_object", methods=["POST"])
def api_add_object():
    data = request.get_json(silent=True) or {}
    obj = wp.add_scene_object(data.get("kind") or "empty")
    return jsonify({"ok": True, "id": (obj or {}).get("id")})


@app.route("/api/add_camera_component", methods=["POST"])
def api_add_camera_component():
    data = request.get_json(silent=True) or {}
    oid = data.get("id") or data.get("oid")
    kind = str(data.get("kind") or "")
    if kind == "sim" or oid == "sim":
        oid = "sim"
    rec = None
    if hasattr(wp, "add_camera_component"):
        rec = wp.add_camera_component(oid)
    return jsonify({"ok": rec is not None, "object": rec or {}})


@app.route("/api/object_props", methods=["POST"])
def api_object_props():
    data = request.get_json(silent=True) or {}
    if "ik" in data:
        viewer_main.prop_requests.append(("ik", data.get("ik")))
    if "q" in data:
        q = data.get("q")
        if data.get("units") == "deg":
            q = [float(v) * math.pi / 180.0 for v in q]
        viewer_main.prop_requests.append(("q", q))
    if any(k in data for k in ("pos", "wxyz", "scale", "space", "name", "camera", "remove_camera")):
        oid = data.get("id") or getattr(wp, "selected_obj_id", None)
        if data.get("kind") == "sim" or oid == "sim":
            if data.get("remove_camera"):
                wp.sim_camera = None
            elif data.get("camera") is not None:
                cam = dict(getattr(wp, "sim_camera", None) or {})
                cam.update(data.get("camera") or {})
                wp.sim_camera = cam
            if hasattr(wp, "sync_camera_tracks"):
                try:
                    wp.sync_camera_tracks()
                except Exception:
                    pass
            if hasattr(wp, "refresh_sim_camera_viz"):
                try:
                    wp.refresh_sim_camera_viz()
                except Exception:
                    pass
            if hasattr(wp, "_tick_sim_previews"):
                try:
                    wp._tick_sim_previews()
                except Exception:
                    pass
        elif hasattr(wp, "update_scene_object"):
            wp.update_scene_object(
                oid,
                name=data.get("name"),
                pos=data.get("pos"),
                wxyz=data.get("wxyz"),
                scale=data.get("scale"),
                space=data.get("space"),
                camera=data.get("camera"),
                remove_camera=bool(data.get("remove_camera")),
            )
    return jsonify({"ok": True})


@app.route("/api/select_object", methods=["POST"])
def api_select_object():
    data = request.get_json(silent=True) or {}
    if hasattr(wp, "select_scene_object"):
        wp.select_scene_object(data.get("id"), mode=data.get("mode") or "replace")
    _status_dirty.set()
    return jsonify({"ok": True})


@app.route("/api/select_waypoint", methods=["POST"])
def api_select_waypoint():
    data = request.get_json(silent=True) or {}
    if data.get("index") is None:
        # The page sends index null when the user clicks empty space.
        wp.deselect_all()
        _status_dirty.set()
        return jsonify({"ok": True})
    wp.select_waypoint(data.get("index"), mode=data.get("mode") or "replace")
    _status_dirty.set()
    return jsonify({"ok": True})


@app.route("/api/select_sample", methods=["POST"])
def api_select_sample():
    data = request.get_json(silent=True) or {}
    if hasattr(bake, "select_sample"):
        bake.select_sample(data.get("seg"), data.get("index"))
    return jsonify({"ok": True})


@app.route("/api/select_pose", methods=["POST"])
def api_select_pose():
    data = request.get_json(silent=True) or {}
    if hasattr(wp, "select_pose"):
        wp.select_pose(data.get("id"), mode=data.get("mode") or "replace")
    return jsonify({"ok": True})


@app.route("/api/select_mods", methods=["POST"])
def api_select_mods():
    data = request.get_json(silent=True) or {}
    if hasattr(wp, "set_select_mods"):
        wp.set_select_mods(ctrl=data.get("ctrl"), shift=data.get("shift"))
    return jsonify({"ok": True})


def _add_waypoint_from(src, data):
    data = data or {}
    replace = data.get("replace")
    t = data.get("t")
    pose = None
    q = None
    try:
        if src == "real":
            pose = rr.get_real_cursor_pose() if hasattr(rr, "get_real_cursor_pose") else None
            q = rr.current_joint_q("real") if hasattr(rr, "current_joint_q") else None
        else:
            pose = rr.get_sim_cursor_pose() if hasattr(rr, "get_sim_cursor_pose") else None
            q = rr.current_joint_q("sim") if hasattr(rr, "current_joint_q") else None
    except Exception as e:
        return {"ok": False, "error": str(e)}
    if pose is None:
        return {"ok": False, "error": "no cursor"}
    if isinstance(pose, (tuple, list)) and len(pose) == 2:
        pos, wxyz = pose
    else:
        pos, wxyz = pose, None
    out = wp.add_waypoint(pos, mode="arc", wxyz=wxyz, q=q, t=t, replace=replace)
    if isinstance(out, dict):
        if "ok" not in out:
            out["ok"] = not out.get("needs_confirm")
        out["added_from"] = src
        return out
    return {"ok": True, "added_from": src}


@app.route("/api/add_waypoint_sim", methods=["POST"])
def api_add_waypoint_sim():
    data = request.get_json(silent=True) or {}
    return jsonify(_add_waypoint_from("sim", data))


@app.route("/api/add_waypoint_real", methods=["POST"])
def api_add_waypoint_real():
    if hasattr(rr, "robot_connected") and not rr.robot_connected():
        return jsonify({"ok": False, "error": "robot offline"}), 409
    data = request.get_json(silent=True) or {}
    return jsonify(_add_waypoint_from("real", data))


@app.route("/api/keyframe_time", methods=["POST"])
def api_keyframe_time():
    if _edits_locked():
        return jsonify({"ok": False, "error": "locked during playback"}), 409
    data = request.get_json(silent=True) or {}
    released = bool(data.get("released"))
    stretch = bool(data.get("stretch"))
    moves = data.get("moves")
    if isinstance(moves, list):
        rows = wp.set_keyframe_times(
            [(int(m["index"]), float(m["t"])) for m in moves if "index" in m and "t" in m],
            released=released,
            stretch=stretch,
        )
        _status_dirty.set()
        return jsonify({"ok": True, "moves": rows})
    t = wp.set_keyframe_time(int(data.get("index")), float(data.get("t")), released=released)
    return jsonify({"ok": True, "t": t})


@app.route("/api/sample_time", methods=["POST"])
def api_sample_time():
    data = request.get_json(silent=True) or {}
    applied = []
    if hasattr(bake, "set_sample_times"):
        applied = bake.set_sample_times(data.get("moves") or []) or []
    _status_dirty.set()
    return jsonify({"ok": True, "moves": applied})


@app.route("/api/force_fk", methods=["POST"])
def api_force_fk():
    data = request.get_json(silent=True) or {}
    idx = data.get("index")
    on = wp.toggle_force_fk(idx)
    return jsonify({"ok": True, "force_fk": on, "index": idx})


@app.route("/api/undo", methods=["POST"])
def api_undo():
    viewer_main.file_requests.append(("undo", None))
    return jsonify({"ok": True})


@app.route("/api/redo", methods=["POST"])
def api_redo():
    viewer_main.file_requests.append(("redo", None))
    return jsonify({"ok": True})


@app.route("/api/bake", methods=["POST"])
def api_bake():
    data = request.get_json(silent=True) or {}
    viewer_main.bake_requests.append(data.get("spacing"))
    return jsonify({"ok": True})


@app.route("/api/clear_bake", methods=["POST"])
def api_clear_bake():
    if hasattr(bake, "clear_bake"):
        bake.clear_bake()
    return jsonify({"ok": True})


@app.route("/api/simulate", methods=["POST"])
def api_simulate():
    data = request.get_json(silent=True) or {}
    loops = data.get("loops", data.get("repeat", 0))
    delay = data.get("delay", 0.0)
    ok = False
    if hasattr(rr, "enter_simulation_mode"):
        ok = bool(rr.enter_simulation_mode(loops=loops, delay=delay))
    else:
        viewer_main.sim_requests.append(data.get("duration"))
        ok = True
    st = rr.sim_status() if hasattr(rr, "sim_status") else {}
    return jsonify({"ok": bool(ok), "sim": st})


@app.route("/api/sim/pause", methods=["POST"])
def api_sim_pause():
    paused = rr.toggle_sim_pause()
    return jsonify({"ok": True, "paused": bool(paused), "sim": rr.sim_status()})


@app.route("/api/real/upload", methods=["POST"])
def api_real_upload():
    data = request.get_json(silent=True) or {}
    ok = False
    if hasattr(rr, "upload_real_path"):
        ok = bool(rr.upload_real_path(
            # The UI sends "repeat" (same field Simulate reads).
            loops=data.get("loops", data.get("repeat")) or 0,
            delay=data.get("delay") or 0.0,
        ))
    elif hasattr(rr, "plan_joint_path"):
        ok = bool(rr.plan_joint_path())
    st = rr.real_path_status() if hasattr(rr, "real_path_status") else {}
    return jsonify({"ok": bool(ok), "real": st, "error": None if ok else (st.get("phase") or "upload not started")})


@app.route("/api/real/start", methods=["POST"])
def api_real_start():
    ok = False
    err = None
    if hasattr(rr, "fire_real_path"):
        try:
            ok = bool(rr.fire_real_path())
        except Exception as e:
            err = str(e)
            print("real start failed", e)
    else:
        viewer_main.real_requests.append(("start", None))
        ok = True
    st = rr.real_path_status() if hasattr(rr, "real_path_status") else {}
    if not ok and not err:
        if st.get("uploading"):
            err = "still uploading"
        elif not st.get("uploaded"):
            err = "not uploaded"
        elif not st.get("at_start"):
            err = "not at start pose"
        else:
            err = st.get("phase") or "not started"
    return jsonify({"ok": bool(ok), "real": st, "error": err})


@app.route("/api/real/pause", methods=["POST"])
def api_real_pause():
    viewer_main.real_requests.append(("pause", None))
    return jsonify({"ok": True})


@app.route("/api/real/stop", methods=["POST"])
def api_real_stop():
    # Stop here, not on the viewer queue: a busy viewer loop must not delay it.
    rr.stop_real_path()
    return jsonify({"ok": True, "real": rr.real_path_status()})


@app.route("/api/robot_model", methods=["POST"])
def api_robot_model():
    data = request.get_json(silent=True) or {}
    mid = str(data.get("id") or data.get("robot_model") or "").strip()
    if not mid:
        return jsonify({"ok": False, "error": "no model"}), 400
    n_wp = len(getattr(wp, "waypoints", None) or [])
    if n_wp > 0 and not data.get("confirm"):
        return jsonify({"ok": False, "needs_confirm": True, "waypoints": n_wp})
    if n_wp > 0 and hasattr(wp, "clear_waypoints"):
        wp.clear_waypoints()
    st = save_settings({"robot_model": mid})
    apply_settings(st)
    reloaded = False
    if hasattr(rr, "reload_robot_models"):
        try:
            reloaded = bool(rr.reload_robot_models())
        except Exception as e:
            print("reload robots failed", e)
    return jsonify({
        "ok": True,
        "robot_model": mid,
        "settings": st,
        "reloaded": reloaded,
        "waypoints_cleared": n_wp,
    })


@app.route("/api/snap_sim_to_real", methods=["POST"])
def api_snap_sim_to_real():
    if hasattr(rr, "robot_connected") and not rr.robot_connected():
        return jsonify({"ok": False, "error": "robot offline"}), 409
    ok = False
    if hasattr(rr, "reset_sim_to_real"):
        ok = bool(rr.reset_sim_to_real())
    return jsonify({"ok": bool(ok)})


@app.route("/api/halt", methods=["POST"])
def api_halt():
    rr.request_halt()
    return jsonify({"ok": True, "sim": rr.sim_status(), "real": rr.real_path_status()})


@app.route("/api/file/new", methods=["POST"])
def api_file_new():
    data = request.get_json(silent=True) or {}
    if not data.get("force") and not fman.scene_is_empty():
        return jsonify({"ok": False, "needs_confirm": True})
    out = fman.new_project()
    return jsonify({**out, "session": fman.session_info()})


@app.route("/api/file/open", methods=["POST"])
def api_file_open():
    data = request.get_json(silent=True) or {}
    try:
        if data.get("doc"):
            out = fman.apply_loaded_doc(data.get("doc"))
            fman.note_opened(data.get("path"), data.get("doc"))
            return jsonify({**out, "session": fman.session_info(), "title": fman.SESSION.get("title")})
        path = data.get("path") or fman.pick_open_path()
        if not path:
            return jsonify({"ok": False, "cancelled": True})
        out = fman.load_from_path(path)
        return jsonify({**out, "session": fman.session_info(), "title": fman.SESSION.get("title"), "path": path})
    except Exception as e:
        print("api_file_open failed", e)
        return jsonify({"ok": False, "error": str(e)})


@app.route("/api/file/save", methods=["POST"])
def api_file_save():
    data = request.get_json(silent=True) or {}
    out = fman.write_save_file(
        path=data.get("path"),
        title=data.get("title"),
        tracks=data.get("tracks"),
        save_as=bool(data.get("save_as")),
        pick=bool(data.get("pick") or data.get("save_as") or not fman.SESSION.get("path")),
    )
    return jsonify({**out, "session": fman.session_info()})


@app.route("/api/file/save_as", methods=["POST"])
def api_file_save_as():
    data = request.get_json(silent=True) or {}
    out = fman.write_save_file(
        path=None,
        title=data.get("title"),
        tracks=data.get("tracks"),
        save_as=True,
        pick=True,
    )
    return jsonify({**out, "session": fman.session_info()})


@app.route("/api/file/session")
def api_file_session():
    return jsonify({"ok": True, **fman.session_info(), "empty": fman.scene_is_empty(), "autosave": fman.autosave_offer()})


@app.route("/api/file/autosave", methods=["POST"])
def api_file_autosave():
    return jsonify(fman.write_autosave())


@app.route("/api/file/load_autosave", methods=["POST"])
def api_file_load_autosave():
    offer = fman.autosave_offer()
    if not offer.get("offer"):
        return jsonify({"ok": False, "error": "no autosave"})
    out = fman.load_from_path(offer.get("auto_path"), recovered=True)
    if hasattr(bake, "bump_timeline_cache"):
        bake.bump_timeline_cache()
    return jsonify({**out, "session": fman.session_info()})


@app.route("/api/file/export", methods=["POST"])
def api_file_export():
    data = request.get_json(silent=True) or {}
    out = fman.write_export_json(path=data.get("path"))
    return jsonify({"ok": True, "path": str(out) if out else None})


@app.route("/api/delete_selected", methods=["POST"])
def api_delete_selected():
    if hasattr(wp, "delete_selected"):
        wp.delete_selected()
    return jsonify({"ok": True})


@app.route("/api/copy", methods=["POST"])
def api_copy():
    data = request.get_json(silent=True) or {}
    sel = data.get("keyframes") or ([wp.selected_idx] if getattr(wp, "selected_idx", None) is not None else [])
    return jsonify(bake.copy_selection(sel) if hasattr(bake, "copy_selection") else {"ok": False})


@app.route("/api/cut", methods=["POST"])
def api_cut():
    data = request.get_json(silent=True) or {}
    sel = data.get("keyframes") or ([wp.selected_idx] if getattr(wp, "selected_idx", None) is not None else [])
    return jsonify(bake.cut_selection(sel) if hasattr(bake, "cut_selection") else {"ok": False})


@app.route("/api/paste", methods=["POST"])
def api_paste():
    data = request.get_json(silent=True) or {}
    t = data.get("t")
    if t is None:
        t = getattr(bake, "playhead_t", 0.0)
    if hasattr(bake, "paste_clipboard"):
        return jsonify(bake.paste_clipboard(t))
    return jsonify({"ok": False})


@app.route("/api/merge", methods=["POST"])
def api_merge():
    data = request.get_json(silent=True) or {}
    if hasattr(wp, "merge_keyframes"):
        return jsonify(wp.merge_keyframes(data.get("i"), data.get("j")))
    return jsonify({"ok": False})


@app.route("/api/tracks", methods=["GET", "POST"])
def api_tracks():
    if request.method == "POST":
        data = request.get_json(silent=True) or {}
        wp.extra_tracks = data.get("tracks") or []
    return jsonify({"ok": True, "tracks": list(getattr(wp, "extra_tracks", None) or [])})


@app.route("/api/transition", methods=["POST"])
def api_transition():
    data = request.get_json(silent=True) or {}
    if hasattr(bake, "set_transition"):
        bake.set_transition(data.get("index"), data.get("mode"), c1=data.get("c1"), c2=data.get("c2"))
    elif hasattr(bake, "apply_transition"):
        bake.apply_transition(data.get("index"), data.get("mode"), c1=data.get("c1"), c2=data.get("c2"))
    if hasattr(bake, "bump_timeline_cache"):
        bake.bump_timeline_cache()
    return jsonify({"ok": True})


@app.route("/api/playhead", methods=["POST"])
def api_playhead():
    data = request.get_json(silent=True) or {}
    t = data.get("t")
    if hasattr(bake, "set_playhead"):
        bake.set_playhead(t)
    if hasattr(rr, "seek_playhead"):
        rr.seek_playhead(t)
    return jsonify({"ok": True, "t": t})


@app.route("/api/gizmo_space", methods=["POST"])
def api_gizmo_space():
    data = request.get_json(silent=True) or {}
    if hasattr(wp, "apply_gizmo_space"):
        wp.apply_gizmo_space(data.get("space"))
    return jsonify({"ok": True})


@app.route("/api/dev_mode", methods=["POST"])
def api_dev_mode():
    data = request.get_json(silent=True) or {}
    st = save_settings({"dev_mode": bool(data.get("on"))})
    return jsonify({"ok": True, "dev_mode": st.get("dev_mode")})


@app.route("/api/list_vis", methods=["POST"])
def api_list_vis():
    data = request.get_json(silent=True) or {}
    if hasattr(wp, "set_list_vis"):
        wp.set_list_vis(data)
    return jsonify({"ok": True})


@app.route("/api/vis", methods=["POST"])
def api_vis():
    data = request.get_json(silent=True) or {}
    if hasattr(wp, "set_item_vis"):
        wp.set_item_vis(data.get("kind"), data.get("id"), data.get("on"))
    return jsonify({"ok": True})


@app.route("/api/view_samples", methods=["POST"])
def api_view_samples():
    data = request.get_json(silent=True) or {}
    if "viewer" in data:
        bake.show_viewer_samples = bool(data.get("viewer"))
        if hasattr(bake, "_draw_sample_dots"):
            bake._draw_sample_dots()
    if "timeline" in data:
        bake.show_tl_samples = bool(data.get("timeline"))
        if hasattr(bake, "bump_timeline_cache"):
            bake.bump_timeline_cache()
    return jsonify({
        "ok": True,
        "viewer": bool(getattr(bake, "show_viewer_samples", True)),
        "timeline": bool(getattr(bake, "show_tl_samples", True)),
    })


@app.route("/api/hide_robot", methods=["POST"])
def api_hide_robot():
    data = request.get_json(silent=True) or {}
    hide = data.get("hide")
    if hide is None:
        hide = data.get("hidden")
    hidden = None
    if hasattr(rr, "set_hide_real_robot"):
        hidden = rr.set_hide_real_robot(bool(hide))
    if hidden is None and hasattr(rr, "hide_real_robot_status"):
        hidden = (rr.hide_real_robot_status() or {}).get("hidden")
    return jsonify({"ok": True, "hidden": bool(hidden if hide is None else hide)})


@app.route("/api/move_hold", methods=["POST"])
def api_move_hold():
    data = request.get_json(silent=True) or {}
    viewer_main.move_requests.append(("start", data))
    return jsonify({"ok": True})


@app.route("/api/move_release", methods=["POST"])
def api_move_release():
    viewer_main.move_requests.append(("stop", None))
    return jsonify({"ok": True})


@app.route("/api/sim_go", methods=["POST"])
def api_sim_go():
    if hasattr(rr, "start_goto_path_start"):
        rr.start_goto_path_start()
    return jsonify({"ok": True})


@app.route("/api/point_go", methods=["POST"])
def api_point_go():
    data = request.get_json(silent=True) or {}
    if hasattr(wp, "go_to_point"):
        out = wp.go_to_point(data)
        if isinstance(out, dict):
            return jsonify(out)
    return jsonify({"ok": False, "error": "go_to_point missing"})


@app.route("/api/point_store", methods=["POST"])
def api_point_store():
    data = request.get_json(silent=True) or {}
    if hasattr(wp, "store_point"):
        wp.store_point(data)
    return jsonify({"ok": True})


@app.route("/api/point_rename", methods=["POST"])
def api_point_rename():
    data = request.get_json(silent=True) or {}
    kind = str(data.get("kind") or "")
    name = data.get("name")
    if kind == "wp" or data.get("index") is not None and kind != "pose":
        if hasattr(wp, "rename_waypoint"):
            wp.rename_waypoint(data.get("index"), name)
        return jsonify({"ok": True})
    if hasattr(wp, "rename_sim_pose"):
        wp.rename_sim_pose(data.get("id") or data.get("pid"), name)
    elif hasattr(wp, "rename_point"):
        wp.rename_point(data.get("id"), name)
    return jsonify({"ok": True})


@app.route("/api/point_order", methods=["POST"])
def api_point_order():
    data = request.get_json(silent=True) or {}
    if hasattr(wp, "set_point_order"):
        wp.set_point_order(data.get("order"))
    return jsonify({"ok": True})


@app.route("/api/reparent", methods=["POST"])
def api_reparent():
    data = request.get_json(silent=True) or {}
    if hasattr(wp, "reparent_object"):
        wp.reparent_object(data.get("id"), data.get("parent_id"))
    return jsonify({"ok": True})


@app.route("/api/pose_markers", methods=["POST"])
def api_pose_markers():
    data = request.get_json(silent=True) or {}
    wp.show_pose_markers = bool(data.get("on", True))
    return jsonify({"ok": True})


@app.route("/api/saved_paths", methods=["GET", "POST"])
def api_saved_paths():
    data = request.get_json(silent=True) or {}
    act = str(data.get("act") or "").lower()
    rec = None
    if request.method == "POST" and act in ("add", "capture"):
        rec = wp.capture_saved_path(data.get("name")) if hasattr(wp, "capture_saved_path") else None
        if rec is None:
            return jsonify({"ok": False, "error": "no waypoints to save", "paths": list(getattr(wp, "saved_paths", None) or [])})
    elif request.method == "POST" and act in ("delete", "remove"):
        if hasattr(wp, "delete_saved_path"):
            wp.delete_saved_path(data.get("id"))
    elif request.method == "POST" and act == "rename":
        if hasattr(wp, "rename_saved_path"):
            rec = wp.rename_saved_path(data.get("id"), data.get("name"))
    elif request.method == "POST" and act == "store":
        if hasattr(wp, "set_path_store"):
            rec = wp.set_path_store(data.get("id"), data.get("store"))
    elif request.method == "POST" and act == "place":
        if hasattr(wp, "place_saved_path"):
            rec = wp.place_saved_path(data.get("id"), data.get("t"), confirm=bool(data.get("confirm")))
            if hasattr(bake, "bump_timeline_cache"):
                bake.bump_timeline_cache()
            tl = bake.timeline_payload() if hasattr(bake, "timeline_payload") else {}
            return jsonify({"ok": True, "path": rec, "timeline": tl, "paths": list(getattr(wp, "saved_paths", None) or [])})
    elif request.method == "POST" and act == "unwrap":
        n = wp.unwrap_clip(data.get("clip_id")) if hasattr(wp, "unwrap_clip") else 0
        return jsonify({"ok": True, "n": n})
    elif request.method == "POST" and act == "select":
        wp.selected_path_id = data.get("id")
    return jsonify({"ok": True, "path": rec, "paths": list(getattr(wp, "saved_paths", None) or []), "selected": getattr(wp, "selected_path_id", None)})


@app.route("/api/sim_poses", methods=["GET", "POST"])
def api_sim_poses():
    data = request.get_json(silent=True) or {}
    act = str(data.get("act") or data.get("action") or "").lower()
    pose = None
    if request.method == "POST" and act in ("add", "store", "capture"):
        src = str(data.get("source") or data.get("from") or "sim")
        if hasattr(wp, "add_sim_pose"):
            pose = wp.add_sim_pose(name=data.get("name"), source=src)
            print("added saved point", None if not pose else pose.get("id"), "from", src)
        else:
            return jsonify({"ok": False, "error": "add_sim_pose missing", "poses": _sim_poses_payload()})
    elif request.method == "POST" and act in ("delete", "remove"):
        pid = data.get("id") or data.get("pid")
        if hasattr(wp, "delete_sim_pose"):
            wp.delete_sim_pose(pid)
    elif request.method == "POST" and act in ("rename",):
        if hasattr(wp, "rename_sim_pose"):
            wp.rename_sim_pose(data.get("id") or data.get("pid"), data.get("name"))
    return jsonify({"ok": True, "pose": pose, "poses": _sim_poses_payload()})


@app.route("/api/track_object", methods=["POST"])
def api_track_object():
    data = request.get_json(silent=True) or {}
    st = {}
    if hasattr(rr, "set_track_object"):
        st = rr.set_track_object(
            on=data.get("on"),
            name=data.get("name"),
            pick=data.get("pick"),
            added=data.get("added"),
            offset=data.get("offset"),
            use_offset=data.get("use_offset"),
        ) or {}
    return jsonify({"ok": True, "track": st})


@app.route("/api/offset", methods=["POST"])
def api_offset():
    """Offset component on a scene object or the sim cursor (owner "sim")."""
    data = request.get_json(silent=True) or {}
    owner = data.get("id") or data.get("owner")
    if data.get("kind") == "sim" or owner in ("sim", "sim_cursor"):
        owner = "sim"
    if not owner:
        owner = getattr(wp, "selected_obj_id", None)
    if not owner:
        return jsonify({"ok": False, "error": "no owner"})
    act = str(data.get("act") or "update").lower()
    if act == "add":
        st = wp.offsets.add_offset_component(owner)
    else:
        st = wp.offsets.update_offset(
            owner,
            enabled=data.get("enabled"),
            pos=data.get("pos"),
            wxyz=data.get("wxyz"),
            remove=(act == "remove"),
        )
    if owner == "sim" and hasattr(rr, "set_track_object"):
        # Look At in offset mode re-aims when the offset changes.
        rr.set_track_object()
    _status_dirty.set()
    return jsonify({"ok": st is not None, "offset": st or {}})


@app.route("/api/path_window", methods=["POST"])
def api_path_window():
    data = request.get_json(silent=True) or {}
    on = bool(data.get("on"))
    radius = data.get("radius")
    try:
        radius = max(0.0, float(radius))
    except (TypeError, ValueError):
        radius = 1.0
    save_settings({"path_window_on": on, "path_window_radius": radius})
    # Apply to the 3D path too; saving the setting alone never clipped it.
    if hasattr(wp, "configure_path_window"):
        t = data.get("t")
        try:
            t = float(t) if t is not None else None
        except (TypeError, ValueError):
            t = None
        wp.configure_path_window(on, radius=radius, t=t)
    return jsonify({"ok": True})


@app.route("/api/camera", methods=["POST"])
def api_camera():
    data = request.get_json(silent=True) or {}
    if data.get("preview_id") is not None and hasattr(wp, "set_preview_cam"):
        wp.set_preview_cam(data.get("preview_id"))
        return jsonify({"ok": True})
    if data.get("view") is not None:
        if hasattr(rr, "apply_camera_view"):
            return jsonify(rr.apply_camera_view(data.get("view")))
        return jsonify({"ok": False, "error": "no viewer"})
    if hasattr(wp, "set_camera_props"):
        wp.set_camera_props(data)
    elif hasattr(wp, "update_scene_object"):
        wp.update_scene_object(data)
    return jsonify({"ok": True})


@app.route("/api/camera_state")
def api_camera_state():
    if hasattr(wp, "tick_cameras"):
        try:
            wp.tick_cameras()
        except Exception:
            pass
    st = wp.camera_status() if hasattr(wp, "camera_status") else {"active": False}
    if not isinstance(st, dict):
        st = {"state": st}
    try:
        if hasattr(rr, "camera_view_angles"):
            st.update(rr.camera_view_angles() or {})
    except Exception:
        pass
    return jsonify({"ok": True, **st})


@app.route("/api/cam_script", methods=["POST"])
def api_cam_script():
    dest_dir = app_paths.ensure_dir(app_paths.CAM_SCRIPTS_DIR)
    f = request.files.get("file")
    oid = request.form.get("id") or (request.get_json(silent=True) or {}).get("id")
    if not f or not f.filename:
        return jsonify({"ok": False, "error": "no file"}), 400
    safe = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in f.filename)
    dest = dest_dir / safe
    n = 1
    stem, suf = dest.stem, dest.suffix
    while dest.exists():
        dest = dest_dir / ("%s_%d%s" % (stem, n, suf))
        n += 1
    f.save(str(dest))
    path = str(dest)
    if oid:
        for obj in list(getattr(wp, "scene_objects", None) or []):
            if obj.get("id") != oid:
                continue
            cam = obj.get("camera")
            if not isinstance(cam, dict):
                cam = {}
                obj["camera"] = cam
            cam["script"] = path
            break
    return jsonify({"ok": True, "path": path})


@app.route("/api/cam_click", methods=["POST"])
def api_cam_click():
    data = request.get_json(silent=True) or {}
    ok = False
    if hasattr(wp, "set_camera_click"):
        ok = bool(wp.set_camera_click(data.get("id"), data.get("u"), data.get("v")))
    return jsonify({"ok": ok})


@app.route("/api/cam_frame")
def api_cam_frame():
    oid = request.args.get("id")
    blob = None
    store = getattr(sv, "_cam_frame_jpeg", None)
    if isinstance(store, dict):
        if oid:
            blob = store.get(oid)
        elif store:
            blob = next(iter(store.values()))
    if not blob:
        return Response(status=204)
    return Response(blob, mimetype="image/jpeg")


@app.route("/api/console")
def api_console():
    after = request.args.get("after", default=0, type=int) or 0
    rows = [r for r in _console_lines if r["i"] > after]
    return jsonify({"ok": True, "lines": rows, "after": _console_i})


@app.route("/api/custom_curve")
def api_custom_curve():
    idx = request.args.get("index", type=int)
    if hasattr(bake, "get_custom_curve"):
        return jsonify({"ok": True, "curve": bake.get_custom_curve(idx)})
    return jsonify({"ok": True})


@app.route("/api/optomise", methods=["POST"])
def api_optomise():
    return jsonify({"ok": True})


@app.route("/api/speed_export")
def api_speed_export():
    # write_overspeed_report writes to the same EXPORTS_DIR file.
    path = app_paths.EXPORTS_DIR / "bake_overspeed.csv"
    if request.args.get("download"):
        if path.is_file():
            return send_file(str(path), as_attachment=True)
        return jsonify({"ok": False, "error": "no overspeed report yet; bake first"}), 404
    res = {"ok": True}
    if hasattr(bake, "export_overspeed"):
        res = bake.export_overspeed() or res
    return jsonify(res)


def _open_ui():
    time.sleep(2)
    try:
        webbrowser.open("http://127.0.0.1:5000")
    except Exception:
        pass


def _autosave_loop():
    time.sleep(60)
    while True:
        try:
            fman.write_autosave()
        except Exception as e:
            print("autosave failed", e)
        time.sleep(60)


def _wait_viewer(timeout=20.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if getattr(viewer_main, "VISER_PORT", None):
            return int(viewer_main.VISER_PORT)
        time.sleep(0.05)
    return 8080


if __name__ == "__main__":
    apply_settings(load_settings())
    start_viewer_thread()
    port = _wait_viewer()
    print("Viser    http://127.0.0.1:%s (iframe)" % port)
    _ensure_status_worker()
    print("Web UI   http://127.0.0.1:5000")
    threading.Thread(target=_open_ui, daemon=True).start()
    threading.Thread(target=_autosave_loop, daemon=True).start()
    app.run(host="127.0.0.1", port=5000, debug=False, use_reloader=False, threaded=True, request_handler=QuietHandler)
