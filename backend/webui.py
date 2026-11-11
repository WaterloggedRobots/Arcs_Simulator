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
import arms
import file_manage as fman
import path_bake as bake
import real_mirror
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
    "real_ghosts": True,
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

# Every request runs as the active Aubo Arm, one at a time with the viewer
# loop and the viewer's click / gizmo callbacks (see arms.py).
_LOCK_FREE = {"/api/cam_frame", "/api/console"}
_lock_held = threading.local()


@app.before_request
def _take_arm_lock():
    if request.path in _LOCK_FREE:
        return
    arms.LOCK.acquire()
    _lock_held.n = getattr(_lock_held, "n", 0) + 1
    arms.ensure_active_loaded()


@app.teardown_request
def _drop_arm_lock(_exc=None):
    if request.path in _LOCK_FREE or not getattr(_lock_held, "n", 0):
        return
    _lock_held.n -= 1
    arms.LOCK.release()


def _unlocked(fn):
    """A file dialog waits on the user: let the viewer run meanwhile."""
    def run(*a, **k):
        held = getattr(_lock_held, "n", 0)
        for _ in range(held):
            arms.LOCK.release()
        try:
            return fn(*a, **k)
        finally:
            for _ in range(held):
                arms.LOCK.acquire()
            arms.ensure_active_loaded()
    return run


fman.pick_save_path = _unlocked(fman.pick_save_path)
fman.pick_open_path = _unlocked(fman.pick_open_path)


def _each_arm(fn):
    """fn() run as every arm in turn: [(arm id, result)]."""
    out = []
    for aid in arms.each():
        with arms.using(aid):
            out.append((aid, fn()))
    return out


def _arm_has_path():
    return len(getattr(wp, "waypoints", None) or []) >= 2


def _merged_status(fn):
    """Playback status for the page: the active arm's, unless only another
    arm is playing (an arm with no path still sees the others play)."""
    rows = _each_arm(fn)
    mine = dict(next((st for aid, st in rows if aid == arms.active()), {}) or {})
    if mine.get("active"):
        return mine
    for _aid, st in rows:
        if (st or {}).get("active"):
            return dict(st)
    return mine


def _display_tracks():
    """The tracks for the page: a gimbal track's frames as played (a "real pose
    at run" frame with the angles it plays: gimbal.keys_for), keeping its flags.
    The page never edits gimbal frames through /api/tracks (the server keeps its own)."""
    out = []
    for tr in list(getattr(wp, "extra_tracks", None) or []):
        tid = str((tr or {}).get("id") or "") if isinstance(tr, dict) else ""
        if tid.startswith(wp.gimbal.TRACK_PREFIX):
            tr = dict(tr, keys=wp.gimbal.keys_for(tid[len(wp.gimbal.TRACK_PREFIX):]))
        out.append(tr)
    return out


def _with_solo(st, kind):
    """No arm playing: the gimbal-only playback of that kind, if any (gimbal.solo_status)."""
    if st.get("active") or getattr(wp, "gimbal", None) is None:
        return st
    solo = wp.gimbal.solo_status(kind)
    return {**st, **solo} if solo else st


def _sim_status_all():
    return _with_solo(_merged_status(rr.sim_status), "sim")


def _real_status_all():
    return _with_solo(_merged_status(rr.real_path_status), "real")
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


def _apply_joint_limits(vel, acc):
    """Speed / accel limits of the arm being run as (each arm has its own)."""
    try:
        if rr.set_joint_limits(vel, acc):
            # New limits: re-score the bake (IK stays) so colours match them.
            bake.evaluate_timing(force=True)
            if hasattr(bake, "bump_timeline_cache"):
                bake.bump_timeline_cache()
    except Exception as e:
        print("joint limits failed", e)


def _arm_settings(st):
    """Overlay the active arm's own robot settings on settings.json's."""
    st = dict(st)
    st["robot_model"] = getattr(rr, "_MODEL_ID", None) or st.get("robot_model") or "aubo_i5"
    st["joint_vel_deg"], st["joint_acc_deg"] = rr.joint_limits()
    try:
        st["robot_address"] = rr.robot_address()
    except Exception:
        pass
    st["arms"] = arms.listing()
    st["active_arm"] = arms.active()
    return st


def apply_settings(data):
    data = data or {}
    if "spacing" in data and hasattr(bake, "set_spacing"):
        try:
            bake.set_spacing(float(data["spacing"]))
        except Exception:
            pass
    if arms.is_default():
        # settings.json holds the default arm's limits; added arms keep their own.
        _apply_joint_limits(data.get("joint_vel_deg"), data.get("joint_acc_deg"))
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
    if "real_ghosts" in data:
        real_mirror.set_ghosts(data.get("real_ghosts") is not False)
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
    if not arms.is_default():
        # settings.json holds the default arm's robot; added arms keep theirs
        # (api_settings applies those, and project files save them).
        return
    if "robot_ip" in data or "robot_port" in data:
        try:
            rr.set_robot_address(data.get("robot_ip"), data.get("robot_port"))
        except Exception as e:
            print("apply robot address failed", e)
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


def _solo_playable():
    """Gimbal frames to play while no arm has a path (gimbal-only timeline)."""
    try:
        # Like an arm's path: only once baked (the bake runs the gimbal speed check).
        return ((not any(r for _aid, r in _each_arm(_arm_has_path))) and wp.gimbal.solo_playable()
                and wp.gimbal.tracks_baked())
    except Exception:
        return False


def _edits_locked():
    try:
        sim = rr.sim_status() if hasattr(rr, "sim_status") else {}
        real = rr.real_path_status() if hasattr(rr, "real_path_status") else {}
        return bool(sim.get("active") or real.get("active") or wp.gimbal.solo_active())
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
            "gimbal_real": wp.gimbal.real_ok(obj.get("id")) if obj.get("kind") == "gimbal" else None,
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
        "runnable": bool((tl.get("runnable") or st.get("runnable") or _solo_playable()) and wp.gimbal.tracks_ok()),
        "sim_runnable": bool(tl.get("sim_runnable") or st.get("sim_runnable") or st.get("runnable") or _solo_playable()),
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
    st["look_bars"] = list(getattr(wp, "look_bars", None) or [])
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
    st["tracks"] = _display_tracks()
    try:
        st["arms"] = arms.listing()
        st["arm_rows"] = _arm_rows()
    except Exception:
        traceback.print_exc()
    return st


def _arm_rows():
    """Timeline rows of the arms not being edited: their keyframes and
    spans, drawn read-only (clicking a row makes that arm active)."""
    rows = []
    for aid in arms.each():
        if aid == arms.active():
            rows.append({"id": aid, "name": arms.name_of(aid), "active": True})
            continue
        with arms.using(aid):
            try:
                tl = bake.timeline_payload() if hasattr(bake, "timeline_payload") else {}
            except Exception:
                tl = {}
            kfs = tl.get("keyframes") or [
                {"index": i, "t": None if w.get("t") is None else float(w.get("t")), "status": "unbaked"}
                for i, w in enumerate(getattr(wp, "waypoints", []) or [])
            ]
            rows.append({
                "id": aid,
                "name": arms.name_of(aid),
                "active": False,
                "keyframes": [{"index": k.get("index"), "t": k.get("t"), "status": k.get("status")} for k in kfs],
                "spans": [{"t0": sp.get("t0"), "t1": sp.get("t1"), "status": sp.get("status")} for sp in (tl.get("spans") or [])],
            })
    return rows


def _arm_wps():
    """Every arm's waypoint rows for the shared Waypoints panel. Each arm keeps
    its own waypoints; the page groups them under the arm's name."""
    out = []
    for aid in arms.each():
        with arms.using(aid):
            try:
                rows = [r for r in wp.list_merged_points() if r.get("kind") == "wp"]
            except Exception:
                rows = []
            vis = wp.item_vis_view() if hasattr(wp, "item_vis_view") else {}
            for r in rows:
                r["visible"] = vis.get("wp:%s" % r.get("index")) is not False
            out.append({"id": aid, "name": arms.name_of(aid), "active": aid == arms.active(), "rows": rows})
    return out


_status_last_req = 0.0
# Set by timeline edits so the next status build runs at once, not on the
# 0.25 s cadence; a stale build made dropped frames jump back.
_status_dirty = threading.Event()
# The page polls /api/status every 1.2 s; rebuilding the heavy payload
# (is_runnable, timeline, padded samples) 8x a second re-scored the same bake.
STATUS_REBUILD_S = 0.25


def _status_loop():
    last_build = 0.0
    while True:
        try:
            with arms.LOCK:
                arms.ensure_active_loaded()
                last_build = _status_pass(last_build)
        except Exception:
            traceback.print_exc()
        _status_dirty.wait(0.12)


def _status_pass(last_build):
    """One status-worker pass, under arms.LOCK. Returns the build time."""
    global _status_heavy
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
    return last_build


def _ensure_status_worker():
    global _status_worker_started
    if _status_worker_started:
        return
    _status_worker_started = True
    threading.Thread(target=_status_loop, daemon=True).start()


@app.route("/")
def index():
    port = getattr(viewer_main, "VISER_PORT", None) or 8080
    base = "http://127.0.0.1:%s/" % port
    # The tags let the server tell the world view from camera panes (robot_runtime._world_ids).
    return render_template("index.html", viser_url=base + rr.view_tag_query("world"),
                           viser_cam_url=base + rr.view_tag_query("camera"))


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
        "runnable": bool((heavy.get("runnable") or _solo_playable()) and wp.gimbal.tracks_ok()),
        "sim_runnable": bool(heavy.get("sim_runnable") or heavy.get("runnable") or _solo_playable()),
        # Gimbal-only timeline with a real gimbal reading: Upload / Run work without the arm.
        "gimbal_only_real": bool(_solo_playable() and wp.gimbal.tracks_ok() and any(
            wp.gimbal.real_ok(o.get("id")) for o in wp.gimbal.gimbal_objects())),
        "sim": _sim_status_all(),
        "real": _real_status_all(),
        "gimbal_speed": wp.gimbal.speed_status(),
        "real_side": real_mirror.view(),
        "gimbal_speed_note": wp.gimbal.speed_summary(),
        "arms": heavy.get("arms") or [],
        "active_arm": arms.active(),
        "arm_rows": heavy.get("arm_rows") or [],
        "object": heavy.get("object") or {},
        "move": heavy.get("move") or {},
        "clipboard": heavy.get("clipboard") or {"can_paste": False},
        "playhead": heavy.get("playhead_t"),
        "scene_objects": _scene_objects_payload(),
        "sim_camera": getattr(wp, "sim_camera", None),
        "look_bars": list(getattr(wp, "look_bars", None) or []),
        "doc_title": fman.SESSION.get("title"),
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
        "tracks": heavy.get("tracks") or _display_tracks(),
        "snap_to_grid": bool(getattr(wp, "snap_to_grid", False)),
        "item_vis": wp.item_vis_view() if hasattr(wp, "item_vis_view") else (getattr(wp, "item_vis", {}) or {}),
        "list_vis": getattr(wp, "list_vis", {}) or {},
        "merged_points": wp.list_merged_points() if hasattr(wp, "list_merged_points") else [],
        "arm_wps": _arm_wps(),
        "tracked_points": wp.cam_target_list() if hasattr(wp, "cam_target_list") else [],
        "arm_move_pending": bool(wp.arm_move_pending()) if hasattr(wp, "arm_move_pending") else False,
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
    for kind in ("sim", "real"):
        solo = wp.gimbal.solo_status(kind) if getattr(wp, "gimbal", None) else None
        if solo and solo.get("active"):
            t = solo["t"]
    return jsonify({
        "ok": True,
        "t": t,
        "playhead": t,
        "lin_sim": lin_sim,
        "lin_real": lin_real,
        "collision": wp.colliders.collision_status() if hasattr(wp, "colliders") else {},
        "sim": _sim_status_all(),
        "real": _real_status_all(),
    })


@app.route("/api/settings", methods=["GET", "POST"])
def api_settings():
    if request.method == "GET":
        st = load_settings()
        if not st.get("robot_models"):
            st["robot_models"] = list(SETTINGS_DEFAULT.get("robot_models") or [])
        return jsonify({"ok": True, "settings": _arm_settings(st)})
    data = request.get_json(silent=True) or {}
    if not arms.is_default():
        # Robot Settings of an added arm: its own address, not settings.json's.
        robot_keys = {k: data.pop(k) for k in ("robot_ip", "robot_port", "robot_model") if k in data}
        if "robot_ip" in robot_keys or "robot_port" in robot_keys:
            rr.set_robot_address(robot_keys.get("robot_ip", rr._robot_addr.get("ip")),
                                 robot_keys.get("robot_port", rr._robot_addr.get("port")))
        # Its speed / accel limits too.
        _apply_joint_limits(data.pop("joint_vel_deg", None), data.pop("joint_acc_deg", None))
    st = save_settings(data)
    apply_settings(st)
    return jsonify({"ok": True, "settings": _arm_settings(st)})


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
    if data.get("kind") == "aubo_arm":
        aid = arms.add_arm()
        wp.select_scene_object(arms.cid("sim_bot", aid))
        _status_dirty.set()
        return jsonify({"ok": True, "id": arms.cid("sim_bot", aid), "arm": aid})
    obj = wp.add_scene_object(data.get("kind") or "empty")
    return jsonify({"ok": True, "id": (obj or {}).get("id")})


@app.route("/api/arm", methods=["POST"])
def api_arm():
    """Aubo Arms: {action: activate|remove|rename, id, name}."""
    data = request.get_json(silent=True) or {}
    act = str(data.get("action") or "activate")
    aid = str(data.get("id") or "")
    if not arms.exists(aid):
        return jsonify({"ok": False, "error": "no such arm"}), 404
    if act == "activate":
        if arms.activate(aid):
            # Show the arm picked in Robot Settings like a click on its cursor.
            wp.select_scene_object(arms.cid("sim_cursor", aid))
        ok = True
    elif act == "remove":
        if rr.edits_locked():
            return jsonify({"ok": False, "error": "stop playback first"}), 409
        ok = arms.remove_arm(aid)
    elif act == "rename":
        ok = arms.rename(aid, data.get("name"))
    else:
        return jsonify({"ok": False, "error": "unknown action"}), 400
    _status_dirty.set()
    return jsonify({"ok": bool(ok), "arms": arms.listing(), "active_arm": arms.active()})


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
    if data.get("snap_parent_offset") and hasattr(wp, "snap_to_parent_offset"):
        ok = wp.snap_to_parent_offset(data.get("id") or getattr(wp, "selected_obj_id", None))
        _status_dirty.set()
        return jsonify({"ok": bool(ok)})
    if any(k in data for k in ("pos", "wxyz", "scale", "space", "name", "camera", "remove_camera")):
        oid = data.get("id") or getattr(wp, "selected_obj_id", None)
        base, aid = arms.split_id(oid)
        if data.get("kind") == "bot" or base == "sim_bot":
            # Aubo Arm: its name and pose are editable (the pose moves both arms).
            with arms.using(aid or arms.active()):
                if data.get("name") is not None:
                    arms.rename(arms.current(), data.get("name"))
                if (data.get("pos") is not None or data.get("wxyz") is not None) and hasattr(wp, "set_sim_bot_pose"):
                    wp._push_history()
                    wp.set_sim_bot_pose(data.get("pos"), data.get("wxyz"), ask=True)
        elif data.get("kind") == "sim" or oid == "sim":
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
            # Properties edits are undo steps too (gizmo drags already are).
            wp._push_history()
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
    if isinstance(out, dict) and out.get("index") is not None:
        # Keep the cursor's Look At on the new (or replaced) waypoint; the UI
        # adds waypoints here, so without this no waypoint was ever aimed.
        i = int(out["index"])
        if 0 <= i < len(wp.waypoints):
            rec = rr.waypoint_track_record() if (src == "sim" and hasattr(rr, "waypoint_track_record")) else None
            wp.waypoints[i]["track_object"] = rec
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
    # Every arm with a path plays it, on one clock.
    res = _each_arm(lambda: bool(_arm_has_path() and rr.enter_simulation_mode(loops=loops, delay=delay)))
    ok = any(r for _aid, r in res)
    if not ok and _solo_playable():
        # Gimbal-only timeline: play the gimbal frames on their own clock.
        ok = wp.gimbal.solo_sim_start(loops, delay, restore=getattr(viewer_main, "playhead_t", 0.0))
    return jsonify({"ok": bool(ok), "sim": _sim_status_all()})


@app.route("/api/sim/pause", methods=["POST"])
def api_sim_pause():
    res = _each_arm(lambda: rr.toggle_sim_pause() if rr._pb.get("active") else None)
    paused = any(r for _aid, r in res if r is not None)
    if (wp.gimbal.solo_status("sim") or {}).get("active"):
        paused = bool(wp.gimbal.solo_pause())
    return jsonify({"ok": True, "paused": bool(paused), "sim": _sim_status_all()})


@app.route("/api/real/upload", methods=["POST"])
def api_real_upload():
    data = request.get_json(silent=True) or {}
    if not wp.gimbal.tracks_ok():
        # Gimbal frames changed since the bake, or a segment is too fast for the gimbal.
        why = wp.gimbal.speed_summary() if wp.gimbal.tracks_baked() else "bake first (the gimbal frames changed)"
        return jsonify({"ok": False, "real": _real_status_all(), "error": why})
    def _up():
        if rr.robot is None or not _arm_has_path():
            return False
        return bool(rr.upload_real_path(
            # The UI sends "repeat" (same field Simulate reads).
            loops=data.get("loops", data.get("repeat")) or 0,
            delay=data.get("delay") or 0.0,
        ))
    ok = any(r for _aid, r in _each_arm(_up))
    if not ok and _solo_playable():
        # Gimbal-only timeline: "upload" sends the gimbals to their initial pose.
        ok = wp.gimbal.solo_upload(data.get("loops", data.get("repeat")) or 0, data.get("delay") or 0.0)
        if not ok:
            return jsonify({"ok": False, "real": _real_status_all(), "error": "the real gimbal is not reading"})
    st = _real_status_all()
    return jsonify({"ok": bool(ok), "real": st, "error": None if ok else (st.get("phase") or "upload not started")})


@app.route("/api/real/start", methods=["POST"])
def api_real_start():
    ok = False
    err = None
    solo = wp.gimbal.solo_status("real")
    if solo is not None:
        ok = wp.gimbal.solo_real_start()
        return jsonify({"ok": bool(ok), "real": _real_status_all(),
                        "error": None if ok else (solo.get("phase") or "not uploaded")})

    def _fire():
        if rr.robot is None or not (rr._rp.get("uploaded") or rr._rp.get("uploading")):
            return None
        try:
            return bool(rr.fire_real_path())
        except Exception as e:
            print("real start failed", arms.name_of(), e)
            return str(e)
    for _aid, r in _each_arm(_fire):
        if r is True:
            ok = True
        elif isinstance(r, str):
            err = r
    st = _real_status_all()
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
    if (wp.gimbal.solo_status("real") or {}).get("active"):
        wp.gimbal.solo_pause()
        return jsonify({"ok": True, "real": _real_status_all()})
    viewer_main.real_requests.append(("pause", None))
    return jsonify({"ok": True})


@app.route("/api/real/stop", methods=["POST"])
def api_real_stop():
    # Stop here, not on the viewer queue: a busy viewer loop must not delay it.
    _each_arm(rr.stop_real_path)
    wp.gimbal.solo_stop()
    return jsonify({"ok": True, "real": _real_status_all()})


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
    if arms.is_default():
        st = save_settings({"robot_model": mid})
        apply_settings(st)
    else:
        st = load_settings()
        rr.set_robot_model(mid)
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
    _each_arm(rr.request_halt)
    wp.gimbal.solo_stop()
    return jsonify({"ok": True, "sim": _sim_status_all(), "real": _real_status_all()})


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
        traceback.print_exc()
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
    out = fman.write_export_json(path=data.get("path"), title=data.get("title")) or {}
    # The page downloads doc (with the gimbals' frames and poses).
    return jsonify({"ok": bool(out.get("ok")), "path": out.get("path"), "count": out.get("count"),
                    "doc": out.get("doc"), "title": out.get("title")})


@app.route("/api/delete_selected", methods=["POST"])
def api_delete_selected():
    if getattr(wp, "selection_kind", None) == "bot" and not arms.is_default() and not rr.edits_locked():
        # An added Aubo Arm, its waypoints and saved points (the default arm stays).
        arms.remove_arm(arms.active())
        _status_dirty.set()
        return jsonify({"ok": True, "removed_arm": True})
    if wp.gimbal.selected_part():
        return jsonify({"ok": False, "error": "a gimbal's Sim / Real twin can't be deleted"})
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
        # Gimbal frames are edited only through /api/gimbal: keep the server's
        # copy, so a page with an older track list can't drop new frames.
        gim = {str(t.get("id")): t.get("keys") for t in list(getattr(wp, "extra_tracks", None) or [])
               if isinstance(t, dict) and str(t.get("id") or "").startswith(wp.gimbal.TRACK_PREFIX)}
        tracks = [t for t in (data.get("tracks") or []) if t]
        for tr in tracks:
            if isinstance(tr, dict) and str(tr.get("id") or "") in gim:
                tr["keys"] = gim[str(tr.get("id"))]
        wp.extra_tracks = tracks
        # Camera, Look At and gimbal tracks follow their objects, not the client list.
        if hasattr(wp, "sync_camera_tracks"):
            wp.sync_camera_tracks()
    return jsonify({"ok": True, "tracks": _display_tracks()})


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
    # The list's own eye: toggles every item of that list (each item's own eye still counts).
    on = wp.set_list_visible(data.get("kind"), data.get("on"))
    _status_dirty.set()
    return jsonify({"ok": True, "on": bool(on)})


@app.route("/api/vis", methods=["POST"])
def api_vis():
    data = request.get_json(silent=True) or {}
    # A waypoint row of an arm that is not active names its arm.
    aid = data.get("arm") if data.get("arm") in arms.each() else arms.current()
    if hasattr(wp, "set_item_vis"):
        with arms.using(aid):
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
    if data.get("path_start"):
        # Run pressed before the arm is at the path start: gimbals go to their initial pose too.
        rr._gimbals_to_initial_pose()
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
            pos_offset=data.get("pos_offset"),
            copy_offset=data.get("copy_offset"),
            tool=data.get("tool"),
            tool_pick=data.get("tool_pick"),
        ) or {}
    return jsonify({"ok": True, "track": st})


@app.route("/api/arm_move_choice", methods=["POST"])
def api_arm_move_choice():
    data = request.get_json(silent=True) or {}
    ok = wp.resolve_arm_move(bool(data.get("follow"))) if hasattr(wp, "resolve_arm_move") else False
    _status_dirty.set()
    return jsonify({"ok": bool(ok)})


@app.route("/api/offset", methods=["POST"])
def api_offset():
    """Offset component on a scene object or the sim cursor (owner "sim")."""
    data = request.get_json(silent=True) or {}
    owner = data.get("id") or data.get("owner")
    if data.get("kind") == "sim" or owner == "sim" or arms.split_id(owner)[0] == "sim_cursor":
        owner = "sim"
    if not owner:
        owner = getattr(wp, "selected_obj_id", None)
    if not owner:
        return jsonify({"ok": False, "error": "no owner"})
    act = str(data.get("act") or "update").lower()
    if act == "add":
        st = wp.offsets.add_offset_component(owner)
    elif act == "edit":
        # Offset panel clicked: same as clicking the yellow marker.
        wp.offsets.begin_offset_edit(owner)
        st = wp.offsets.offset_status(owner)
    else:
        st = wp.offsets.update_offset(
            owner,
            enabled=data.get("enabled"),
            pos=data.get("pos"),
            wxyz=data.get("wxyz"),
            remove=(act == "remove"),
        )
    _status_dirty.set()
    return jsonify({"ok": st is not None, "offset": st or {}})


def _collider_act(owner, data):
    act = str(data.get("act") or "update").lower()
    if act == "add":
        st = wp.colliders.add_collider_component(owner)
    elif act == "edit":
        # Collider panel clicked: show the green outlines.
        wp.colliders.begin_collider_edit(owner)
        st = wp.colliders.collider_status(owner)
    else:
        st = wp.colliders.update_collider(
            owner,
            act=act,
            enabled=data.get("enabled"),
            stop=data.get("stop"),
            margin=data.get("margin"),
            index=data.get("index"),
            shape=data.get("shape"),
            pos=data.get("pos"),
            wxyz=data.get("wxyz"),
            scale=data.get("scale"),
            ignore_parent=data.get("ignore_parent"),
        )
    return st


@app.route("/api/collider", methods=["POST"])
def api_collider():
    """Collider component on a scene object or the Aubo Arm (sim_bot: the
    robot-shaped collider). The sim cursor cannot have one."""
    data = request.get_json(silent=True) or {}
    owner = data.get("id") or data.get("owner")
    if data.get("kind") == "sim" or owner == "sim" or arms.split_id(owner)[0] == "sim_cursor":
        return jsonify({"ok": False, "error": "the Collider component is on the Aubo Arm, not the sim cursor"})
    if data.get("kind") == "bot":
        owner = owner or "sim_bot"
    if not owner:
        owner = getattr(wp, "selected_obj_id", None)
    if not owner:
        return jsonify({"ok": False, "error": "no owner"})
    base, aid = arms.split_id(owner)
    if base == "sim_bot" and aid is not None:
        # An Aubo Arm: its robot collider, read and written as that arm.
        with arms.using(aid):
            st = _collider_act("sim", data)
    else:
        st = _collider_act(owner, data)
    _status_dirty.set()
    return jsonify({"ok": st is not None, "collider": st or {}})


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


@app.route("/api/gimbal", methods=["POST"])
def api_gimbal():
    """Gimbal object: {id, action, ...}.
    update {cfg?, target?, look?} | move {target?} | insert_frame {t?, target?} |
    delete_frame {t | index | ts} | move_frames {moves: [{t0, t}]} |
    frame_dir {index | t, dir: [J1, J2, J3]} | edit_frame {index, t?, angles?, dir?} |
    paste_frames {frames} | use_read (target = angles read from the gimbal)."""
    data = request.get_json(silent=True) or {}
    oid = str(data.get("id") or getattr(wp, "selected_obj_id", None) or "")
    act = str(data.get("action") or "update")
    if wp.gimbal._find(oid) is None:
        return jsonify({"ok": False, "error": "no such gimbal"}), 404
    ok = True
    # One undo step per edit of the gimbal's settings, target or frames (without
    # one, Undo went back past them to before the gimbal was added).
    if (any(data.get(k) is not None for k in ("cfg", "target", "look", "world_target"))
            or act in ("use_read", "insert_frame", "delete_frame", "move_frames", "frame_dir",
                       "edit_frame", "paste_frames")) and not _edits_locked():
        wp._push_history()
    if any(data.get(k) is not None for k in ("cfg", "target", "look")):
        ok = wp.gimbal.update(oid, data)
    if data.get("world_target") is not None:
        # World-mode Target: camera yaw / pitch / roll -> J1-J3 (inverse kinematics).
        err = wp.gimbal.set_world_target(oid, data.get("world_target"))
        ok = err is not None
        if err is None:
            return jsonify({"ok": False, "error": "Look At is aiming the gimbal",
                            "gimbal": wp.gimbal.status(oid)})
    if act == "move":
        # direction: [J1, J2, J3] (short / cw / ccw), or one word for J1.
        ok = wp.gimbal.request_move(oid, direction=data.get("direction") or "short")
    elif act == "calibrate":
        # The gimbal is in its home pose (camera at +x, level): its reading becomes the model's zero pose.
        ok = wp.gimbal.request_calibrate(oid)
    elif act == "use_read":
        st = wp.gimbal.status(oid) or {}
        ok = st.get("read") is not None
        if ok:
            read = list(st.get("read"))
            read[0] = wp.gimbal.unwrap_yaw(read[0], (st.get("target") or [None])[0])
            wp.gimbal.update(oid, {"target": read})
    elif act == "insert_frame":
        if _edits_locked():
            return jsonify({"ok": False, "error": "stop playback first"}), 409
        t = data.get("t")
        if t is None:
            t = getattr(viewer_main, "playhead_t", 0.0)
        angles = None
        if data.get("source") == "real":
            # The timeline's orange +: the real gimbal's reading.
            angles = (wp.gimbal.status(oid) or {}).get("read")
            if angles is None:
                return jsonify({"ok": False, "error": "the real gimbal is not reading",
                                "gimbal": wp.gimbal.status(oid)})
        # live: a "real pose at run" frame (the reading when the run starts; Sim: camera along +x).
        ok = wp.gimbal.insert_frame(oid, t, angles, reading=angles is not None, live=bool(data.get("live")))
    elif act == "delete_frame":
        if _edits_locked():
            return jsonify({"ok": False, "error": "stop playback first"}), 409
        ok = wp.gimbal.delete_frame(oid, t=data.get("t"), index=data.get("index"), ts=data.get("ts"))
    elif act == "move_frames":
        # The timeline's drag: {moves: [{t0, t}]}.
        if _edits_locked():
            return jsonify({"ok": False, "error": "stop playback first"}), 409
        ok = wp.gimbal.move_frames(oid, data.get("moves")) is not None
    elif act == "frame_dir":
        # A frame's direction per joint: {index | t, dir: [J1, J2, J3]} ("" / short / cw / ccw, null keeps one).
        if _edits_locked():
            return jsonify({"ok": False, "error": "stop playback first"}), 409
        ok = wp.gimbal.set_frame_dir(oid, data.get("dir"), index=data.get("index"), t=data.get("t"))
    elif act in ("edit_frame", "paste_frames"):
        # The timeline's Edit window {index, t?, angles?, dir?} and Paste {frames: [{t, yaw, roll, pitch, dir?}]}.
        if _edits_locked():
            return jsonify({"ok": False, "error": "stop playback first"}), 409
        if act == "edit_frame":
            ok = wp.gimbal.edit_frame(oid, data.get("index"), t=data.get("t"), angles=data.get("angles"),
                                      dirs=data.get("dir"), live=data.get("live"))
        else:
            ok = wp.gimbal.paste_frames(oid, data.get("frames"))
    _status_dirty.set()
    return jsonify({"ok": bool(ok), "gimbal": wp.gimbal.status(oid),
                    "tracks": _display_tracks()})


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


@app.route("/api/speed_export", methods=["GET", "POST"])
def api_speed_export():
    """The Speed Graph's Export. POST {sim, real} (the graph's samples, [{t, v}]):
    writes exports/speed_export.json with those, the bake's overspeed report
    (also rewritten as bake_overspeed.csv) and the gimbals' frames and poses.
    GET ?download=1 sends that file; plain GET just rewrites the overspeed report.
    (The page POSTed to a GET-only route: every export failed with a 405.)"""
    path = app_paths.ensure_dir(app_paths.EXPORTS_DIR) / "speed_export.json"
    if request.method == "GET" and request.args.get("download"):
        if path.is_file():
            return send_file(str(path), as_attachment=True, download_name="speed_export.json")
        return jsonify({"ok": False, "error": "nothing exported yet"}), 404
    over = None
    if hasattr(bake, "export_overspeed"):
        try:
            over = bake.export_overspeed()
        except Exception as e:
            over = {"ok": False, "error": str(e)}
    if request.method == "GET":
        return jsonify(over or {"ok": True})
    data = request.get_json(silent=True) or {}

    def series(rows):
        out = []
        for r in list(rows or []):
            try:
                out.append({"t": round(float(r.get("t")), 4), "v": round(float(r.get("v")), 5)})
            except Exception:
                pass
        return out
    try:
        gimbals = wp.gimbal.export_tracks()
    except Exception as e:
        print("speed export gimbals failed", e)
        gimbals = []
    doc = {"format": "speed_export", "title": fman.SESSION.get("title"), "units": {"speed": "m/s", "t": "s"},
           "sim_ee": series(data.get("sim")), "real_ee": series(data.get("real")),
           "overspeed": over, "gimbals": gimbals}
    path.write_text(json.dumps(doc, indent=2, default=str), encoding="utf-8")
    return jsonify({"ok": True, "path": str(path), "sim": len(doc["sim_ee"]), "real": len(doc["real_ee"]),
                    "gimbals": len(gimbals), "doc": json.loads(json.dumps(doc, default=str))})


def _open_ui():
    time.sleep(2)
    try:
        webbrowser.open("http://127.0.0.1:5000")
    except Exception:
        pass


_autosave_soon = threading.Event()
AUTOSAVE_EVERY_S = 60.0
AUTOSAVE_AFTER_EDIT_S = 2.0


def _autosave_loop():
    """Every minute, and a couple of seconds after each edit (an undo step:
    waypoints, objects, gimbal settings / target / frames, ...). Unchanged
    documents are skipped (write_autosave compares a hash)."""
    wp.on_edit.append(_autosave_soon.set)
    _autosave_soon.wait(AUTOSAVE_EVERY_S)
    while True:
        if _autosave_soon.is_set():
            time.sleep(AUTOSAVE_AFTER_EDIT_S)  # let a burst of edits finish
            _autosave_soon.clear()
        try:
            fman.write_autosave()
        except Exception as e:
            print("autosave failed", e)
        _autosave_soon.wait(AUTOSAVE_EVERY_S)


def _start_controller():
    """Xbox / PlayStation controller jogging (gamepad.py, same jog as the
    Steam Deck edition in deck_webui.py)."""
    try:
        import deck_webui
        deck_webui.start_controller()
    except Exception as e:
        traceback.print_exc()
        print("controller unavailable:", e)
        reason = "%s: %s" % (type(e).__name__, e)
        if "/api/controller" not in [r.rule for r in app.url_map.iter_rules()]:
            # Tell the page why, instead of leaving it without a Controller row.
            app.add_url_rule("/api/controller", "api_controller_unavailable",
                             lambda: jsonify({"ok": False, "started": False, "error": reason}),
                             methods=["GET", "POST"])


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
    _start_controller()
    print("Web UI   http://127.0.0.1:5000")
    threading.Thread(target=_open_ui, daemon=True).start()
    threading.Thread(target=_autosave_loop, daemon=True).start()
    app.run(host="127.0.0.1", port=5000, debug=False, use_reloader=False, threaded=True, request_handler=QuietHandler)
