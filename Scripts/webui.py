# version: beta.0904
"""
Parent Web UI.
Run from the folder that contains templates/ and robot_sim/.
"""
import json
import sys
import threading
import time
import traceback
import webbrowser
from pathlib import Path

from collections import deque
from flask import Flask, Response, render_template, jsonify, request
from werkzeug.serving import WSGIRequestHandler

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "robot_sim"))
sys.path.insert(0, str(ROOT))
SETTINGS_PATH = ROOT / "settings.json"
JOINT_LIMIT_NAMES = (
    "shoulder_joint",
    "upperArm_joint",
    "foreArm_joint",
    "wrist1_joint",
    "wrist2_joint",
    "wrist3_joint",
)
SETTINGS_DEFAULT = {
    "joint_vel_deg": [223.0, 223.0, 223.0, 267.0, 267.0, 267.0],
    "joint_acc_deg": [496.0, 496.0, 496.0, 594.0, 594.0, 594.0],
    "spacing": 0.05,
    "path_window_radius": 1.0,
    "path_window_on": False,
    "auto_sample": False,
    "hold_lin_mps": 0.05,
    "burner_duration": 2.0,
    "grid_spacing": 0.01,
    "snap_to_grid": True,
    "dev_mode": False,
}

DEV_MODE_PASSWORD = "1"

import waypoints as wp
import robot_runtime as rr
import robot_sim_run as viewer_main
import path_bake as bake
import file_manage as fman

app = Flask(__name__, template_folder=str(ROOT / "templates"))
print("path_bake loaded from", getattr(bake, "__file__", bake), "spacing", getattr(bake, "spacing", None))
print("robot_runtime loaded from", getattr(rr, "__file__", rr), "init", hasattr(rr, "init"))


def _settings_defaults():
    return {
        "joint_vel_deg": list(SETTINGS_DEFAULT["joint_vel_deg"]),
        "joint_acc_deg": list(SETTINGS_DEFAULT["joint_acc_deg"]),
        "spacing": float(SETTINGS_DEFAULT["spacing"]),
        "path_window_radius": float(SETTINGS_DEFAULT["path_window_radius"]),
        "path_window_on": bool(SETTINGS_DEFAULT["path_window_on"]),
        "auto_sample": bool(SETTINGS_DEFAULT.get("auto_sample", False)),
        "hold_lin_mps": float(SETTINGS_DEFAULT.get("hold_lin_mps", 0.05)),
        "burner_duration": float(SETTINGS_DEFAULT.get("burner_duration", 2.0)),
        "grid_spacing": float(SETTINGS_DEFAULT.get("grid_spacing", 0.01)),
        "snap_to_grid": bool(SETTINGS_DEFAULT.get("snap_to_grid", True)),
        "dev_mode": bool(SETTINGS_DEFAULT.get("dev_mode", False)),
        "camera_urdf": str(SETTINGS_DEFAULT.get("camera_urdf", "")),
    }


def load_settings():
    data = _settings_defaults()
    if not SETTINGS_PATH.exists():
        try:
            SETTINGS_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")
        except Exception as e:
            print("settings.json create failed", e)
        apply_settings(data)
        return data
    try:
        raw = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
        if isinstance(raw, dict):
            data.update(raw)
    except Exception as e:
        print("settings.json read failed", e)
    apply_settings(data)
    return data


def apply_settings(data):
    vel = list(data.get("joint_vel_deg") or SETTINGS_DEFAULT["joint_vel_deg"])
    acc = list(data.get("joint_acc_deg") or SETTINGS_DEFAULT["joint_acc_deg"])
    while len(vel) < 6:
        vel.append(SETTINGS_DEFAULT["joint_vel_deg"][len(vel)])
    while len(acc) < 6:
        acc.append(SETTINGS_DEFAULT["joint_acc_deg"][len(acc)])
    try:
        for i, name in enumerate(JOINT_LIMIT_NAMES):
            rr.AUBO_I5_MAX_VEL_DEG[name] = float(vel[i])
            rr.AUBO_I5_MAX_ACCEL_DEG[name] = float(acc[i])
    except Exception as e:
        print("apply joint limits failed", e)
    try:
        if hasattr(bake, "set_spacing"):
            bake.set_spacing(float(data.get("spacing", 0.05)))
    except Exception:
        pass
    try:
        rr.HOLD_LIN_MPS = max(0.001, float(data.get("hold_lin_mps", 0.05)))
    except Exception:
        pass
    try:
        rr.BURNER_DURATION = max(0.05, float(data.get("burner_duration", 2.0)))
    except Exception:
        pass


def save_settings(patch):
    data = load_settings()
    if isinstance(patch, dict):
        data.update(patch)
    try:
        SETTINGS_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")
    except Exception as e:
        print("settings.json write failed", e)
    apply_settings(data)
    return data


load_settings()
try:
    wp.load_global_stored_points((load_settings() or {}).get("stored_points") or [])
except Exception:
    pass


class QuietHandler(WSGIRequestHandler):
    def log_request(self, code="-", size="-"):
        if self.path.startswith("/api/status") or self.path.startswith("/api/console") or self.path.startswith("/api/clock") or self.path.startswith("/api/camera_state") or self.path.startswith("/api/cam_frame"):
            return
        super().log_request(code, size)


_console_lines = deque(maxlen=3000)
_console_seq = 0
_console_lock = threading.Lock()


class _ConsoleTee:
    def __init__(self, stream):
        self._stream = stream

    def write(self, data):
        if data:
            try:
                self._stream.write(data)
                self._stream.flush()
            except Exception:
                pass
            global _console_seq
            with _console_lock:
                for line in str(data).splitlines():
                    if line == "":
                        continue
                    _console_seq += 1
                    _console_lines.append((_console_seq, line))
        return len(data) if data else 0

    def flush(self):
        try:
            self._stream.flush()
        except Exception:
            pass

    def isatty(self):
        return False

    def __getattr__(self, name):
        return getattr(self._stream, name)


if not isinstance(sys.stdout, _ConsoleTee):
    sys.stdout = _ConsoleTee(sys.stdout)
if not isinstance(sys.stderr, _ConsoleTee):
    sys.stderr = _ConsoleTee(sys.stderr)


def start_viewer_thread():
    t = threading.Thread(target=viewer_main.start_viewer, daemon=True)
    t.start()
    for _ in range(200):
        if getattr(viewer_main, "VISER_PORT", None):
            break
        time.sleep(0.1)


# Native index.html owns #tlLabels / extraTracks. This shim must NOT capture
# .tl-add-track clicks or rewrite the label DOM — that raced renderLabels()
# (parallel window.__extraTracks vs script-scoped extraTracks) so POST
# succeeded while the row vanished on the next refresh.
_TRACKS_BOOTSTRAP = """
<script id="tracks-bootstrap">
(function () {
  if (window.__tracksBoot) return;
  window.__tracksBoot = true;
  document.addEventListener("pointerdown", function (e) {
    var btn = e.target && e.target.closest ? e.target.closest(".tl-add-track") : null;
    if (!btn) return;
    var ui = window.TracksUI;
    if (!ui || typeof ui.add !== "function") return;
    e.preventDefault();
    e.stopPropagation();
    if (btn.__trackArmed) return;
    btn.__trackArmed = true;
    try { ui.add(); } finally {
      setTimeout(function () { btn.__trackArmed = false; }, 0);
    }
  }, true);
})();
</script>
"""


@app.route("/")
def index():
    port = viewer_main.VISER_PORT or 8080
    html = render_template("index.html", viser_url=f"http://127.0.0.1:{port}")
    if "</body>" in html:
        html = html.replace("</body>", _TRACKS_BOOTSTRAP + "\n</body>", 1)
    else:
        html = html + _TRACKS_BOOTSTRAP
    return html


def _edits_locked():
    return bool(hasattr(rr, "edits_locked") and rr.edits_locked())


def _queue_add(source):
    if _edits_locked():
        return jsonify({"ok": False, "error": "locked during playback"}), 409
    if getattr(wp, "server", None) is None:
        return jsonify({"ok": False, "error": "viewer not ready"}), 503
    data = request.get_json(silent=True) or {}
    t = data.get("t", None)
    if t is not None:
        try:
            t = float(t)
            viewer_main.playhead_t = t
        except Exception:
            t = None
    replace = bool(data.get("replace"))
    check_t = t if t is not None else getattr(viewer_main, "playhead_t", 0.0)
    if not replace and hasattr(wp, "find_waypoint_at_time"):
        hit = wp.find_waypoint_at_time(check_t)
        if hit is not None:
            return jsonify({
                "ok": False,
                "needs_confirm": True,
                "index": hit,
                "t": check_t,
                "source": source,
            })
    viewer_main.add_requests.append({"source": source, "t": t, "replace": replace})
    return jsonify({"ok": True, "queued": True, "source": source, "t": t, "count": len(wp.waypoints)})


@app.route("/api/add_waypoint", methods=["POST"])
@app.route("/api/add_waypoint_real", methods=["POST"])
def api_add_waypoint_real():
    try:
        return _queue_add("real")
    except Exception as e:
        traceback.print_exc()
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/api/add_waypoint_sim", methods=["POST"])
def api_add_waypoint_sim():
    try:
        return _queue_add("sim")
    except Exception as e:
        traceback.print_exc()
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/api/clear_bake", methods=["POST"])
def api_clear_bake():
    if _edits_locked():
        return jsonify({"ok": False, "error": "locked during playback"}), 409
    if hasattr(bake, "is_baking") and bake.is_baking():
        return jsonify({"ok": False, "error": "bake in progress"}), 409
    try:
        out = bake.clear_bake() if hasattr(bake, "clear_bake") else {"ok": False}
        out = dict(out or {})
        out["ok"] = bool(out.get("ok", True))
        out["bake"] = bake.status_dict() if hasattr(bake, "status_dict") else out.get("bake") or {}
        out["timeline"] = bake.timeline_payload() if hasattr(bake, "timeline_payload") else {}
        return jsonify(out)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/api/optomise", methods=["POST"])
def api_optomise():
    if _edits_locked():
        return jsonify({"ok": False, "error": "locked during playback"}), 409
    if hasattr(bake, "is_baking") and bake.is_baking():
        return jsonify({"ok": False, "error": "bake in progress"}), 409
    try:
        data = request.get_json(silent=True) or {}
        spacing = data.get("spacing", getattr(bake, "spacing", 0.05))
        result = bake.run_optomise(float(spacing)) if hasattr(bake, "run_optomise") else {"ok": False}
        return jsonify({
            "ok": bool(result.get("ok", True)),
            "queued": False,
            "spacing": float(spacing),
            "baked": result.get("baked"),
            "total": result.get("total"),
            "failed_at": result.get("failed_at"),
            "error": result.get("error"),
            "optomise": result.get("optomise"),
            "bake": bake.status_dict() if hasattr(bake, "status_dict") else {},
            "timeline": bake.timeline_payload() if hasattr(bake, "timeline_payload") else {},
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/api/bake", methods=["POST"])
def api_bake():
    if _edits_locked():
        return jsonify({"ok": False, "error": "locked during playback"}), 409
    if hasattr(bake, "is_baking") and bake.is_baking():
        return jsonify({"ok": False, "error": "bake in progress"}), 409
    try:
        data = request.get_json(silent=True) or {}
        spacing = data.get("spacing", getattr(bake, "spacing", 0.05))
        try:
            if hasattr(rr, "drop_uploaded_buffer"):
                rr.drop_uploaded_buffer()
        except Exception:
            pass
        result = bake.run_bake(float(spacing)) if hasattr(bake, "run_bake") else {"ok": False}
        return jsonify({
            "ok": bool(result.get("ok", True)),
            "queued": False,
            "spacing": float(spacing),
            "baked": result.get("baked"),
            "total": result.get("total"),
            "failed_at": result.get("failed_at"),
            "error": result.get("error"),
            "bake": bake.status_dict() if hasattr(bake, "status_dict") else {},
            "timeline": bake.timeline_payload() if hasattr(bake, "timeline_payload") else {},
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/api/gizmo_space", methods=["POST"])
def api_gizmo_space():
    data = request.get_json(silent=True) or {}
    space = str(data.get("space", "local")).lower()
    mode = "local" if space.startswith("loc") else "global"
    viewer_main.gizmo_space_requests.append(mode)
    return jsonify({"ok": True, "queued": True, "space": mode})


@app.route("/api/reset_sim", methods=["POST"])
def api_reset_sim():
    try:
        ok = rr.reset_sim_to_real()
        viewer_main.reset_requests.append(True)
        return jsonify({"ok": bool(ok)})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/api/reset_sim_home", methods=["POST"])
def api_reset_sim_home():
    try:
        ok = rr.reset_sim_to_home()
        return jsonify({"ok": bool(ok)})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/api/sim_poses", methods=["GET", "POST"])
def api_sim_poses():
    if request.method == "GET":
        return jsonify({"ok": True, "poses": _sim_poses_payload()})
    data = request.get_json(silent=True) or {}
    act = str(data.get("act") or "add").lower()
    if act == "delete":
        pid = str(data.get("id") or "")
        if pid in ("real", "home"):
            if hasattr(wp, "ensure_builtin_sim_poses"):
                wp.ensure_builtin_sim_poses()
            return jsonify({"ok": False, "error": "builtin", "poses": _sim_poses_payload()}), 400
        ok = wp.delete_sim_pose(pid) if hasattr(wp, "delete_sim_pose") else False
        return jsonify({"ok": bool(ok), "poses": _sim_poses_payload()})
    name = data.get("name")
    ik = _q6(data.get("ik"))
    units = str(data.get("units") or "rad").lower()
    raw_q = _q6(data.get("q"))
    q_deg = None
    if units.startswith("deg"):
        q_deg = raw_q
        q = [v * 3.141592653589793 / 180.0 for v in raw_q]
    else:
        q = raw_q
    rec = wp.add_sim_pose(
        name, ik=ik, q=q, q_deg=q_deg, source=data.get("source")
    ) if hasattr(wp, "add_sim_pose") else {}
    return jsonify({"ok": True, "pose": rec, "poses": _sim_poses_payload()})


@app.route("/api/sim_go", methods=["POST"])
def api_sim_go():
    data = request.get_json(silent=True) or {}
    pid = str(data.get("id") or "real")
    try:
        if pid == "real":
            ok = rr.reset_sim_to_real()
            viewer_main.reset_requests.append(True)
        elif pid == "home":
            ok = rr.reset_sim_to_home()
        else:
            rec = None
            for p in list(getattr(wp, "sim_poses", None) or []):
                if str(p.get("id")) == pid:
                    rec = p
                    break
            if rec is None:
                return jsonify({"ok": False, "error": "unknown pose"}), 404
            if rec.get("q_deg"):
                ok = rr.reset_sim_to_q(rec.get("q_deg"), units="deg")
            else:
                ok = rr.reset_sim_to_q(rec.get("q"), units="rad")
        return jsonify({"ok": bool(ok), "id": pid})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/api/point_rename", methods=["POST"])
def api_point_rename():
    data = request.get_json(silent=True) or {}
    kind = str(data.get("kind") or "")
    name = data.get("name")
    if kind == "wp":
        try:
            idx = int(data.get("index"))
        except Exception:
            return jsonify({"ok": False, "error": "bad index"}), 400
        ok = wp.rename_waypoint(idx, name) if hasattr(wp, "rename_waypoint") else None
        return jsonify({"ok": bool(ok), "name": ok})
    pid = str(data.get("id") or "")
    rec = wp.rename_sim_pose(pid, name) if hasattr(wp, "rename_sim_pose") else None
    if rec and str(rec.get("store")) == "global":
        try:
            wp.persist_stored_points()
        except Exception:
            pass
    return jsonify({"ok": rec is not None, "pose": rec})


@app.route("/api/point_store", methods=["POST"])
def api_point_store():
    data = request.get_json(silent=True) or {}
    rec = wp.set_pose_store(data.get("id"), data.get("store")) if hasattr(wp, "set_pose_store") else None
    return jsonify({"ok": rec is not None, "pose": rec})


@app.route("/api/point_order", methods=["POST"])
def api_point_order():
    data = request.get_json(silent=True) or {}
    rows = wp.set_point_order(data.get("ids") or [], data.get("sort")) if hasattr(wp, "set_point_order") else []
    return jsonify({"ok": True, "points": rows, "sort": getattr(wp, "point_sort", "type")})


@app.route("/api/point_go", methods=["POST"])
def api_point_go():
    data = request.get_json(silent=True) or {}
    kind = str(data.get("kind") or "")
    try:
        if kind == "wp":
            idx = int(data.get("index"))
            if idx < 0 or idx >= len(wp.waypoints):
                return jsonify({"ok": False, "error": "bad index"}), 400
            q = wp.waypoints[idx].get("q")
            if q is None:
                return jsonify({"ok": False, "error": "no q"}), 400
            ok = rr.reset_sim_to_q(q, units="rad") if hasattr(rr, "reset_sim_to_q") else False
            return jsonify({"ok": bool(ok)})
        return api_sim_go()
    except Exception as e:
        traceback.print_exc()
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/api/real/goto", methods=["POST"])
def api_real_goto():
    viewer_main.real_requests.append("goto")
    return jsonify({"ok": True})


@app.route("/api/real/upload", methods=["POST"])
def api_real_upload():
    data = request.get_json(silent=True) or {}
    try:
        loops = int(data.get("repeat") if data.get("repeat") is not None else 0)
    except Exception:
        loops = 0
    try:
        delay = float(data.get("delay") or 0.0)
    except Exception:
        delay = 0.0
    try:
        ok = bool(rr.upload_real_path(loops=loops, delay=delay))
    except Exception as e:
        traceback.print_exc()
        return jsonify({"ok": False, "error": str(e), "real": rr.real_path_status()}), 500
    st = rr.real_path_status()
    if not ok:
        return jsonify({"ok": False, "error": "upload did not start", "real": st})
    return jsonify({"ok": True, "real": st})


@app.route("/api/real/start", methods=["POST"])
def api_real_start():
    ok = False
    try:
        ok = bool(rr.fire_real_path()) if hasattr(rr, "fire_real_path") else bool(rr.start_real_path())
    except Exception as e:
        traceback.print_exc()
        return jsonify({"ok": False, "error": str(e), "real": rr.real_path_status() if hasattr(rr, "real_path_status") else {}}), 500
    if not ok:
        st = rr.real_path_status() if hasattr(rr, "real_path_status") else {}
        return jsonify({"ok": False, "error": "real path did not start (upload first, at start pose)", "real": st})
    return jsonify({"ok": True, "real": rr.real_path_status() if hasattr(rr, "real_path_status") else {}})


@app.route("/api/real/pause", methods=["POST"])
def api_real_pause():
    paused = None
    try:
        paused = rr.pause_real_path() if hasattr(rr, "pause_real_path") else None
    except Exception:
        viewer_main.real_requests.append("pause")
    return jsonify({"ok": True, "paused": paused, "real": rr.real_path_status() if hasattr(rr, "real_path_status") else {}})


@app.route("/api/real/stop", methods=["POST"])
def api_real_stop():
    try:
        if hasattr(rr, "request_halt"):
            rr.request_halt()
        elif hasattr(rr, "stop_real_path"):
            rr.stop_real_path()
        else:
            viewer_main.real_requests.append("stop")
    except Exception:
        viewer_main.real_requests.append("stop")
    return jsonify({"ok": True, "real": rr.real_path_status() if hasattr(rr, "real_path_status") else {}})


@app.route("/api/simulate", methods=["POST"])
def api_simulate():
    try:
        if hasattr(bake, "is_baking") and bake.is_baking():
            return jsonify({"ok": False, "error": "bake in progress"}), 409
        data = request.get_json(silent=True) or {}
        try:
            loops = int(data.get("repeat") if data.get("repeat") is not None else 0)
        except Exception:
            loops = 0
        try:
            delay = float(data.get("delay") or 0.0)
        except Exception:
            delay = 0.0
        ok = rr.enter_simulation_mode(loops=loops, delay=delay) if hasattr(rr, "enter_simulation_mode") else False
        return jsonify({"ok": bool(ok), "sim": rr.sim_status() if hasattr(rr, "sim_status") else {}})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/api/sim/pause", methods=["POST"])
def api_sim_pause():
    paused = rr.toggle_sim_pause() if hasattr(rr, "toggle_sim_pause") else None
    return jsonify({"ok": True, "paused": paused, "sim": rr.sim_status()})


@app.route("/api/sim/stop", methods=["POST"])
def api_sim_stop():
    if hasattr(rr, "request_halt"):
        rr.request_halt()
    else:
        rr.exit_simulation_mode()
    return jsonify({"ok": True, "sim": rr.sim_status(), "real": rr.real_path_status() if hasattr(rr, "real_path_status") else {}})


@app.route("/api/halt", methods=["POST"])
def api_halt():
    if hasattr(rr, "request_halt"):
        rr.request_halt()
    else:
        if hasattr(rr, "stop_real_path"):
            rr.stop_real_path()
        if hasattr(rr, "exit_simulation_mode"):
            rr.exit_simulation_mode()
    return jsonify({
        "ok": True,
        "sim": rr.sim_status() if hasattr(rr, "sim_status") else {},
        "real": rr.real_path_status() if hasattr(rr, "real_path_status") else {},
    })


@app.route("/api/keyframe_time", methods=["POST"])
def api_keyframe_time():
    if _edits_locked():
        return jsonify({"ok": False, "error": "locked during playback"}), 409
    data = request.get_json(silent=True) or {}
    released = bool(data.get("released", False))
    raw_moves = data.get("moves")
    moves = []
    if isinstance(raw_moves, list) and raw_moves:
        for rec in raw_moves:
            try:
                moves.append((int(rec.get("index")), float(rec.get("t", 0.0))))
            except Exception:
                continue
    else:
        try:
            moves.append((int(data.get("index")), float(data.get("t", 0.0))))
        except Exception:
            return jsonify({"ok": False, "error": "bad args"}), 400
    if hasattr(wp, "set_keyframe_times"):
        rows = wp.set_keyframe_times(moves, released=released)
    else:
        rows = []
        nt = wp.set_keyframe_time(moves[0][0], moves[0][1], released=released)
        rows = [{"index": moves[0][0], "t": nt}]
    first = rows[0] if rows else {"index": moves[0][0], "t": moves[0][1]}
    return jsonify({
        "ok": True,
        "t": first.get("t"),
        "index": first.get("index"),
        "moves": rows,
    })


@app.route("/api/sample_time", methods=["POST"])
def api_sample_time():
    if _edits_locked():
        return jsonify({"ok": False, "error": "locked during playback"}), 409
    data = request.get_json(silent=True) or {}
    raw = data.get("moves")
    if isinstance(raw, list) and raw:
        rows = bake.set_sample_times(raw) if hasattr(bake, "set_sample_times") else []
        first = rows[0] if rows else {}
        return jsonify({"ok": bool(rows), "moves": rows, "t": first.get("t"), "seg": first.get("seg"), "index": first.get("index")})
    try:
        seg = int(data.get("seg"))
        k = int(data.get("index"))
        t = float(data.get("t", 0.0))
    except Exception:
        return jsonify({"ok": False, "error": "bad args"}), 400
    nt = bake.set_sample_time(seg, k, t) if hasattr(bake, "set_sample_time") else None
    return jsonify({"ok": nt is not None, "t": nt, "seg": seg, "index": k})


@app.route("/api/select_sample", methods=["POST"])
def api_select_sample():
    data = request.get_json(silent=True) or {}
    if data.get("seg") is None:
        bake.selected_sample = None
        return jsonify({"ok": True})
    try:
        bake.select_sample(int(data.get("seg")), int(data.get("index")))
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 400
    return jsonify({"ok": True})


@app.route("/api/transition", methods=["POST"])
def api_transition():
    if _edits_locked():
        return jsonify({"ok": False, "error": "locked during playback"}), 409
    data = request.get_json(silent=True) or {}
    try:
        idx = int(data.get("index"))
    except Exception:
        return jsonify({"ok": False, "error": "bad args"}), 400
    mode = str(data.get("mode") or "linear")
    c1 = data.get("c1")
    c2 = data.get("c2")
    ok = bake.apply_transition(idx, mode, c1=c1, c2=c2) if hasattr(bake, "apply_transition") else False
    return jsonify({"ok": bool(ok), "mode": mode, "index": idx})


@app.route("/api/custom_curve")
def api_custom_curve():
    idx = int(request.args.get("index", 0))
    rec = bake.get_custom_curve(idx) if hasattr(bake, "get_custom_curve") else {"c1": [0.33, 0.33], "c2": [0.66, 0.66]}
    return jsonify({"ok": True, **rec})


@app.route("/api/copy", methods=["POST"])
def api_copy():
    data = request.get_json(silent=True) or {}
    out = bake.copy_selection(data.get("keyframes") or [], data.get("samples") or [])
    return jsonify(out)


@app.route("/api/cut", methods=["POST"])
def api_cut():
    if _edits_locked():
        return jsonify({"ok": False, "error": "locked"}), 409
    data = request.get_json(silent=True) or {}
    out = bake.cut_selection(data.get("keyframes") or [], data.get("samples") or [])
    return jsonify(out)


@app.route("/api/paste", methods=["POST"])
def api_paste():
    if _edits_locked():
        return jsonify({"ok": False, "error": "locked"}), 409
    data = request.get_json(silent=True) or {}
    t0 = data.get("t", getattr(viewer_main, "playhead_t", 0.0))
    out = bake.paste_clipboard(t0)
    return jsonify(out)


@app.route("/api/settings", methods=["GET", "POST"])
def api_settings():
    if request.method == "GET":
        return jsonify({"ok": True, "settings": load_settings()})
    data = request.get_json(silent=True) or {}
    if data.get("dev_mode") and not load_settings().get("dev_mode"):
        data = dict(data)
        data["dev_mode"] = False
    out = save_settings(data)
    return jsonify({"ok": True, "settings": out})


@app.route("/api/dev_mode", methods=["POST"])
def api_dev_mode():
    data = request.get_json(silent=True) or {}
    if str(data.get("password") or "") != str(DEV_MODE_PASSWORD):
        return jsonify({"ok": False, "error": "bad password"}), 403
    out = save_settings({"dev_mode": True})
    return jsonify({"ok": True, "dev_mode": True, "settings": out})


@app.route("/api/linear_speed", methods=["POST"])
def api_linear_speed():
    if _edits_locked():
        return jsonify({"ok": False, "error": "locked during playback"}), 409
    data = request.get_json(silent=True) or {}
    try:
        speed = float(data.get("speed"))
    except Exception:
        return jsonify({"ok": False, "error": "bad speed"}), 400
    raw = data.get("indices")
    if raw is None and data.get("index") is not None:
        raw = [data.get("index")]
    try:
        indices = [int(v) for v in (raw or [])]
    except Exception:
        return jsonify({"ok": False, "error": "bad indices"}), 400
    if hasattr(wp, "apply_linear_speed"):
        out = wp.apply_linear_speed(indices, speed)
    elif hasattr(bake, "apply_linear_speed"):
        out = bake.apply_linear_speed(indices, speed)
    else:
        out = {"ok": False, "error": "missing"}
    return jsonify(out)


@app.route("/api/force_fk", methods=["POST"])
def api_force_fk():
    if _edits_locked():
        return jsonify({"ok": False, "error": "locked during playback"}), 409
    data = request.get_json(silent=True) or {}
    try:
        idx = int(data.get("index"))
    except Exception:
        return jsonify({"ok": False, "error": "bad args"}), 400
    if hasattr(wp, "toggle_force_fk"):
        on = wp.toggle_force_fk(idx)
    else:
        if 0 <= idx < len(wp.waypoints):
            if idx >= len(wp.waypoints) - 1 and len(wp.waypoints) >= 2:
                idx = len(wp.waypoints) - 2
            wp.waypoints[idx]["force_fk"] = not bool(wp.waypoints[idx].get("force_fk"))
            on = bool(wp.waypoints[idx]["force_fk"])
            wp.draw_path()
        else:
            on = False
    print("api force_fk", idx, on)
    return jsonify({"ok": True, "index": idx, "force_fk": bool(on)})


@app.route("/api/merge", methods=["POST"])
def api_merge():
    if _edits_locked():
        return jsonify({"ok": False, "error": "locked"}), 409
    data = request.get_json(silent=True) or {}
    a, b = data.get("a"), data.get("b")
    if not hasattr(wp, "merge_keyframes"):
        return jsonify({"ok": False, "error": "merge missing"})
    return jsonify(wp.merge_keyframes(a, b))


@app.route("/api/path_window", methods=["POST"])
def api_path_window():
    data = request.get_json(silent=True) or {}
    on = bool(data.get("on"))
    try:
        radius = max(0.0, float(data.get("radius", 1.0)))
    except Exception:
        radius = 1.0
    t = float(getattr(viewer_main, "playhead_t", 0.0) or 0.0)
    try:
        if data.get("t") is not None:
            t = float(data.get("t"))
    except Exception:
        pass
    try:
        if rr.sim_status().get("active") and data.get("t") is None:
            t = float(rr.sim_status().get("t") if rr.sim_status().get("t") is not None else t)
    except Exception:
        pass
    try:
        if hasattr(rr, "real_path_status") and rr.real_path_status().get("active") and data.get("t") is None:
            t = float(rr.real_path_status().get("t") if rr.real_path_status().get("t") is not None else t)
    except Exception:
        pass
    if hasattr(wp, "configure_path_window"):
        wp.configure_path_window(on, radius, t=t)
    elif on and hasattr(wp, "set_path_window"):
        wp.set_path_window(t - radius, t + radius)
    elif hasattr(wp, "set_path_window"):
        wp.set_path_window(None, None)
    return jsonify({"ok": True, "on": on, "radius": radius, "t": t})


@app.route("/api/camera_state", methods=["GET"])
def api_camera_state():
    if hasattr(rr, "camera_view_angles"):
        return jsonify({"ok": True, **rr.camera_view_angles()})
    return jsonify({"ok": False, "yaw": -32, "pitch": 28})


@app.route("/api/camera", methods=["POST"])
def api_camera():
    data = request.get_json(silent=True) or {}
    view = data.get("view") or data.get("name") or "front"
    if hasattr(rr, "apply_camera_view"):
        return jsonify(rr.apply_camera_view(view))
    return jsonify({"ok": False, "error": "no camera"}), 500


def _jsonable(obj):
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    try:
        return [float(v) for v in list(obj)]
    except Exception:
        try:
            return float(obj)
        except Exception:
            return str(obj)


def _predicted_lin(t):
    recs = []
    try:
        if hasattr(bake, "_iter_motion_samples"):
            recs = bake._iter_motion_samples() or []
    except Exception:
        recs = []
    if len(recs) < 2:
        return None
    t = float(t or 0.0)
    i = 0
    while i < len(recs) - 1:
        a = recs[i]
        b = recs[i + 1]
        nxt = recs[i + 2] if i + 2 < len(recs) else None
        if b.get("is_waypoint") and nxt is not None and not nxt.get("is_waypoint"):
            b = nxt
            step = 2
        else:
            step = 1
        t0 = float(a.get("t") or 0.0)
        t1 = float(b.get("t") or 0.0)
        if t < t0 - 1e-9:
            break
        if t <= t1 + 1e-9:
            dt = max(1e-6, t1 - t0)
            pa = a.get("pos") or [0, 0, 0]
            pb = b.get("pos") or [0, 0, 0]
            dist = sum((float(pb[j]) - float(pa[j])) ** 2 for j in range(3)) ** 0.5
            return dist / dt
        i += step
    return None


@app.route("/api/clock")
def api_clock():
    try:
        if hasattr(rr, "tick_afterburner") and (
            (rr.real_path_status() or {}).get("active")
        ):
            rr.tick_afterburner()
    except Exception:
        pass
    sim = rr.sim_status() if hasattr(rr, "sim_status") else {}
    real = rr.real_path_status() if hasattr(rr, "real_path_status") else {}
    t = getattr(viewer_main, "playhead_t", 0.0)
    try:
        t = float(t or 0.0)
    except Exception:
        t = 0.0
    if sim.get("active"):
        try:
            t = float(sim.get("t"))
        except Exception:
            t = t
    elif real.get("active"):
        try:
            t = float(real.get("t"))
        except Exception:
            t = t
    try:
        viewer_main.playhead_t = float(t)
    except Exception:
        pass
    lin_sim = 0.0
    lin_real = 0.0
    try:
        lin_sim = float((rr._motion.get("sim") or {}).get("lin_mps") or sim.get("lin_mps") or 0.0)
        lin_real = float((rr._motion.get("real") or {}).get("lin_mps") or 0.0)
    except Exception:
        pass
    return jsonify({"ok": True, "t": t, "sim": sim, "real": real, "lin_sim": lin_sim, "lin_real": lin_real})


@app.route("/api/playhead", methods=["POST"])
def api_playhead():
    data = request.get_json(silent=True) or {}
    try:
        viewer_main.playhead_t = float(data.get("t", 0.0))
    except Exception:
        viewer_main.playhead_t = 0.0
    if hasattr(rr, "sim_status") and rr.sim_status().get("active") and rr.sim_status().get("paused"):
        rr.set_sim_time(viewer_main.playhead_t)
    return jsonify({"ok": True, "t": viewer_main.playhead_t})


@app.route("/api/delete_selected", methods=["POST"])
def api_delete_selected():
    if _edits_locked():
        return jsonify({"ok": False, "error": "locked during playback"}), 409
    if getattr(wp, "server", None) is None:
        return jsonify({"ok": False, "error": "viewer not ready"}), 503
    viewer_main.delete_requests.append(True)
    return jsonify({"ok": True, "queued": True})


@app.route("/api/select_sim", methods=["POST"])
def api_select_sim():
    try:
        if hasattr(rr, "show_sim_gizmo"):
            rr.show_sim_gizmo()
        elif hasattr(rr, "_toggle_sim_gizmo"):
            rr._toggle_sim_gizmo()
    except Exception:
        pass
    return jsonify({"ok": True})


@app.route("/api/vis", methods=["POST"])
def api_vis():
    data = request.get_json(silent=True) or {}
    kind = data.get("kind")
    oid = data.get("id")
    on = data.get("on")
    if hasattr(wp, "set_item_visible"):
        wp.set_item_visible(kind, oid, on)
    return jsonify({"ok": True})


def _speed_export_path():
    return ROOT / "speed_export.json"


@app.route("/api/speed_export", methods=["GET", "POST"])
def api_speed_export():
    dest = _speed_export_path()
    if request.method == "GET":
        if not dest.exists():
            return jsonify({"ok": False, "error": "no export yet"}), 404
        from flask import send_file
        return send_file(str(dest), as_attachment=True, download_name="speed_export.json", mimetype="application/json")
    body = request.get_json(silent=True) or {}
    series = {"sim": [], "real": []}
    try:
        if hasattr(rr, "speed_series"):
            series = rr.speed_series() or series
    except Exception:
        pass
    if not series.get("sim") and body.get("sim"):
        series["sim"] = body.get("sim")
    if not series.get("real") and body.get("real"):
        series["real"] = body.get("real")
    recs = []
    try:
        recs = bake._iter_motion_samples() if hasattr(bake, "_iter_motion_samples") else []
    except Exception:
        recs = []
    hops_slim = []
    for hop in (getattr(bake, "baked_hops", None) or []):
        hops_slim.append({
            "t0": hop.get("t0"),
            "t1": hop.get("t1"),
            "dt": hop.get("dt"),
            "speed": hop.get("speed"),
            "lin_m": hop.get("lin_m"),
            "acc": hop.get("acc"),
            "over": hop.get("over"),
            "over_vel": hop.get("over_vel"),
            "over_acc": hop.get("over_acc"),
            "status": hop.get("status"),
        })
    payload = {
        "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "series": {
            "sim": series.get("sim") or [],
            "real": series.get("real") or [],
        },
        "motion": _jsonable({
            "sim": rr._motion.get("sim") if hasattr(rr, "_motion") else {},
            "real": rr._motion.get("real") if hasattr(rr, "_motion") else {},
        }),
        "hops": _jsonable(hops_slim),
        "samples": _jsonable([
            {"t": r.get("t"), "q": r.get("q"), "pos": r.get("pos"), "is_waypoint": r.get("is_waypoint"), "wp_index": r.get("wp_index")}
            for r in recs
        ]),
        "waypoints": _jsonable([
            {
                "i": i,
                "t": w.get("t"),
                "q": None if w.get("q") is None else [float(v) for v in list(w.get("q"))[:6]],
                "pos": None if w.get("pos") is None else [float(v) for v in list(w.get("pos"))[:3]],
                "wxyz": None if w.get("wxyz") is None else [float(v) for v in list(w.get("wxyz"))[:4]],
                "force_fk": bool(w.get("force_fk")),
            }
            for i, w in enumerate(wp.waypoints or [])
        ]),
    }
    done = threading.Event()
    err = []

    def _write():
        try:
            dest.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        except Exception as e:
            err.append(str(e))
        finally:
            done.set()

    threading.Thread(target=_write, name="speed-export", daemon=True).start()
    if not done.wait(12):
        return jsonify({"ok": False, "error": "export write timeout"}), 504
    if err:
        return jsonify({"ok": False, "error": err[0]}), 500
    return jsonify({"ok": True, "path": str(dest), "url": "/api/speed_export"})


@app.route("/api/select_object", methods=["POST"])
def api_select_object():
    data = request.get_json(silent=True) or {}
    oid = data.get("id")
    mode = data.get("mode") or "replace"
    if data.get("additive") or data.get("ctrl"):
        mode = "add"
    if data.get("range") or data.get("shift"):
        mode = "range"
    if hasattr(wp, "set_select_mods"):
        wp.set_select_mods(ctrl=data.get("ctrl"), shift=data.get("shift"))
    ok = wp.select_scene_object(oid, mode=mode) if hasattr(wp, "select_scene_object") else False
    return jsonify({"ok": bool(ok), "ids": list(getattr(wp, "selected_obj_ids", []) or [])})


@app.route("/api/select_pose", methods=["POST"])
def api_select_pose():
    data = request.get_json(silent=True) or {}
    pid = data.get("id")
    mode = data.get("mode") or "replace"
    if data.get("ctrl"):
        mode = "add"
    if data.get("shift"):
        mode = "range"
    if hasattr(wp, "set_select_mods"):
        wp.set_select_mods(ctrl=data.get("ctrl"), shift=data.get("shift"))
    ok = wp.select_item("pose", pid, mode=mode) if hasattr(wp, "select_item") else False
    return jsonify({"ok": bool(ok)})


@app.route("/api/select_mods", methods=["POST"])
def api_select_mods():
    data = request.get_json(silent=True) or {}
    if hasattr(wp, "set_select_mods"):
        wp.set_select_mods(ctrl=data.get("ctrl"), shift=data.get("shift"))
    return jsonify({"ok": True})


@app.route("/api/select_waypoint", methods=["POST"])
def api_select_waypoint():
    data = request.get_json(silent=True) or {}
    raw = data.get("index", None)
    if raw is None or raw == "" or str(raw).lower() == "none":
        try:
            wp.deselect_all()
        except Exception:
            traceback.print_exc()
        viewer_main.select_requests.append(None)
        return jsonify({"ok": True, "queued": True, "index": None})
    try:
        idx = int(raw)
    except Exception:
        return jsonify({"ok": False, "error": "bad index"}), 400
    mode = data.get("mode") or "replace"
    if data.get("ctrl"):
        mode = "add"
    if data.get("shift"):
        mode = "range"
    if data.get("from_timeline"):
        mode = "replace"
    viewer_main.select_requests.append({"index": idx, "mode": mode})
    return jsonify({"ok": True, "queued": True, "index": idx, "mode": mode})


@app.route("/api/console")
def api_console():
    try:
        after = int(request.args.get("after") or 0)
    except Exception:
        after = 0
    with _console_lock:
        rows = [list(item) for item in _console_lines if item[0] > after]
        last = _console_seq
    return jsonify({"ok": True, "lines": rows, "after": last})


def _status_keyframes(tl):
    kfs = list((tl or {}).get("keyframes") or [])
    if not kfs and hasattr(wp, "export_keyframes"):
        kfs = wp.export_keyframes()
    for i, kf in enumerate(kfs):
        if i < len(wp.waypoints):
            tw = wp.waypoints[i].get("t")
            kf["t"] = None if tw is None else float(tw)
            kf["index"] = i
    return kfs


_status_lock = threading.Lock()
_status_heavy = {}
_status_worker_on = False


def _build_status_heavy():
    st = bake.status_dict() if hasattr(bake, "status_dict") else {}
    try:
        tl = bake.timeline_payload() if hasattr(bake, "timeline_payload") else {}
    except Exception:
        traceback.print_exc()
        tl = {}
    return {
        "bake": st,
        "timeline": tl,
        "keyframes": _status_keyframes(tl),
        "samples": tl.get("samples") or [],
        "spans": tl.get("spans") or [],
        "runnable": tl.get("runnable", False),
        "sim_runnable": tl.get("sim_runnable", False),
        "camera": rr.camera_view_angles() if hasattr(rr, "camera_view_angles") else {},
        "object": rr.get_selected_props() if hasattr(rr, "get_selected_props") else {},
        "clipboard": bake.clipboard_info(getattr(viewer_main, "playhead_t", 0.0)) if hasattr(bake, "clipboard_info") else {},
        "scene_objects": _scene_objects_payload(),
        "cam_view": _cam_view_payload(),
        "measure": wp.measure_selected_objects() if hasattr(wp, "measure_selected_objects") else None,
        "tracks": list(getattr(wp, "extra_tracks", []) or []),
        "sim_poses": _sim_poses_payload(),
    }


def _status_worker():
    global _status_heavy
    try:
        if hasattr(wp, "tick_cameras"):
            wp.tick_cameras()
    except Exception:
        pass
    while True:
        try:
            blob = _build_status_heavy()
            with _status_lock:
                _status_heavy = blob
        except Exception:
            traceback.print_exc()
        time.sleep(0.25)


def _ensure_status_worker():
    global _status_worker_on
    if _status_worker_on:
        return
    _status_worker_on = True
    threading.Thread(target=_status_worker, name="status-heavy", daemon=True).start()


@app.route("/api/status")
def api_status():
    _ensure_status_worker()
    with _status_lock:
        heavy = dict(_status_heavy)
    out = {
        "ok": True,
        "viewer_ready": getattr(wp, "server", None) is not None,
        "viser_port": viewer_main.VISER_PORT,
        "waypoint_count": len(wp.waypoints),
        "selected": None if not wp.waypoints else wp.selected_idx,
        "selected_sample": getattr(bake, "selected_sample", None),
        "mode": wp.get_mode() if hasattr(wp, "get_mode") else "arc",
        "bake": heavy.get("bake") or {},
        "timeline": heavy.get("timeline") or {},
        "keyframes": heavy.get("keyframes") or [],
        "samples": heavy.get("samples") or [],
        "spans": heavy.get("spans") or [],
        "runnable": heavy.get("runnable", False),
        "sim_runnable": heavy.get("sim_runnable", False),
        "camera": heavy.get("camera") or {},
        "sim": rr.sim_status() if hasattr(rr, "sim_status") else {},
        "real": rr.real_path_status() if hasattr(rr, "real_path_status") else {},
        "allow_multiple_waypoints_per_keyframe": getattr(
            wp, "allow_multiple_waypoints_per_keyframe", True
        ),
        "export_path": getattr(bake, "export_path", True),
        "gizmo_space": getattr(wp, "gizmo_space", "local"),
        "object": heavy.get("object") or {},
        "move": rr.hold_move_status() if hasattr(rr, "hold_move_status") else {},
        "clipboard": heavy.get("clipboard") or {},
        "playhead": float(getattr(viewer_main, "playhead_t", 0.0) or 0.0),
        "scene_objects": heavy.get("scene_objects") or [],
        "cam_view": heavy.get("cam_view") or {},
        "selected_obj_ids": list(getattr(wp, "selected_obj_ids", []) or []),
        "selected_wp_idxs": list(getattr(wp, "selected_wp_idxs", []) or []),
        "selected_pose_ids": list(getattr(wp, "selected_pose_ids", []) or []),
        "measure": heavy.get("measure"),
        "pose_markers": bool(getattr(wp, "show_pose_markers", True)),
        "list_vis": dict(getattr(wp, "list_vis", None) or {"obj": True, "wp": True, "pose": True}),
        "item_vis": dict(getattr(wp, "item_vis", None) or {}),
        "tracks": heavy.get("tracks") or [],
        "sim_poses": heavy.get("sim_poses") or [],
        "merged_points": wp.list_merged_points() if hasattr(wp, "list_merged_points") else [],
        "point_sort": getattr(wp, "point_sort", "type"),
        "snap_to_grid": bool((load_settings() or {}).get("snap_to_grid", True)),
    }
    return jsonify(out)


def _q6(vals):
    out = []
    for v in list(vals or [])[:6]:
        try:
            out.append(float(v))
        except Exception:
            out.append(0.0)
    while len(out) < 6:
        out.append(0.0)
    return out


def _xyz(v):
    try:
        import numpy as np
        a = np.asarray(v, dtype=float).reshape(-1)
        if a.size == 0:
            return [0.0, 0.0, 0.0]
        out = [float(a[i]) if i < a.size else 0.0 for i in range(3)]
        return out
    except Exception:
        return [0.0, 0.0, 0.0]


def _cam_view_payload():
    try:
        if hasattr(wp, "camera_status"):
            return wp.camera_status() or {"active": False}
    except Exception as e:
        print("camera_status failed", e)
    return {"active": False}


def _scene_objects_payload():
    rows = []
    for o in list(getattr(wp, "scene_objects", None) or []):
        if not isinstance(o, dict):
            continue
        cam = o.get("camera") or {}
        oid = o.get("id")
        has = False
        if o.get("kind") == "camera" and hasattr(wp, "_cam_frame_jpeg"):
            has = bool(wp._cam_frame_jpeg.get(oid))
        rows.append({
            "id": oid,
            "name": o.get("name"),
            "kind": o.get("kind"),
            "parent_id": o.get("parent_id"),
            "pos": _xyz(o.get("pos")),
            "cam_type": str(cam.get("type") or "sim") if o.get("kind") == "camera" else None,
            "has_frame": has,
            "cam_error": str(cam.get("error") or "") if o.get("kind") == "camera" else "",
        })
    return rows


def _sim_poses_payload():
    if hasattr(wp, "ensure_builtin_sim_poses"):
        wp.ensure_builtin_sim_poses()
    rows = []
    real_q = rr.current_joint_q("real") if hasattr(rr, "current_joint_q") else None
    home_deg = list(getattr(rr, "HOME_Q_DEG", [0.0, 0.0, 90.0, 0.0, 90.0, 0.0]))
    for p in list(getattr(wp, "sim_poses", None) or []):
        if not isinstance(p, dict):
            continue
        pid = str(p.get("id") or "")
        rec = {
            "id": pid,
            "name": str(p.get("name") or ""),
            "builtin": bool(p.get("builtin") or pid in ("real", "home")),
            "ik": _q6(p.get("ik")),
            "q": _q6(p.get("q")),
        }
        if pid == "real" and real_q:
            rec["q"] = _q6(real_q)
        elif pid == "home":
            rec["q_deg"] = _q6(p.get("q_deg") or home_deg)
            rec["q"] = [v * 3.141592653589793 / 180.0 for v in rec["q_deg"]]
        elif p.get("q_deg"):
            rec["q_deg"] = _q6(p.get("q_deg"))
        rows.append(rec)
    if not any(r["id"] == "real" for r in rows) or not any(r["id"] == "home" for r in rows):
        if hasattr(wp, "ensure_builtin_sim_poses"):
            wp.ensure_builtin_sim_poses()
        return _sim_poses_payload() if rows else [
            {"id": "real", "name": "Real pose", "builtin": True, "ik": [0]*6, "q": _q6(real_q)},
            {"id": "home", "name": "Home pose", "builtin": True, "q_deg": _q6(home_deg),
             "q": [v * 3.141592653589793 / 180.0 for v in _q6(home_deg)], "ik": [0]*6},
        ]
    return rows


@app.route("/api/move_hold", methods=["POST"])
def api_move_hold():
    data = dict(request.get_json(silent=True) or {})
    mode = str(data.get("mode") or "fk").lower()
    q = data.get("q")
    if mode != "ik" and (q is None or not isinstance(q, (list, tuple)) or len(q) < 6):
        try:
            q = rr._flat(rr.robot_sim.angle_vector(), 6) if rr.robot_sim is not None else None
        except Exception:
            q = None
        if q is not None:
            data["q"] = q
        data["mode"] = "fk"
        data.pop("ik", None)
    viewer_main.move_requests.append(("start", data))
    return jsonify({"ok": True})


@app.route("/api/move_release", methods=["POST"])
def api_move_release():
    viewer_main.move_requests.append(("stop", None))
    st = rr.hold_move_status() if hasattr(rr, "hold_move_status") else {}
    return jsonify({"ok": True, "arrived": bool(st.get("arrived"))})


@app.route("/api/view_samples", methods=["GET", "POST"])
def api_view_samples():
    if request.method == "POST":
        data = request.get_json(silent=True) or {}
        if "viewer" in data:
            bake.show_viewer_samples = bool(data.get("viewer"))
        if "timeline" in data:
            bake.show_tl_samples = bool(data.get("timeline"))
        try:
            bake._draw_sample_dots()
        except Exception:
            pass
        try:
            bake.bump_timeline_cache()
        except Exception:
            pass
    return jsonify({
        "ok": True,
        "viewer": bool(getattr(bake, "show_viewer_samples", True)),
        "timeline": bool(getattr(bake, "show_tl_samples", True)),
    })


@app.route("/api/pose_markers", methods=["POST"])
def api_pose_markers():
    data = request.get_json(silent=True) or {}
    on = bool(data.get("on", True))
    shown = wp.set_show_pose_markers(on) if hasattr(wp, "set_show_pose_markers") else on
    return jsonify({"ok": True, "on": bool(shown)})


@app.route("/api/list_vis", methods=["POST"])
def api_list_vis():
    data = request.get_json(silent=True) or {}
    kind = data.get("kind")
    on = data.get("on")
    shown = wp.set_list_visible(kind, on) if hasattr(wp, "set_list_visible") else True
    return jsonify({"ok": True, "on": bool(shown), "list_vis": dict(getattr(wp, "list_vis", {}) or {})})


@app.route("/api/hide_robot", methods=["GET", "POST"])
def api_hide_robot():
    if request.method == "GET":
        st = rr.hide_real_robot_status() if hasattr(rr, "hide_real_robot_status") else {}
        return jsonify({"ok": True, "hidden": bool(st.get("hidden"))})
    data = request.get_json(silent=True) or {}
    hide = bool(data.get("hide", True))
    vis = rr.set_hide_real_robot(hide) if hasattr(rr, "set_hide_real_robot") else True
    return jsonify({"ok": True, "hidden": not vis})


@app.route("/api/track_object", methods=["GET", "POST"])
def api_track_object():
    if request.method == "GET":
        st = rr.track_status() if hasattr(rr, "track_status") else {}
        return jsonify({"ok": True, "track": st})
    data = request.get_json(silent=True) or {}
    if data.get("pick"):
        st = rr.begin_track_pick() if hasattr(rr, "begin_track_pick") else {}
        return jsonify({"ok": True, "track": st})
    idx = data.get("index")
    if idx is None:
        idx = getattr(wp, "selected_idx", None)
    kind = getattr(wp, "selection_kind", None)
    if kind == "waypoint" and idx is not None and 0 <= int(idx) < len(wp.waypoints):
        w = wp.waypoints[int(idx)]
        rec = dict(w.get("track_object") or {})
        if data.get("on") is not None:
            rec["on"] = bool(data.get("on"))
        if data.get("name") is not None:
            rec["name"] = str(data.get("name"))
        if data.get("offset") is not None:
            rec["offset"] = list(data.get("offset"))
        rec["added"] = True
        w["track_object"] = rec
        st = dict(rec)
        if hasattr(rr, "_track_target_pos"):
            tgt = rr._track_target_pos(rec.get("name"))
            st["valid"] = tgt is not None
            st["pos"] = None if tgt is None else [float(v) for v in tgt]
        return jsonify({"ok": True, "track": st})
    st = rr.set_track_object(
        on=data.get("on"),
        name=data.get("name"),
        pick=data.get("picking"),
        added=data.get("added"),
        offset=data.get("offset"),
    ) if hasattr(rr, "set_track_object") else {}
    return jsonify({"ok": True, "track": st})


@app.route("/api/add_object", methods=["POST"])
def api_add_object():
    data = request.get_json(silent=True) or {}
    kind = data.get("kind") or data.get("type") or "empty"
    if hasattr(wp, "add_scene_object"):
        obj = wp.add_scene_object(kind)
        return jsonify({"ok": True, "id": obj.get("id"), "kind": obj.get("kind")})


@app.route("/api/cam_script", methods=["POST"])
def api_cam_script():
    f = request.files.get("file")
    oid = request.form.get("id") or ""
    if f is None or not f.filename:
        return jsonify({"ok": False, "error": "no file"}), 400
    dest_dir = ROOT / "cam_scripts"
    dest_dir.mkdir(exist_ok=True)
    safe = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in f.filename)
    dest = dest_dir / (("%s_" % oid if oid else "") + safe)
    f.save(str(dest))
    path = str(dest)
    if oid and hasattr(wp, "update_scene_object"):
        wp.update_scene_object(oid, camera={"script": path, "type": "opencv"})
    return jsonify({"ok": True, "path": path})


@app.route("/api/reparent", methods=["POST"])
def api_reparent():
    data = request.get_json(silent=True) or {}
    ok = wp.reparent_object(data.get("id"), data.get("parent_id")) if hasattr(wp, "reparent_object") else False
    return jsonify({"ok": bool(ok)})


@app.route("/api/cam_click", methods=["POST"])
def api_cam_click():
    data = request.get_json(silent=True) or {}
    oid = data.get("id")
    try:
        u = float(data.get("u"))
        v = float(data.get("v"))
    except Exception:
        return jsonify({"ok": False, "error": "u,v required"}), 400
    ok = wp.set_camera_click(oid, u, v) if hasattr(wp, "set_camera_click") else False
    return jsonify({"ok": bool(ok), "u": u, "v": v})


@app.route("/api/cam_frame")
def api_cam_frame():
    oid = request.args.get("id")
    blob = None
    if hasattr(wp, "_cam_frame_jpeg"):
        if oid:
            blob = wp._cam_frame_jpeg.get(oid)
        elif wp._cam_frame_jpeg:
            blob = next(iter(wp._cam_frame_jpeg.values()))
    if not blob:
        return ("", 204)
    return Response(blob, mimetype="image/jpeg")


@app.route("/api/cam_template")
def api_cam_template():
    text = (
        "# camera CV template — expose get_frame() and tracked_cam_pos\\n"
        "tracked_cam_pos = []  # [x_px, y_px, dist_m], updated each frame\\n\\n"
        "def get_frame():\\n"
        "    import numpy as np\\n"
        "    h, w = 480, 640\\n"
        "    frame = np.zeros((h, w, 3), dtype=np.uint8)\\n"
        "    # draw / process, then optionally:\\n"
        "    # tracked_cam_pos[:] = [[w/2, h/2, 1.0]]\\n"
        "    return frame\\n"
    )
    return Response(text.replace("\\\\n", "\\n"), mimetype="text/x-python")
    return jsonify({"ok": False, "error": "add_scene_object missing"}), 500


@app.route("/api/object_props", methods=["POST"])
def api_object_props():
    if _edits_locked():
        return jsonify({"ok": False, "error": "locked during playback"}), 409
    data = request.get_json(silent=True) or {}
    oid = data.get("id") or getattr(wp, "selected_obj_id", None)
    scene_update = (
        data.get("kind") in ("empty", "sphere", "cube", "camera")
        or data.get("id")
        or (oid and any(k in data for k in ("space", "pos", "wxyz", "scale", "name", "camera")))
    )
    if scene_update:
        ok = wp.update_scene_object(
            oid,
            name=data.get("name"),
            pos=data.get("pos"),
            wxyz=data.get("wxyz"),
            scale=data.get("scale"),
            space=data.get("space"),
            camera=data.get("camera"),
        ) if hasattr(wp, "update_scene_object") else False
        return jsonify({"ok": bool(ok)})
    if "ik" in data:
        viewer_main.prop_requests.append(("ik", data["ik"]))
    elif "q" in data:
        q = [float(v) for v in data["q"][:6]]
        if str(data.get("units", "rad")).lower().startswith("deg"):
            q = [v * 3.141592653589793 / 180.0 for v in q]
        viewer_main.prop_requests.append(("q", q))
    else:
        return jsonify({"ok": False, "error": "need ik or q"}), 400
    return jsonify({"ok": True, "queued": True})


@app.route("/api/undo", methods=["POST"])
def api_undo():
    viewer_main.file_requests.append(("undo", None))
    return jsonify({"ok": True})


@app.route("/api/redo", methods=["POST"])
def api_redo():
    viewer_main.file_requests.append(("redo", None))
    return jsonify({"ok": True})


@app.route("/api/tracks", methods=["POST"])
def api_tracks():
    data = request.get_json(silent=True) or {}
    rows = data.get("tracks") or []
    cleaned = []
    if isinstance(rows, list):
        for i, tr in enumerate(rows):
            if isinstance(tr, dict):
                rec = {"name": str(tr.get("name") or ("Track %d" % (i + 1)))}
                if tr.get("id"):
                    rec["id"] = str(tr.get("id"))
                if isinstance(tr.get("keys"), list):
                    rec["keys"] = tr.get("keys")
                cleaned.append(rec)
            else:
                cleaned.append({"name": str(tr)})
    wp.extra_tracks = cleaned
    return jsonify({"ok": True, "tracks": cleaned})


@app.route("/api/sony_rec", methods=["POST", "GET"])
def api_sony_rec():
    """POST {on: true/false} fires in-camera REC. GET returns debug."""
    import sony_liveview
    data = request.get_json(silent=True) or {}
    if request.method == "POST" and "on" in data:
        ok = bool(sony_liveview.set_recording(bool(data.get("on"))))
        st = sony_liveview.status()
        return jsonify({"ok": ok, "sony": st})
    keys = []
    rec_want = False
    oid = None
    try:
        cams = [o for o in wp.scene_objects if o.get("kind") == "camera"]
        obj = next((o for o in cams if str((o.get("camera") or {}).get("type")) == "sony"), None)
        if obj is None and cams:
            obj = cams[0]
        if obj is not None:
            oid = obj.get("id")
            rec_want = bool(wp.camera_record_at(oid)) if hasattr(wp, "camera_record_at") else False
            tid = "cam_" + str(oid)
            for tr in list(getattr(wp, "extra_tracks", []) or []):
                if isinstance(tr, dict) and str(tr.get("id") or "") == tid:
                    keys = list(tr.get("keys") or [])
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)})
    return jsonify({
        "ok": True,
        "oid": oid,
        "rec_want": rec_want,
        "real": bool(wp._real_playing()) if hasattr(wp, "_real_playing") else False,
        "t": float(wp._playhead_t()) if hasattr(wp, "_playhead_t") else 0,
        "keys": keys,
        "sony": sony_liveview.status(),
    })


@app.route("/api/file/new", methods=["POST"])
def api_file_new():
    viewer_main.file_requests.append(("new", None))
    return jsonify({"ok": True})


@app.route("/api/file/save", methods=["POST"])
def api_file_save():
    try:
        data = request.get_json(silent=True) or {}
        title = data.get("title")
        tracks = data.get("tracks") if isinstance(data.get("tracks"), list) else None
        if tracks is not None:
            cleaned = []
            for i, tr in enumerate(tracks):
                if isinstance(tr, dict):
                    rec = {"name": str(tr.get("name") or ("Track %d" % (i + 1)))}
                    if tr.get("id"):
                        rec["id"] = str(tr.get("id"))
                    if isinstance(tr.get("keys"), list):
                        rec["keys"] = tr.get("keys")
                    cleaned.append(rec)
            wp.extra_tracks = cleaned
        result = fman.write_save_file(title=title, tracks=tracks)
        return jsonify(result)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/api/file/export", methods=["POST"])
def api_file_export():
    try:
        data = request.get_json(silent=True) or {}
        title = data.get("title")
        result = fman.write_export_json(title=title)
        return jsonify(result)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/api/file/open", methods=["POST"])
def api_file_open():
    data = request.get_json(silent=True) or {}
    doc = data.get("doc", data)
    viewer_main.file_requests.append(("load", doc))
    return jsonify({"ok": True, "queued": True})


@app.route("/api/export", methods=["POST"])
def api_export():
    return api_file_export()


def _open_ui():
    time.sleep(2.0)
    webbrowser.open("http://127.0.0.1:5000")


if __name__ == "__main__":
    start_viewer_thread()
    _ensure_status_worker()
    print("Web UI   http://127.0.0.1:5000")
    print("Viser    http://127.0.0.1:%s (iframe only)" % (viewer_main.VISER_PORT or "?"))
    threading.Thread(target=_open_ui, daemon=True).start()
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False,
        use_reloader=False,
        threaded=True,
        request_handler=QuietHandler,
    )
