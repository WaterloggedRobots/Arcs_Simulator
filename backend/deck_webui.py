# version: v_beta10.06
"""Steam Deck edition: a stripped-down controller for the real arm.

Start with `python backend/deck_webui.py` (instead of webui.py). It runs the
same Flask app and Viser viewer, but "/" serves templates/deck.html: the 3D
viewer, saved points (left), sim / real IK and joints (right), Hold to move,
Axes Mode and Settings. No waypoints, bake, timeline, objects or components.

On start it switches Steam Deck Tools' SteamController.exe to its "Arcs Deck"
profile and switches back on exit (deck/steam_profile.py); the Deck's sticks
and back buttons then jog the sim cursor (deck/pad.py).
"""
from __future__ import annotations

import math
import threading
import time

import numpy as np
from flask import jsonify, render_template, request
from scipy.spatial.transform import Rotation as R

import webui
from webui import app

import arms
import robot_runtime as rr
import robot_sim_run as viewer_main
import waypoints as wp
from deck import pad, steam_profile

ROT_DPS_DEFAULT = 20.0
# While Hold to move is held, re-aim the real arm at the jogged sim pose this often.
FOLLOW_REAL_S = 0.25

_jog = {"rot_dps": ROT_DPS_DEFAULT, "ik_fail": False, "dirty": False, "last_follow": 0.0}


def _wxyz(rot):
    x, y, z, w = rot.as_quat()
    return np.array([w, x, y, z], dtype=float)


def _move_sim_cursor(lin, rot, dt):
    """Jog the sim cursor: lin in -1..1 of the linear speed setting, rot
    (roll, pitch, yaw) in -1..1 of the rotation speed. Local / Global follows
    Axes Mode. The step is dropped when IK has no solution."""
    pos, wxyz = rr.get_sim_cursor_pose()
    pos = np.asarray(pos, dtype=float).reshape(3)
    r0 = R.from_quat(rr.wxyz_to_xyzw(wxyz))
    d = np.asarray(lin, dtype=float) * float(rr.HOLD_LIN_MPS) * dt
    rv = np.asarray(rot, dtype=float) * math.radians(_jog["rot_dps"]) * dt
    local = getattr(wp, "gizmo_space", "local") == "local"
    if local:
        pos1 = pos + r0.apply(d)
        r1 = r0 * R.from_rotvec(rv)
    else:
        pos1 = pos + d
        r1 = R.from_rotvec(rv) * r0
    wxyz1 = _wxyz(r1)
    ee_pos, ee_wxyz = rr._ee_from_cursor(pos1, wxyz1, "sim")
    q = rr._ik_to_pose(ee_pos, ee_wxyz)
    if q is None or rr.robot_sim is None:
        return False
    rr.robot_sim.angle_vector(q)
    rr._last_good_q = q
    rr._set_sim_heading(wxyz1)
    pin = tuple(float(v) for v in pos1)
    wp._suspend_gizmo_cb = True
    try:
        if rr._cursor_sim is not None:
            rr._cursor_sim.position = pin
        if rr._sim_gizmo is not None:
            rr._sim_gizmo.position = pin
            if local:
                # update_cursors takes the heading from a local gizmo each tick.
                rr._sim_gizmo.wxyz = tuple(float(v) for v in rr._sim_wxyz)
    finally:
        wp._suspend_gizmo_cb = False
    rr._draw_sim_dir(pos1)
    return True


def _follow_real(force=False):
    """Hold to move aims at the sim pose when pressed; keep it on the jogged pose."""
    if not getattr(rr, "_hold_move", False) or not _jog["dirty"]:
        return
    now = time.time()
    if not force and now - _jog["last_follow"] < FOLLOW_REAL_S:
        return
    _jog["last_follow"] = now
    _jog["dirty"] = False
    rr.start_hold_move(mode="fk")


def _jog_step(lin, rot, dt):
    with arms.LOCK:
        arms.ensure_active_loaded()
        if lin is None:
            # Sticks let go: send the real arm to where the sim cursor stopped.
            _jog["ik_fail"] = False
            _follow_real(force=True)
            return
        if rr.edits_locked() or not rr.ik_allowed():
            return
        ok = _move_sim_cursor(lin, rot, dt)
        _jog["ik_fail"] = not ok
        if ok:
            _jog["dirty"] = True
            _follow_real()


def _flat6(v):
    return [float(x) for x in np.asarray(v, dtype=float).reshape(-1)[:6]]


def _sim_payload():
    pos, wxyz = rr.get_sim_cursor_pose()
    ee, ee_wxyz = rr._ee_from_cursor(pos, wxyz, "sim")
    return {
        "ik": [float(v) for v in rr.aubo_pose_from_viewer(ee, ee_wxyz)],
        "q": _flat6(rr.robot_sim.angle_vector()) if rr.robot_sim is not None else [0.0] * 6,
        "lin_mps": float(rr._motion["sim"]["lin_mps"] or 0.0),
    }


def _real_payload():
    out = {"connected": bool(rr.robot_connected())}
    if out["connected"] and rr.robot_actual is not None:
        pos, wxyz = rr._ee_frame(rr.robot_actual)
        out["ik"] = [float(v) for v in rr.aubo_pose_from_viewer(pos, wxyz)]
        out["q"] = _flat6(rr.robot_actual.angle_vector())
        out["lin_mps"] = float(rr._motion["real"]["lin_mps"] or 0.0)
    return out


def _settings_payload():
    st = webui.load_settings()
    return {
        "robot_address": rr.robot_address(),
        "robot_model": getattr(rr, "_MODEL_ID", None) or st.get("robot_model") or "aubo_i5",
        "robot_models": st.get("robot_models") or webui.SETTINGS_DEFAULT["robot_models"],
        "hold_lin_mps": float(rr.HOLD_LIN_MPS),
        "deck_rot_dps": float(_jog["rot_dps"]),
    }


def deck_index():
    port = getattr(viewer_main, "VISER_PORT", None) or 8080
    return render_template("deck.html", viser_url="http://127.0.0.1:%s" % port)


app.view_functions["index"] = deck_index


@app.route("/api/deck/state")
def api_deck_state():
    try:
        sim = _sim_payload()
    except Exception as e:
        sim = {"error": str(e)}
    try:
        real = _real_payload()
    except Exception as e:
        real = {"connected": False, "error": str(e)}
    try:
        move = rr.hold_move_status() if real.get("connected") else {"holding": bool(getattr(rr, "_hold_move", False))}
    except Exception:
        move = {}
    return jsonify({
        "ok": True,
        "viewer_ready": bool(getattr(viewer_main, "VISER_PORT", None)),
        "sim": sim,
        "real": real,
        "move": move,
        "ik_fail": bool(_jog["ik_fail"]),
        "gizmo_space": getattr(wp, "gizmo_space", "local"),
        "points": webui._sim_poses_payload(),
        "pad": pad.status(),
        "steam_profile": steam_profile.status(),
    })


@app.route("/api/deck/settings", methods=["GET", "POST"])
def api_deck_settings():
    if request.method == "GET":
        return jsonify({"ok": True, "settings": _settings_payload()})
    data = request.get_json(silent=True) or {}
    save = {}
    if "robot_ip" in data or "robot_port" in data:
        ip = str(data.get("robot_ip") or "").strip()
        port = data.get("robot_port")
        save["robot_ip"] = ip
        save["robot_port"] = port
        rr.set_robot_address(ip, port)
    if data.get("hold_lin_mps") not in (None, ""):
        save["hold_lin_mps"] = rr.set_hold_lin_mps(data.get("hold_lin_mps"))
    if data.get("deck_rot_dps") not in (None, ""):
        try:
            _jog["rot_dps"] = max(0.1, float(data.get("deck_rot_dps")))
        except Exception:
            pass
        save["deck_rot_dps"] = _jog["rot_dps"]
    if save:
        webui.save_settings(save)
    mid = str(data.get("robot_model") or "").strip()
    if mid and mid != (getattr(rr, "_MODEL_ID", None) or ""):
        # Same as /api/robot_model; this edition has no waypoints to clear.
        webui.apply_settings(webui.save_settings({"robot_model": mid}))
        try:
            rr.reload_robot_models()
        except Exception as e:
            print("reload robots failed", e)
    return jsonify({"ok": True, "settings": _settings_payload()})


def _load_deck_settings():
    try:
        _jog["rot_dps"] = max(0.1, float(webui.load_settings().get("deck_rot_dps") or ROT_DPS_DEFAULT))
    except Exception:
        _jog["rot_dps"] = ROT_DPS_DEFAULT


if __name__ == "__main__":
    webui.apply_settings(webui.load_settings())
    _load_deck_settings()
    webui.start_viewer_thread()
    port = webui._wait_viewer()
    print("Viser    http://127.0.0.1:%s (iframe)" % port)
    steam_profile.activate()
    pad.start(_jog_step)
    print("Deck UI  http://127.0.0.1:5000")
    threading.Thread(target=webui._open_ui, daemon=True).start()
    try:
        app.run(host="127.0.0.1", port=5000, debug=False, use_reloader=False, threaded=True,
                request_handler=webui.QuietHandler)
    finally:
        steam_profile.restore()
