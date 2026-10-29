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
import os
import subprocess
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
# Left stick and L4 / L5 at full tilt, sim and real (Hold to move has its own speed).
LIN_MPS_DEFAULT = 0.1
LIN_MPS_MAX = 0.5
# While Hold to move is held, re-aim the real arm at the jogged sim pose this often.
FOLLOW_REAL_S = 0.25

_jog = {"rot_dps": ROT_DPS_DEFAULT, "lin_mps": LIN_MPS_DEFAULT, "ik_fail": False, "dirty": False, "last_follow": 0.0}


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
    d = np.asarray(lin, dtype=float) * float(_jog["lin_mps"]) * dt
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


# Which arm the sticks drive: "sim" moves only the sim cursor; "real" jogs the
# real arm directly, and snaps the sim to it when let go. Real jog commands:
#   servoJoint  servo mode stream: each pad tick moves a target TCP pose along
#               the stick twist and sends the controller's IK of it
#   speedLine   TCP velocity (RobotArcs.SpeedLine)
#   speedJoint  joint velocities toward the controller's IK of a pose a little
#               ahead on the same twist (RobotArcs.SpeedJoint)
#   auto        servoJoint, then speedLine if the arm does not move
# Every speed command restarts the controller's move, so speedLine and
# speedJoint are re-sent only when the sticks change (in steps of
# REAL_STEP) or every REAL_KEEP_S; sent each tick they jitter in place.
_target = {"mode": "sim", "error": "", "jogging": False}
_real = {"setting": "auto", "method": "servoJoint", "ret": None, "q0": None,
         "cmd": 0.0, "moved": False, "checked": False,
         "sent": None, "sent_t": 0.0, "pose": None, "q": None}
REAL_METHODS = ("auto", "servoJoint", "speedLine", "speedJoint")
REAL_SPEED_T = 1.0  # s the controller keeps a speed without a newer one
REAL_KEEP_S = 0.8   # re-send an unchanged speed this often
REAL_STEP = 0.1     # stick change (of full tilt) that re-sends a speed
REAL_ACC = 0.5      # m/s^2 (and rad/s^2) to reach / change the speed
REAL_JACC = 2.0     # rad/s^2 for speedJoint
REAL_LOOKAHEAD_S = 0.25
REAL_MAX_QD = 1.0   # rad/s, per joint, for speedJoint
# servoJoint: lookahead (s) and gain as in the AUBO servo examples; the
# target may lead the real TCP by at most REAL_SERVO_LEAD m.
REAL_SERVO_LOOKAHEAD = 0.1
REAL_SERVO_GAIN = 200
REAL_SERVO_LEAD = 0.03
# Commanded travel (m, or rad / 5) before deciding the arm is not following.
REAL_CHECK_AFTER = 0.01


def _set_target(mode):
    mode = "real" if str(mode).lower() == "real" else "sim"
    _target["error"] = ""
    if mode == "real":
        if not rr.robot_connected():
            _target["error"] = "robot not connected"
            return _target["mode"]
        # Start from where the real arm is, so the first move is not a jump.
        rr.reset_sim_to_real()
        _jog["dirty"] = False
    elif _target["mode"] == "real":
        _real_stop()
    _target["mode"] = mode
    print("deck: sticks drive the", mode, "bot")
    return mode


def _real_twist(lin, rot):
    """Stick values to a base-frame TCP velocity for speedLine. Local = the
    real TCP's axes (from its pose), Global = world axes."""
    v = np.asarray(lin, dtype=float) * float(_jog["lin_mps"])
    w = np.asarray(rot, dtype=float) * math.radians(_jog["rot_dps"])
    if getattr(wp, "gizmo_space", "local") == "local":
        pose = rr.robot.GetTcpPose()
        if pose is None:
            return None
        r = R.from_euler("xyz", [float(x) for x in list(pose)[3:6]])
    else:
        r = rr._base_rot().inv()
    return [float(x) for x in np.concatenate([r.apply(v), r.apply(w)])]


def _real_begin(bot):
    """First tick of a jog: un-pause the motion queue, note where the arm is,
    and for servoJoint enter servo mode with the target on the real TCP."""
    st = {}
    try:
        st = bot.PrepareJog() if hasattr(bot, "PrepareJog") else {}
    except Exception as e:
        print("real jog: prepare failed", e)
    q0 = bot.GetJoints()
    pose = bot.GetTcpPose()
    _real.update(ret=None, cmd=0.0, moved=False, checked=False, q0=q0,
                 sent=None, sent_t=0.0,
                 pose=[float(x) for x in list(pose)[:6]] if pose is not None else None,
                 q=[float(x) for x in list(q0)[:6]] if q0 is not None else None)
    if _real["method"] == "servoJoint":
        ok = bot.ServoMode(True) if hasattr(bot, "ServoMode") else False
        st["servo_on"] = ok
    print("real jog: start with", _real["method"], st)


def _ik_ahead(bot, pose, xd, dt, seed):
    """The controller's IK of pose moved dt along the base-frame twist xd."""
    p = np.asarray(pose[:3]) + np.asarray(xd[:3]) * dt
    r = R.from_rotvec(np.asarray(xd[3:6]) * dt) * R.from_euler("xyz", pose[3:6])
    target = [float(x) for x in list(p) + list(r.as_euler("xyz"))]
    q = bot.GetIK(target, seed=list(seed), verbose=False)
    return target, q


def _servo_joint(bot, xd, dt):
    """Move the servo target along the twist and stream its joints."""
    if _real["pose"] is None or _real["q"] is None:
        return "no robot state"
    t = max(0.01, min(0.1, dt))
    pose = _real["pose"]
    now = bot.GetTcpPose()
    lead = 0.0 if now is None else float(np.linalg.norm(np.asarray(pose[:3]) - np.asarray(list(now)[:3])))
    if lead < REAL_SERVO_LEAD:
        target, q = _ik_ahead(bot, pose, xd, t, _real["q"])
        if q is not None:
            _real["pose"], _real["q"] = target, [float(x) for x in q]
    return bot.ServoJoint(_real["q"], t, REAL_SERVO_LOOKAHEAD, REAL_SERVO_GAIN)


def _changed(cmd):
    """True when a speed command should be re-sent: the sticks moved a step,
    or REAL_KEEP_S passed (the controller drops it after REAL_SPEED_T)."""
    key = tuple(int(round(v / REAL_STEP)) for v in cmd)
    now = time.time()
    if key == _real["sent"] and now - _real["sent_t"] < REAL_KEEP_S:
        return False
    _real["sent"], _real["sent_t"] = key, now
    return True


def _speed_joint(bot, xd):
    """speedJoint toward the controller's IK of the TCP moved REAL_LOOKAHEAD_S
    along the twist xd (base frame)."""
    q = bot.GetJoints()
    pose = bot.GetTcpPose()
    if q is None or pose is None:
        return "no robot state"
    q = np.asarray(list(q)[:6], dtype=float)
    pose = [float(x) for x in list(pose)[:6]]
    la = REAL_LOOKAHEAD_S
    _, qt = _ik_ahead(bot, pose, xd, la, q)
    if qt is None:
        return "no IK ahead"
    dq = (np.asarray(qt, dtype=float) - q + math.pi) % (2.0 * math.pi) - math.pi
    qd = dq / la
    top = float(np.max(np.abs(qd)))
    if top > REAL_MAX_QD:
        qd *= REAL_MAX_QD / top
    return bot.SpeedJoint(list(qd), REAL_JACC, REAL_SPEED_T)


def _real_check(bot, xd, dt):
    """Once enough motion was asked for, check the TCP moved most of it; if
    not, say so with the controller state, and in auto try speedLine."""
    if _real["checked"]:
        return
    _real["cmd"] += (np.linalg.norm(xd[:3]) + np.linalg.norm(xd[3:6]) / 5.0) * dt
    if _real["cmd"] < REAL_CHECK_AFTER:
        return
    _real["checked"] = True
    q = bot.GetJoints()
    q0 = _real["q0"]
    if q is None or q0 is None:
        return
    moved = float(np.max(np.abs(np.asarray(list(q)[:6]) - np.asarray(list(q0)[:6]))))
    if moved > 0.002:
        _real["moved"] = True
        print("real jog: %s moves the arm (%.3f rad)" % (_real["method"], moved))
        if " did not move" in _target["error"]:
            _target["error"] = ""
        return
    st = bot.JogState() if hasattr(bot, "JogState") else {}
    print("real jog: %s sent (ret %s) but the arm did not move; controller %s"
          % (_real["method"], _real["ret"], st))
    if _real["setting"] == "auto" and _real["method"] == "servoJoint":
        _end_servo(bot)
        _real["method"] = "speedLine"
        _target["error"] = "servoJoint did not move the arm, trying speedLine"
        _real_begin(bot)
        return
    _target["error"] = "%s did not move the arm (ret %s): see the console" % (_real["method"], _real["ret"])


def _real_jog(lin, rot, dt=0.03):
    """One pad tick of Real control. RPCs run outside arms.LOCK."""
    bot = rr.robot
    if bot is None or not hasattr(bot, "SpeedLine"):
        return False
    if not _target["jogging"]:
        _target["jogging"] = True
        _real_begin(bot)
    xd = _real_twist(lin, rot)
    if xd is None:
        return False
    method = _real["method"]
    if method == "servoJoint":
        ret = _servo_joint(bot, xd, dt)
    elif not _changed(list(lin) + list(rot)):
        ret = _real["ret"]
    elif method == "speedJoint":
        ret = _speed_joint(bot, xd)
    else:
        ret = bot.SpeedLine(xd, REAL_ACC, REAL_SPEED_T)
    if ret not in (None, 0) and ret != _real["ret"]:
        print("real jog:", method, "returned", ret)
        _target["error"] = "%s returned %s" % (method, ret)
    _real["ret"] = ret
    _real_check(bot, xd, dt)
    return True


def _end_servo(bot):
    try:
        if hasattr(bot, "ServoMode"):
            bot.ServoMode(False)
    except Exception as e:
        print("real jog: servo off failed", e)


def _real_stop():
    if not _target["jogging"]:
        return
    _target["jogging"] = False
    bot = rr.robot
    if bot is None:
        return
    if _real["method"] == "servoJoint":
        # The arm stops on the last target; leave servo mode for other moves.
        _end_servo(bot)
        return
    try:
        if hasattr(bot, "StopSpeed"):
            bot.StopSpeed()
    except Exception as e:
        print("real jog stop failed", e)


def _follow_real(force=False):
    """Hold to move aims at the sim pose when pressed; keep it on the jogged pose."""
    if not _jog["dirty"] or not getattr(rr, "_hold_move", False):
        return
    now = time.time()
    if not force and now - _jog["last_follow"] < FOLLOW_REAL_S:
        return
    _jog["last_follow"] = now
    _jog["dirty"] = False
    rr.start_hold_move(mode="fk")


# Hold to move: the page posts {on: true} on press and again every 0.5 s while
# held; with no beat for HOLD_BEAT_S the move stops (a lost release must not
# leave the arm moving).
HOLD_BEAT_S = 1.5
_hold = {"beat": 0.0, "error": ""}


def _hold_start():
    if not rr.robot_connected():
        return "robot not connected"
    if rr.edits_locked():
        return "stop playback first"
    if not rr.start_hold_move(mode="fk"):
        return "the controller refused the move (see the console)"
    _jog["dirty"] = False
    return ""


def _hold_stop():
    _hold["beat"] = 0.0
    rr.stop_hold_move()


def _hold_watchdog():
    while True:
        time.sleep(0.2)
        try:
            with arms.LOCK:
                arms.ensure_active_loaded()
                if _hold["beat"] and time.time() - _hold["beat"] > HOLD_BEAT_S:
                    print("hold to move: no beat from the page, stopping")
                    _hold_stop()
        except Exception as e:
            print("hold watchdog failed", e)


def _jog_step(lin, rot, dt):
    if _target["mode"] == "real" or _target["jogging"]:
        _real_step(lin, rot, dt)
        return
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


def _real_step(lin, rot, dt=0.03):
    if lin is None or _target["mode"] != "real":
        _real_stop()
        with arms.LOCK:
            arms.ensure_active_loaded()
            if rr.robot_connected():
                # Show where the real arm stopped.
                rr.reset_sim_to_real()
        return
    if not rr.robot_connected():
        _real_stop()
        _target["mode"] = "sim"
        _target["error"] = "robot disconnected: back to Sim"
        return
    try:
        _real_jog(lin, rot, dt)
    except Exception as e:
        print("real jog failed", e)
        _real_stop()


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
        "deck_lin_mps": float(_jog["lin_mps"]),
        "deck_real_jog": _real["setting"],
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
        "target": _target["mode"],
        "target_error": _target["error"],
        "points": webui._sim_poses_payload(),
        "pad": pad.status(),
        "steam_profile": steam_profile.status(),
    })


@app.route("/api/deck/move", methods=["POST"])
def api_deck_move():
    """Hold to move, run here and not on the viewer queue, so the page hears
    whether the controller took the move."""
    data = request.get_json(silent=True) or {}
    if not data.get("on"):
        _hold_stop()
        return jsonify({"ok": True, "holding": False})
    first = not _hold["beat"] or not getattr(rr, "_hold_move", False)
    _hold["beat"] = time.time()
    if first:
        _hold["error"] = _hold_start()
        if _hold["error"]:
            _hold["beat"] = 0.0
            print("hold to move:", _hold["error"])
    return jsonify({"ok": not _hold["error"], "holding": bool(getattr(rr, "_hold_move", False)),
                    "error": _hold["error"]})


@app.route("/api/deck/target", methods=["POST"])
def api_deck_target():
    """Sticks drive the sim bot or the real bot: {target: "sim" | "real" | "toggle"}."""
    data = request.get_json(silent=True) or {}
    want = str(data.get("target") or "toggle")
    if want == "toggle":
        want = "sim" if _target["mode"] == "real" else "real"
    mode = _set_target(want)
    return jsonify({"ok": not _target["error"], "target": mode, "error": _target["error"]})


def _on_pad_button(name):
    if name == "X":
        with arms.LOCK:
            arms.ensure_active_loaded()
            _set_target("sim" if _target["mode"] == "real" else "real")


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
    if data.get("deck_lin_mps") not in (None, ""):
        try:
            _jog["lin_mps"] = min(LIN_MPS_MAX, max(0.001, float(data.get("deck_lin_mps"))))
        except Exception:
            pass
        save["deck_lin_mps"] = _jog["lin_mps"]
    if data.get("deck_real_jog") in REAL_METHODS:
        _set_real_method(data.get("deck_real_jog"))
        save["deck_real_jog"] = _real["setting"]
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


def _open_deck_ui(url="http://127.0.0.1:5000"):
    """Open the page as an app window (Edge / Chrome --app): no tabs, no Back,
    so a right-drag Back gesture cannot leave the page. Else the default browser."""
    time.sleep(2)
    if os.name == "nt":
        roots = [os.environ.get(k) for k in ("ProgramFiles(x86)", "ProgramFiles", "LOCALAPPDATA")]
        rels = [r"Microsoft\Edge\Application\msedge.exe", r"Google\Chrome\Application\chrome.exe"]
        for root in filter(None, roots):
            for rel in rels:
                exe = os.path.join(root, rel)
                if os.path.isfile(exe):
                    try:
                        subprocess.Popen([exe, "--app=" + url, "--start-maximized"])
                        return
                    except Exception as e:
                        print("app window failed", exe, e)
    webui._open_ui()


def _set_real_method(name):
    _real["setting"] = name if name in REAL_METHODS else "auto"
    _real["method"] = "servoJoint" if _real["setting"] == "auto" else _real["setting"]


def _load_deck_settings():
    st = webui.load_settings()
    try:
        _jog["rot_dps"] = max(0.1, float(st.get("deck_rot_dps") or ROT_DPS_DEFAULT))
    except Exception:
        _jog["rot_dps"] = ROT_DPS_DEFAULT
    try:
        _jog["lin_mps"] = min(LIN_MPS_MAX, max(0.001, float(st.get("deck_lin_mps") or LIN_MPS_DEFAULT)))
    except Exception:
        _jog["lin_mps"] = LIN_MPS_DEFAULT
    _set_real_method(st.get("deck_real_jog") or "auto")


if __name__ == "__main__":
    webui.apply_settings(webui.load_settings())
    _load_deck_settings()
    webui.start_viewer_thread()
    port = webui._wait_viewer()
    print("Viser    http://127.0.0.1:%s (iframe)" % port)
    steam_profile.activate()
    pad.start(_jog_step, _on_pad_button)
    threading.Thread(target=_hold_watchdog, daemon=True, name="deck-hold").start()
    print("Deck UI  http://127.0.0.1:5000")
    threading.Thread(target=_open_deck_ui, daemon=True).start()
    try:
        app.run(host="127.0.0.1", port=5000, debug=False, use_reloader=False, threaded=True,
                request_handler=webui.QuietHandler)
    finally:
        steam_profile.restore()
