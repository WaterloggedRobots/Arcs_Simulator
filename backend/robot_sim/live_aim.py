"""Live Look At (scenario 1): the path stays as baked, the aim follows the
Look At target where it is now.

`Aimer(t, q_base)` takes the baked joints at path time t, puts the Sim Arm
Cursor where they put it, aims it at the span's Look At target as it is now
(`rr._track_target_pos`: an object, the camera script's primary target, ...)
and solves the joints for that aim with the local IK (`rr.solve_backup_ik`,
its own model and lock, so any thread can call it). The aim change from the
baked rotation is a world rotation `corr`, limited to `MAX_DEV_DEG` and moved
toward its goal with a speed and acceleration limit and a deadband; outside a
Look At span it goes back to none the same way. The joints follow the IK
answers through a speed / acceleration-limited filter. An IK failure, a joint
past its limit or a jump bigger than `JUMP_MAX_DEG` from the last answer keeps
the last good answer.

Used by Simulate (`robot_runtime._apply_q_at_t`, on the viewer thread) and by
the streamed real run (`Streamer` here: trackJoint at `STREAM_DT`, on the run's
clock), both only while the Look At component's **Live** switch is on.
"""
from __future__ import annotations

import math
import sys
import threading
import time

import numpy as np
from scipy.spatial.transform import Rotation as R

MAX_DEV_DEG = 60.0      # largest turn away from the baked aim
JUMP_MAX_DEG = 4.0      # a joint jump bigger than this between IK answers is refused
STREAM_DT = 0.02        # streamed run: 50 points/s
IK_DEADBAND_DEG = 0.01  # IK answers changing less than this are noise

# Real Time Motion tuning (Edit > Preferences > Real Time Motion, saved in
# settings.json "live_tuning"; the Look At panel's smoothing fields edit the
# same values). Used by the live aim (Simulate and the streamed Run) and, for
# the target filter, by the Sim Arm Cursor's own Look At.
#   smooth_s / deadband_mm   the target: low-pass time constant, held while it moves less
#   predict_s                the target between its updates (a camera target changes at the
#                            camera's frame rate: steps): carried on at its speed this long
#   aim_deadband_deg         the aim's goal is held while the target's aim moves less than
#                            this, then follows it at that distance (no stop-start: the
#                            first version waited at rest for 0.4 deg, caught up and
#                            stopped again, so a moving target was followed in steps)
#   aim_response_s           the turn's speed is at most (what is left) / this (0 = none)
#   turn_dps / turn_acc      the aim's speed and acceleration limits
#   joint_smooth_s           the joints' filter time constant (0 = none)
#   joint_dps / joint_acc    the aim's joint change on top of the baked path, limits
#   ik_every                 (unused since the streamed run aims on its own thread)
#   aim_hz                   streamed run: aim answers per second of path, evenly spaced
#   aim_solver               1 = wrist first (`_wrist_first`: points the lens, the position may drift
#                            up to aim_hold_mm, J4-J6 before J1-J3), 0 = the exact pose (precise IK)
#   horizon_s / horizon_dead_deg  wrist first: the level horizon (roll about the lens) is
#                            corrected with this time constant, errors under the deadband left
#   stream_process           streamed run: 1 = send from a process of its own (stream_proc.py,
#                            its own controller connection), 0 = from a thread in the app
#   track_smooth / track_delay  trackJoint's smooth and delay scales
# Smoothness before accuracy (the general direction is enough for live tracking):
# in a test these made the joints' acceleration 4-20 x smaller for a pointing
# error of ~4-6 deg instead of ~1 deg.
TUNE_DEFAULTS = {
    "smooth_s": 0.5, "deadband_mm": 5.0, "predict_s": 0.1,
    "aim_deadband_deg": 2.0, "aim_response_s": 0.8,
    "turn_dps": 20.0, "turn_acc": 40.0,
    "joint_smooth_s": 0.15, "joint_dps": 45.0, "joint_acc": 180.0,
    "ik_every": 1, "aim_hz": 10.0, "track_smooth": 0.5, "track_delay": 1.0,
    "stream_process": 1.0, "aim_solver": 1.0, "aim_hold_mm": 20.0, "horizon_s": 1.5, "horizon_dead_deg": 2.0,
}
TUNE_RANGES = {
    "smooth_s": (0.0, 5.0), "deadband_mm": (0.0, 50.0), "predict_s": (0.0, 0.5),
    "aim_deadband_deg": (0.0, 10.0), "aim_response_s": (0.0, 5.0),
    "turn_dps": (1.0, 180.0), "turn_acc": (1.0, 2000.0),
    "joint_smooth_s": (0.0, 1.0), "joint_dps": (1.0, 180.0), "joint_acc": (1.0, 2000.0),
    "ik_every": (1, 10), "aim_hz": (2.0, 50.0), "track_smooth": (0.0, 1.0), "track_delay": (0.0, 2.0),
    "stream_process": (0.0, 1.0), "aim_solver": (0.0, 1.0), "aim_hold_mm": (0.5, 50.0), "horizon_s": (0.0, 10.0), "horizon_dead_deg": (0.0, 10.0),
}
TUNE = dict(TUNE_DEFAULTS)


def tune(key):
    return TUNE.get(key, TUNE_DEFAULTS[key])


def set_tuning(data):
    """Take the known keys of data (clamped to TUNE_RANGES); returns TUNE."""
    for k, v in dict(data or {}).items():
        if k not in TUNE_DEFAULTS:
            continue
        try:
            lo, hi = TUNE_RANGES[k]
            v = min(hi, max(lo, float(v)))
            TUNE[k] = int(round(v)) if k == "ik_every" else v
        except (TypeError, ValueError):
            pass
    return dict(TUNE)


def _limits():
    return math.radians(tune("turn_dps")), math.radians(tune("turn_acc"))


def _follow(x, v, goal, vmax, amax, dt, tau):
    """A critically damped follower (arrays, same units) with speed and
    acceleration caps, in substeps of at most 5 ms: steps of the goal (IK every
    other point, a held answer) come out as smooth curves, never as a stop-start
    chatter."""
    if tau <= 1e-4:
        # No smoothing: straight to the goal, still within the speed cap.
        if dt <= 0:
            return x, v
        step = np.clip(goal - x, -vmax * dt, vmax * dt)
        return x + step, step / dt
    n = max(1, int(math.ceil(dt / 0.005)))
    h = dt / n
    for _ in range(n):
        acc = np.clip((goal - x) / (tau * tau) - 2.0 * v / tau, -amax, amax)
        v = np.clip(v + acc * h, -vmax, vmax)
        x = x + v * h
    return x, v


def _rr():
    import robot_runtime as rr
    return rr


def _bake():
    import path_bake as bake
    return bake


def enabled():
    try:
        return bool(_rr()._track.get("live"))
    except Exception:
        return False


def _xyzw(wxyz):
    w = np.asarray(wxyz, dtype=float).reshape(4)
    return [w[1], w[2], w[3], w[0]]


def _wxyz(rot):
    q = rot.as_quat()
    return np.array([q[3], q[0], q[1], q[2]], dtype=float)


def is_primary(name):
    """Live follows only the camera script's primary target: every other Look At
    target is still or animated, and the bake already aims at it."""
    return str(name or "").strip().lower() == str(getattr(_rr(), "PRIMARY_TARGET_NAME", "Primary target")).lower()


def span_specs():
    """[(t0, t1, spec)] of the arm's spans with Look At at the primary target (path time)."""
    import waypoints as wp
    bake = _bake()
    out = []
    wps = wp.waypoints
    for i in range(max(0, len(wps) - 1)):
        spec = bake._track_spec_for_seg(i)
        t0, t1 = wps[i].get("t"), wps[i + 1].get("t")
        if spec and t0 is not None and t1 is not None and is_primary(spec.get("name")):
            out.append((float(t0), float(t1), dict(spec)))
    return out


def _cursor_pose(model, q):
    """The Sim Arm Cursor's pose for joints q (as Aimer places it), under the caller's lock."""
    rr = _rr()
    rr._set_q(model, q)
    p_ee, w_ee = rr._ee_frame(model)
    r_ee = R.from_quat(_xyzw(w_ee))
    pos = np.asarray(p_ee, dtype=float).reshape(3) + r_ee.as_matrix() @ rr.CURSOR_POS_OFFSET["sim"]
    return pos, R.from_euler("xyz", rr.CURSOR_RPY_OFFSET["sim"]) * r_ee


WRIST_FIRST_STEPS = 3
WRIST_FIRST_EPS = 1e-4                       # rad, for the numeric Jacobian
WRIST_FIRST_JOINT_COST = np.array([4.0, 4.0, 3.0, 1.0, 1.0, 1.0])   # J1..J6: the wrist first
WRIST_FIRST_DAMP = 2e-3


def _wrist_first(model, lock, q0, p_d, r_d, look_local, roll_gain=1.0, roll_dead=0.0):
    """Joints that point the lens like r_d, found by a few smooth damped
    least-squares steps from q0 (the last answer). The position may drift up
    to about aim_hold_mm from p_d and the wrist is preferred over J1-J3, so a
    small aim change is mostly a small wrist move. The roll about the lens axis
    (the level horizon Look At keeps) is corrected gently: errors under
    roll_dead are left, and of the rest only the share roll_gain per answer
    (holding it exactly turned J6 ~17 deg for a 3 deg aim change in a test
    pose, so target jitter became large J6 swings)."""
    q = np.asarray(q0, dtype=float).reshape(6).copy()
    hold = max(0.5, float(tune("aim_hold_mm"))) / 1000.0
    w_pos = math.radians(1.0) / hold           # hold_mm of drift costs as much as 1 deg of aim
    lam = WRIST_FIRST_DAMP * WRIST_FIRST_JOINT_COST ** 2
    with lock:
        p0, r0 = _cursor_pose(model, q)
        e_r0 = (r_d * r0.inv()).as_rotvec()
        if look_local is not None:
            # The turn to make: all of the pointing, part of the roll.
            a0 = r0.apply(look_local)
            e_roll = float(np.dot(e_r0, a0))
            use = math.copysign(max(0.0, abs(e_roll) - roll_dead), e_roll) * roll_gain
            r_t = R.from_rotvec(e_r0 - (e_roll - use) * a0) * r0
        else:
            r_t = r_d                          # no target: the whole turn back to the bake
        for _ in range(WRIST_FIRST_STEPS):
            p, r = _cursor_pose(model, q)
            e_p = np.asarray(p_d, dtype=float) - p
            e_r = (r_t * r.inv()).as_rotvec()
            jac = np.zeros((6, 6))
            for j in range(6):
                qj = q.copy()
                qj[j] += WRIST_FIRST_EPS
                pj, rj = _cursor_pose(model, qj)
                jac[:3, j] = (pj - p) / WRIST_FIRST_EPS
                jac[3:, j] = (rj * r.inv()).as_rotvec() / WRIST_FIRST_EPS
            jw = np.vstack([w_pos * jac[:3], jac[3:]])
            ew = np.concatenate([w_pos * e_p, e_r])
            dq = np.linalg.solve(jw.T @ jw + np.diag(lam), jw.T @ ew)
            q = q + np.clip(dq, -0.1, 0.1)
            if float(np.max(np.abs(dq))) < 1e-6:
                break
    return q


class Aimer:
    def __init__(self, specs=None, model=None, lock=None):
        self.specs = span_specs() if specs is None else specs
        self.model = model   # None: the viewer's rr.robot_ik / rr._ik_lock
        self.lock = lock
        self.reset()
        self.err = ""
        self.dev_deg = 0.0
        self.lag_deg = 0.0

    def reset(self):
        self.corr = R.identity()      # the aim's turn from the bake (world rotation)
        self.corr_ok = R.identity()   # ... at the last IK answer taken
        self.w = np.zeros(3)          # its angular velocity, rad/s (world rotvec)
        self.goal_f = R.identity()    # the goal through the aim deadzone
        self.delta_t = np.zeros(6)    # joints - baked joints, as the IK answered
        self.delta = np.zeros(6)      # ... as sent (filtered)
        self.dv = np.zeros(6)         # its rate, rad/s
        self.ik_d = None              # last IK delta and its path time (for rate_t)
        self.ik_t = None
        self.rate_t = np.zeros(6)     # rate of the IK answers, rad/s
        self.q_prev = None
        self.t_prev = None

    def _spec_at(self, t):
        for t0, t1, spec in self.specs:
            if t0 - 1e-6 <= t <= t1 + 1e-6:
                return spec
        return None

    def _out(self, q_base, dt, why=""):
        """Joints sent: the baked joints plus the filtered IK delta."""
        if why:
            # Held: the aim goes back to the last turn the joints took (else it
            # ran on ahead and every later answer was a jump too), and stops.
            self.err = why
            self.rate_t = np.zeros(6)
            self.corr = self.corr_ok
            self.w = np.zeros(3)
        self.delta, self.dv = _follow(self.delta, self.dv, self.delta_t,
                                      math.radians(tune("joint_dps")), math.radians(tune("joint_acc")),
                                      dt, tune("joint_smooth_s"))
        q = np.asarray(q_base, dtype=float) + self.delta
        self.q_prev = q
        return q

    def _turn(self, goal, dt):
        """Move corr toward goal: the goal through a deadzone (goal_f, held
        while the aim moves less than aim_deadband_deg, then dragged along at
        that distance), the turn with the response time and speed /
        acceleration limits. No rest state, so a moving target is followed
        smoothly, not caught up in steps."""
        vmax, amax = _limits()
        db = math.radians(tune("aim_deadband_deg"))
        ge = (goal * self.goal_f.inv()).as_rotvec()
        gd = float(np.linalg.norm(ge))
        if gd > db:
            self.goal_f = R.from_rotvec(ge * ((gd - db) / gd)) * self.goal_f
        if dt <= 0:
            return
        e = (self.goal_f * self.corr.inv()).as_rotvec()
        dist = float(np.linalg.norm(e))
        if dist < 1e-7 and float(np.linalg.norm(self.w)) < 1e-6:
            self.w = np.zeros(3)
            return
        u = e / max(dist, 1e-12)
        speed = min(vmax, math.sqrt(2.0 * amax * dist), dist / dt)
        resp = tune("aim_response_s")
        if resp > 1e-4:
            speed = min(speed, dist / resp)
        dw = u * speed - self.w
        n = float(np.linalg.norm(dw))
        if n > amax * dt:
            dw *= amax * dt / n
        self.w = self.w + dw
        self.corr = R.from_rotvec(self.w * dt) * self.corr

    def __call__(self, t, q_base, solve=True):
        rr = _rr()
        bake = _bake()
        q_base = np.asarray(q_base, dtype=float).reshape(-1)[:6]
        t = float(t)
        dt = 0.04 if self.t_prev is None else min(0.2, max(0.0, t - self.t_prev))
        if self.t_prev is not None and t < self.t_prev - 1e-6:
            self.reset()  # the clock went back (a loop, a seek)
            dt = 0.04
        self.t_prev = t
        if not solve:
            # Between IK answers: carry the answer on at its last rate.
            self.delta_t = self.delta_t + self.rate_t * dt
            return self._out(q_base, dt)
        spec = self._spec_at(t)
        model = self.model if self.model is not None else getattr(rr, "robot_ik", None)
        lock = self.lock if self.lock is not None else rr._ik_lock
        if model is None:
            return self._out(q_base, dt, "no IK model")
        # The cursor where the baked joints put it.
        with lock:
            rr._set_q(model, q_base)
            p_ee, w_ee = rr._ee_frame(model)
        r_ee = R.from_quat(_xyzw(w_ee))
        pos = np.asarray(p_ee, dtype=float).reshape(3) + r_ee.as_matrix() @ rr.CURSOR_POS_OFFSET["sim"]
        r_base = R.from_euler("xyz", rr.CURSOR_RPY_OFFSET["sim"]) * r_ee
        # Where the aim should be turned now.
        goal = R.identity()
        if spec is not None:
            # Smoothed like the cursor's own Look At (deadband + low-pass on the target).
            tgt = rr.smooth_target("live:%s" % id(self), rr._track_target_pos(spec.get("name")))
            if tgt is None:
                return self._out(q_base, dt, "target %r not found" % spec.get("name"))
            aim = R.from_quat(_xyzw(rr._aim_wxyz(pos, tgt, spec.get("offset"), spec.get("pos_offset"))))
            goal = aim * r_base.inv()
            # The cursor-local axis Look At points at the target (for the wrist-first solver).
            lp = np.asarray(list(spec.get("pos_offset") or [0.0, 0.0, 0.0])[:3], dtype=float)
            to_t = np.asarray(tgt, dtype=float).reshape(3) - (pos + aim.apply(lp))
            nt = float(np.linalg.norm(to_t))
            look_local = aim.inv().apply(to_t / nt) if nt > 1e-6 else None
        else:
            look_local = None
            ang = np.degrees(goal.magnitude())
            if ang > MAX_DEV_DEG:
                goal = R.from_rotvec(goal.as_rotvec() * (MAX_DEV_DEG / ang))
        self._turn(goal, dt)
        corr = self.corr
        # How far the aim trails the target (deadband, speed / deviation limits).
        self.lag_deg = float(np.degrees((goal * corr.inv()).magnitude()))
        self.dev_deg = float(np.degrees(corr.magnitude()))
        if self.dev_deg < 1e-3 and float(np.linalg.norm(self.w)) < 1e-6:
            # No turn from the bake: the baked joints (the filter brings them back).
            self.delta_t = np.zeros(6)
            self.rate_t = np.zeros(6)
            self.ik_d = None
            self.corr_ok = self.corr
            self.err = ""
            return self._out(q_base, dt)
        # Seed and jump check from the last answer's delta on today's base joints.
        ref = q_base + self.delta_t
        if float(tune("aim_solver")) >= 0.5:
            hs = float(tune("horizon_s"))
            gain = 1.0 if hs <= 1e-3 else 1.0 - math.exp(-max(dt, 1e-3) / hs)
            q = _wrist_first(model, lock, ref, pos, corr * r_base, look_local,
                             roll_gain=gain, roll_dead=math.radians(float(tune("horizon_dead_deg"))))
        else:
            ee_pos, ee_wxyz = rr._ee_from_cursor(pos, _wxyz(corr * r_base), "sim")
            q = rr.solve_backup_ik(ee_pos, ee_wxyz, seed_q=ref, quiet=True, precise=True,
                                   model=self.model, lock=self.lock)
        if q is None:
            return self._out(q_base, dt, "IK failed")
        q = bake._unwrap_toward(np.asarray(q, dtype=float), q_base)
        if bake._over_limit(q) is not None:
            return self._out(q_base, dt, "joint limit")
        if float(np.degrees(np.max(np.abs(q - ref)))) > JUMP_MAX_DEG:
            return self._out(q_base, dt, "joint jump")
        d = q - q_base
        if self.ik_d is not None and t > self.ik_t + 1e-6:
            cap = math.radians(tune("joint_dps"))
            self.rate_t = np.clip((d - self.ik_d) / (t - self.ik_t), -cap, cap)
        self.ik_d, self.ik_t = d, t
        if float(np.degrees(np.max(np.abs(d - self.delta_t)))) >= IK_DEADBAND_DEG:
            self.delta_t = d  # smaller changes are IK noise
        self.corr_ok = corr
        self.err = ""
        return self._out(q_base, dt)


# ---------------------------------------------------------------------------
# Simulate
# ---------------------------------------------------------------------------
_sim = {"aimer": None, "gen": None}


def sim_q(t, q_base, gen):
    """Simulate: the aimed joints at path time t (gen: the play it belongs to)."""
    if not enabled():
        _sim["aimer"] = None
        return q_base
    if _sim["aimer"] is None or _sim["gen"] != gen:
        _sim["aimer"] = Aimer()
        _sim["gen"] = gen
    try:
        return _sim["aimer"](t, q_base)
    except Exception as e:
        print("live aim failed:", e)
        return q_base


def sim_status():
    a = _sim.get("aimer")
    return {} if a is None else {"err": a.err, "dev_deg": round(a.dev_deg, 1), "lag_deg": round(a.lag_deg, 1)}


# ---------------------------------------------------------------------------
# Streamed real run
# ---------------------------------------------------------------------------
def _interp(samples, t):
    if t <= samples[0]["t"]:
        return np.asarray(samples[0]["q"], dtype=float).reshape(-1)[:6]
    if t >= samples[-1]["t"]:
        return np.asarray(samples[-1]["q"], dtype=float).reshape(-1)[:6]
    for a, b in zip(samples, samples[1:]):
        if a["t"] <= t <= b["t"]:
            qa = np.asarray(a["q"], dtype=float).reshape(-1)[:6]
            qb = np.asarray(b["q"], dtype=float).reshape(-1)[:6]
            u = (t - a["t"]) / max(1e-9, b["t"] - a["t"])
            return (1 - u) * qa + u * qb
    return np.asarray(samples[-1]["q"], dtype=float).reshape(-1)[:6]


# Streamed run safety. The first version dropped a point the controller
# refused (queue full) while its clock ran on, and let any joint step up to
# 5 deg per point (250 deg/s) to catch up: after a delay the arm sped up hard.
STEP_MARGIN_DPS = 20.0   # a point may move this much faster than the path + aim limits ...
STOP_BEHIND_DEG = 3.0    # ... and the stream stops if it falls this far behind what it should send
FOLLOW_STOP_DEG = 10.0   # the arm this far from the last point sent -> stop
READ_EVERY = 5           # read the arm's joints every N points
FULL_STOP_S = 1.0        # the controller refusing points (queue full) this long -> stop
SETTLE_MAX_S = 0.5       # at the end, keep sending the last point while the aim settles
LEAD_POINTS = 3          # points kept queued ahead on the controller (60 ms): a late tick doesn't starve it
QUEUE_MAX = 8            # with getTrajectoryQueueSize: don't send while this many are queued
RESYNC_S = 0.25          # a stall longer than this restarts the schedule (else late points go out back to back)
ANSWERS_KEPT = 12        # the aimer's recent answers the sender blends between


def _fine_timer(on):
    """Windows sleeps in 15.6 ms steps unless the timer is set to 1 ms: the
    stream's 20 ms ticks became ~31 ms, so the arm ran each point and waited
    (motion in steps) and the path ran at ~0.6 x, ending long after the timeline."""
    import os
    if os.name != "nt":
        return
    try:
        import ctypes
        (ctypes.windll.winmm.timeBeginPeriod if on else ctypes.windll.winmm.timeEndPeriod)(1)
    except Exception:
        pass


def _sleep_until(t):
    """Sleep to t: short sleeps, then a spin for the last 2 ms (exact on Windows too)."""
    while True:
        left = t - time.perf_counter()
        if left <= 0:
            return
        if left > 0.002:
            time.sleep(min(left - 0.0015, 0.005))
LOG_MAX = 6000           # points kept for the speed export


_own = {"path": None, "model": None, "lock": threading.Lock()}


def own_model():
    """A private IK model for the streamed run (loaded once per URDF, ~1 s):
    on the viewer's robot_ik the aim waited for the viewer's own IK / FK under
    _ik_lock, and points went out 80-360 ms apart."""
    rr = _rr()
    path = str(getattr(rr, "URDF_SIM_PATH", "") or "")
    if _own["model"] is not None and _own["path"] == path:
        try:
            with _own["lock"]:
                rr._apply_base(_own["model"])  # the arm's base as it is now
        except Exception:
            pass
        return _own["model"], _own["lock"]
    from skrobot.model import RobotModel
    m = RobotModel()
    m.load_urdf_file(path)
    try:
        rr.apply_aubo_i5_joint_limits(m)
    except Exception:
        pass
    try:
        rr._apply_base(m)
    except Exception:
        pass
    _own.update(path=path, model=m)
    return m, _own["lock"]


class Streamer:
    """The streamed Live run, as two threads:

    - the sender (`_run`): every STREAM_DT, the baked joints at the next path
      time plus the aim's joint change (`delta`, filtered here at the send
      rate), checked (`_limit`) and sent with trackJoint. It does nothing slow,
      so the points go out evenly (with the aim solved in the same loop, IK and
      the viewer's IK lock made the points 80-360 ms apart: jitter, and the
      path ran at ~0.7 x speed).
    - the aimer (`_aim_loop`): as often as it can, the Aimer at a path time
      on a fixed rhythm (path times k / aim_hz, each solved once the sender gets within `_aim_lead()`), on its own IK model;
      the sender blends between the two answers either side of each point
      (`_delta_goal`), and its filter smooths what is left.

    The path time moves on only when a point is taken (a refused point is
    sent again); the run stops (stopJoint) when the aim asks for more than the
    limits allow or the arm falls behind. stop() ends it."""

    def __init__(self, bot, samples, external=False):
        """external: the aim answers come from outside (`feed_answer`; the
        stream process, stream_proc.py), so no aimer of its own runs here."""
        self.bot = bot
        self.samples = list(samples)
        self.t_end = float(self.samples[-1]["t"]) if self.samples else 0.0
        self.external = bool(external)
        self.on_sent = None           # hook: on_sent(s, q) for every point taken
        if self.external:
            self.aimer = Aimer(specs=[])  # status only
        else:
            try:
                model, lock = own_model()
            except Exception as e:
                print("live stream: no private IK model, sharing the viewer's:", e)
                model, lock = None, None
            self.aimer = Aimer(model=model, lock=lock)
        self.stop_ev = threading.Event()
        self.thread = None
        self.aim_thread = None
        self.stats = {"sent": 0, "full": 0, "cleared": 0, "late": 0, "aim_err": "",
                      "max_follow_deg": 0.0, "max_behind_deg": 0.0, "max_dps": 0.0,
                      "aims": 0, "max_aim_ms": 0.0}
        self.error = ""
        self.log = []
        self.s_sent = 0.0
        self.aim_d = np.zeros(6)      # the aimer's latest joint change (rad)
        self.answers = []             # [(path time, joint change)] of the aimer's recent answers
        self.aim_gap = 0.1            # its time between answers (s), averaged
        self.aim_at = None            # perf_counter of the latest answer
        self.delta = np.zeros(6)      # joint change as sent (filtered)
        self.dv = np.zeros(6)
        # The arm stands on the first point (checked before the run).
        self.q_last = _interp(self.samples, 0.0) if self.samples else None

    def start(self):
        mc = self.bot.mc
        for fn in (lambda: self.bot.PrepareJog() if hasattr(self.bot, "PrepareJog") else None,
                   lambda: mc.clearPath()):
            try:
                fn()
            except Exception:
                pass
        if not self.external:
            self.aim_thread = threading.Thread(target=self._aim_loop, name="live_aim", daemon=True)
            self.aim_thread.start()
        self.thread = threading.Thread(target=self._run, name="live_stream", daemon=True)
        self.thread.start()

    def feed_answer(self, s, d, dev=None, err=None):
        """An aim answer from outside (external mode); dev / err: the app's aim
        status, for the log (the stand-in aimer here logged 0 deg)."""
        if dev is not None:
            self.aimer.dev_deg = float(dev)
            self.aimer.err = str(err or "")
        d = np.asarray(d, dtype=float).reshape(6)
        self.answers = (self.answers + [(float(s), d)])[-ANSWERS_KEPT:]
        self.aim_d = d
        self.aim_at = time.perf_counter()

    def _on_answer(self, s, d):
        """Hook: each new aim answer (ProcStreamer passes it to the stream process)."""

    def stop(self):
        self.stop_ev.set()
        for th in (self.thread, self.aim_thread):
            if th is not None and th is not threading.current_thread():
                th.join(1.0)

    # -- aimer thread -------------------------------------------------------
    def _aim_lead(self):
        """How far ahead of the sender the answers must reach: the queued points,
        two rhythm steps, and the path a catching-up sender covers in one send."""
        return LEAD_POINTS * STREAM_DT + 2.0 / max(1.0, float(tune("aim_hz"))) + 4 * STREAM_DT

    def _aim_loop(self):
        """Answers on a fixed rhythm: at path times k / aim_hz (evenly spaced),
        each solved once the sender gets within `_aim_lead()` of it, nothing in
        between. (Solving as often as it could, at whatever path time the sender
        had reached, gave unevenly spaced answers, and the aimer computing most
        of the time kept the sender waiting for Python's interpreter.)"""
        last = None
        while not self.stop_ev.is_set():
            step = 1.0 / max(1.0, float(tune("aim_hz")))
            s_sent = self.s_sent
            if last is not None and last >= self.t_end - 1e-9:
                time.sleep(0.01)  # the end's answer is in
                continue
            # The next grid time after the last answer (missed ones are skipped).
            k = math.floor((last if last is not None else s_sent) / step + 1e-9) + 1
            s = max(k * step, math.ceil(s_sent / step - 1e-9) * step)
            s = min(self.t_end, s)
            if last is not None and s - s_sent > self._aim_lead():
                time.sleep(0.005)  # far enough ahead: wait for the sender
                continue
            t0 = time.perf_counter()
            try:
                q_base = _interp(self.samples, s)
                self.aimer(s, q_base)
                d = np.asarray(self.aimer.delta_t, dtype=float).copy()
            except Exception as e:
                self.aimer.err = "aim failed: %s" % e
                time.sleep(0.02)
                continue
            now = time.perf_counter()
            if self.aim_at is not None:
                self.aim_gap = 0.8 * self.aim_gap + 0.2 * min(1.0, now - self.aim_at)
            # A new list each time (the sender reads it without a lock).
            self.answers = (self.answers + [(s, d)])[-ANSWERS_KEPT:]
            self._on_answer(s, d)
            self.aim_d, self.aim_at, last = d, now, s
            self.stats["aims"] += 1
            self.stats["max_aim_ms"] = max(self.stats["max_aim_ms"], (now - t0) * 1000.0)
            self.stats["aim_err"] = self.aimer.err
            self.stats["aim_gap_ms"] = round(self.aim_gap * 1000.0, 1)
            self.stats["aim_ahead_ms"] = round((s - self.s_sent) * 1000.0, 1)

    def _delta_goal(self, s):
        """The aim's joint change at path time s: a smooth curve through the
        answers either side of it. (Carrying the latest answer on
        at its rate and then holding it made a sawtooth: the commanded speed
        jumped ~5 deg/s at each answer and ~2.5 deg/s between.) Before the first
        answer: the change as sent; past the latest: the latest."""
        ans = self.answers
        if not ans:
            return self.delta
        if s <= ans[0][0]:
            # Before the oldest answer kept (the run's start): blend toward it from
            # the change as sent at the last point.
            s_prev = min(self.s_sent, s)
            u = 1.0 if ans[0][0] - s_prev < 1e-9 else (s - s_prev) / (ans[0][0] - s_prev)
            return (1.0 - u) * self.delta + u * ans[0][1]
        for i in range(len(ans) - 1):
            (s0, d0), (s1, d1) = ans[i], ans[i + 1]
            if s0 <= s <= s1:
                # A smooth curve through the answers (Catmull-Rom): straight blends
                # between them changed the speed suddenly at every answer.
                h = max(1e-9, s1 - s0)
                u = (s - s0) / h
                sa, da = ans[i - 1] if i > 0 else (s0, d0)
                sb, db = ans[i + 2] if i + 2 < len(ans) else (s1, d1)
                m0 = (d1 - da) / max(1e-9, s1 - sa)
                m1 = (db - d0) / max(1e-9, sb - s0)
                u2, u3 = u * u, u * u * u
                return ((2 * u3 - 3 * u2 + 1) * d0 + (u3 - 2 * u2 + u) * h * m0
                        + (-2 * u3 + 3 * u2) * d1 + (u3 - u2) * h * m1)
        self.stats["past_aim"] = self.stats.get("past_aim", 0) + 1
        return ans[-1][1]

    def _step_delta(self, dur, goal=None, s=None):
        self.delta, self.dv = _follow(self.delta, self.dv, self._delta_goal(s) if goal is None else goal,
                                      math.radians(tune("joint_dps")), math.radians(tune("joint_acc")),
                                      dur, tune("joint_smooth_s"))
        return self.delta

    # -- sender thread ------------------------------------------------------
    def _limit(self, q, q_base, qb_prev, dur=STREAM_DT):
        """q held to the path's own step plus the aim's speed (and a margin) over dur."""
        lim = np.abs(q_base - qb_prev) + math.radians(tune("joint_dps") + STEP_MARGIN_DPS) * dur
        step = np.clip(q - self.q_last, -lim, lim)
        out = self.q_last + step
        behind = float(np.degrees(np.max(np.abs(q - out))))
        self.stats["max_behind_deg"] = max(self.stats["max_behind_deg"], behind)
        if behind > STOP_BEHIND_DEG:
            raise RuntimeError("the aimed path asks for a %.1f deg jump, more than the arm may make: stopped" % behind)
        return out

    def _send(self, mc, q, dur=STREAM_DT):
        return int(mc.trackJoint([float(v) for v in q], float(dur), float(tune("track_smooth")), float(tune("track_delay"))))

    def _queue(self, mc):
        if self._qfn is False:
            return None
        try:
            return int(mc.getTrajectoryQueueSize())
        except Exception:
            self._qfn = False  # not on this controller / SDK
            return None

    def _run(self):
        mc = self.bot.mc
        _fine_timer(True)
        self._qfn = None
        t_wall0 = time.perf_counter()
        due = t_wall0 - LEAD_POINTS * STREAM_DT  # the first LEAD_POINTS go out at once
        s = 0.0                      # path time of the last point taken
        qb_prev = _interp(self.samples, 0.0)
        full_since = None
        n = 0
        settle = None                # wall time the end's settling began
        last_sent = None
        try:
            while not self.stop_ev.is_set():
                qn = self._queue(mc)
                if qn is not None and qn >= QUEUE_MAX:
                    _sleep_until(time.perf_counter() + 0.004)  # plenty queued: wait
                    continue
                n += 1
                due += STREAM_DT
                if settle is None:
                    # One tick of path per point; behind the wall clock (a stall) a point
                    # covers the missed time, up to 4 ticks, with a matching duration.
                    behind = time.perf_counter() - t_wall0 - LEAD_POINTS * STREAM_DT - s
                    step = min(4 * STREAM_DT, max(STREAM_DT, behind)) if behind > 2 * STREAM_DT else STREAM_DT
                    s_new = min(s + step, self.t_end)
                    dur = max(STREAM_DT, s_new - s)
                    q_base = _interp(self.samples, s_new)
                    q = q_base + self._step_delta(dur, s=s_new)
                else:
                    # Past the end: the last point, the aim's joints coming to rest on its last answer.
                    s_new, dur, q_base = self.t_end, STREAM_DT, qb_prev
                    q = q_base + self._step_delta(dur, goal=self.aim_d)
                q = self._limit(q, q_base, qb_prev, dur)
                rc = self._send(mc, q, dur)
                now = time.perf_counter()
                gap = None
                if rc == 0:
                    dps = float(np.degrees(np.max(np.abs(q - self.q_last)))) / dur
                    self.stats["max_dps"] = max(self.stats["max_dps"], dps)
                    self.q_last = q
                    s, qb_prev = s_new, q_base
                    self.s_sent = s
                    if self.on_sent is not None:
                        self.on_sent(s, q)
                    self.stats["sent"] += 1
                    full_since = None
                    if last_sent is not None:
                        gap = now - last_sent
                        self.stats["max_gap_ms"] = max(self.stats.get("max_gap_ms", 0.0), gap * 1000.0)
                    last_sent = now
                elif rc == 2:
                    # Queue full: the same point again next time (the path time waits).
                    self.stats["full"] += 1
                    full_since = full_since or now
                    if now - full_since > FULL_STOP_S:
                        raise RuntimeError("the controller kept refusing points (queue full) for %.1f s" % FULL_STOP_S)
                elif rc in (-3, -4):
                    # A stop left the path held: clear it and go on from where the arm is.
                    mc.clearPath()
                    self.stats["cleared"] += 1
                    cur = self.bot.GetJoints()
                    if cur is not None:
                        self.q_last = np.asarray(cur, dtype=float).reshape(-1)[:6]
                elif rc == -22:
                    raise RuntimeError("the controller predicts a self-collision at t %.2f" % s_new)
                else:
                    raise RuntimeError("trackJoint rc %s at t %.2f" % (rc, s_new))
                follow = None
                if n % READ_EVERY == 0:
                    cur = self.bot.GetJoints()
                    if cur is not None:
                        follow = float(np.degrees(np.max(np.abs(np.asarray(cur, dtype=float)[:6] - self.q_last))))
                        self.stats["max_follow_deg"] = max(self.stats["max_follow_deg"], follow)
                        if follow > FOLLOW_STOP_DEG:
                            raise RuntimeError("the arm is %.1f deg behind the stream: stopped" % follow)
                if len(self.log) < LOG_MAX:
                    self.log.append({"t": round(now - t_wall0, 4), "s": round(s, 4), "rc": rc,
                                     "gap_ms": None if gap is None else round(gap * 1000.0, 1), "queue": qn,
                                     "dur_ms": round(dur * 1000.0, 1),
                                     "q": [round(math.degrees(v), 3) for v in self.q_last],
                                     "follow_deg": None if follow is None else round(follow, 2),
                                     "dev_deg": round(self.aimer.dev_deg, 2), "aim_err": self.aimer.err})
                if settle is None and s >= self.t_end - 1e-9:
                    settle = now
                if settle is not None and (now - settle > SETTLE_MAX_S
                                           or float(np.max(np.abs(self.dv))) < math.radians(0.5)):
                    break  # the last point is sent and the aim has come to rest
                if now - due > RESYNC_S:
                    self.stats["late"] += 1
                    due = now  # a long stall: start the schedule again (no burst)
                elif qn is not None and qn <= 1:
                    due = min(due, now)  # the controller is about to run dry: send the next one now
                else:
                    _sleep_until(due)
        except Exception as e:
            self.error = str(e)
            print("live stream stopped:", e)
            try:
                mc.stopJoint(3.0)
            except Exception:
                pass
        finally:
            _fine_timer(False)
            self.stop_ev.set()  # the aimer too
        self.stats["wall_s"] = round(time.perf_counter() - t_wall0, 3)
        self.stats["path_s"] = round(s, 3)
        print("live stream done", self.stats)


class ProcStreamer(Streamer):
    """The streamed run with the sender in a process of its own (stream_proc.py,
    its own RPC connection): here only the aimer runs (on the app's private IK
    model) and its answers go to the process, which sends the points. A thread
    here (`thread`) reads what the process reports: the points it sent (the sim
    arm shows them, the aimer aims ahead of them) and, at the end, its stats and
    log. A heartbeat every HEARTBEAT_S lets the process stop the arm itself if
    the app goes quiet."""

    HEARTBEAT_S = 0.1

    def __init__(self, bot, samples, child):
        super().__init__(bot, samples)
        self.child = child            # a ready _Child (connected to the controller)
        self.hb_thread = None

    def start(self):
        samples = [{"t": float(x["t"]), "q": [float(v) for v in np.asarray(x["q"], dtype=float).reshape(-1)[:6]]}
                   for x in self.samples]
        self.child.send("start", {"samples": samples, "tune": dict(TUNE)})
        self.aim_thread = threading.Thread(target=self._aim_loop, name="live_aim", daemon=True)
        self.aim_thread.start()
        self.hb_thread = threading.Thread(target=self._heartbeat, name="live_hb", daemon=True)
        self.hb_thread.start()
        self.thread = threading.Thread(target=self._read, name="live_stream_rx", daemon=True)
        self.thread.start()

    def _on_answer(self, s, d):
        self.child.send("aim", float(s), [float(v) for v in d], float(self.aimer.dev_deg), str(self.aimer.err or ""))

    def _heartbeat(self):
        sent_tune = dict(TUNE)
        while not self.stop_ev.is_set():
            self.child.send("hb")
            if TUNE != sent_tune:
                sent_tune = dict(TUNE)
                self.child.send("tune", sent_tune)  # Real Time Motion edits during the run
            time.sleep(self.HEARTBEAT_S)

    def _read(self):
        try:
            while True:
                m = self.child.recv(timeout=2.0)
                if m is None:
                    if self.child.proc.poll() is not None:
                        raise RuntimeError("the stream process ended")
                    continue
                kind = m[0]
                if kind == "sent":
                    self.s_sent = float(m[1])
                    self.q_last = np.asarray(m[2], dtype=float)
                    self.stats["sent"] += 1
                elif kind == "log":
                    print("stream process:", m[1])
                elif kind == "done":
                    stats, err, log = m[1], m[2], m[3]
                    self.stats.update({k: v for k, v in stats.items() if k not in ("aims", "max_aim_ms", "aim_err")})
                    self.log = log
                    self.error = err or ""
                    break
        except Exception as e:
            self.error = str(e)
            print("live stream (process) stopped:", e)
        self.stop_ev.set()
        self.child.close()
        print("live stream (process) done", self.stats)

    def stop(self):
        self.stop_ev.set()
        self.child.send("stop")
        for th in (self.thread, self.aim_thread, self.hb_thread):
            if th is not None and th is not threading.current_thread():
                th.join(1.5)
        self.child.close()


class _Child:
    """The stream process: started and connected to the controller at Upload, so
    Run only has to hand it the path."""

    def __init__(self, ip, port, user, pwd, timeout_s=15.0):
        import os
        import secrets
        import subprocess
        from multiprocessing.connection import Listener
        self.lock = threading.Lock()
        key = secrets.token_bytes(16)
        self.listener = Listener(("127.0.0.1", 0), authkey=key)
        here = os.path.dirname(os.path.abspath(__file__))
        args = [sys.executable, os.path.join(here, "stream_proc.py"), str(self.listener.address[1]), key.hex(),
                str(ip), str(int(port)), str(user), str(pwd)]
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        self.proc = subprocess.Popen(args, cwd=here, creationflags=flags)
        self.conn = None
        box = {}

        def accept():
            try:
                box["c"] = self.listener.accept()
            except Exception as e:
                box["e"] = e
        th = threading.Thread(target=accept, daemon=True)
        th.start()
        th.join(timeout_s)
        if "c" not in box:
            self.close()
            raise RuntimeError("the stream process did not start (%s)" % box.get("e", "timeout"))
        self.conn = box["c"]
        m = self.recv(timeout=timeout_s)
        while m is not None and m[0] == "log":
            print("stream process:", m[1])
            m = self.recv(timeout=timeout_s)
        if m is None or m[0] != "ready" or not m[1]:
            why = (m[2] if m is not None and m[0] == "ready" else "no answer")
            self.close()
            raise RuntimeError(why)

    def send(self, *msg):
        if self.conn is None:
            return
        try:
            with self.lock:
                self.conn.send(msg)
        except Exception:
            pass

    def recv(self, timeout=1.0):
        try:
            if self.conn is not None and self.conn.poll(timeout):
                return self.conn.recv()
        except (EOFError, OSError):
            raise RuntimeError("the stream process closed the connection")
        return None

    def alive(self):
        return self.conn is not None and self.proc.poll() is None

    def close(self):
        for fn in (lambda: self.conn.close() if self.conn is not None else None,
                   lambda: self.listener.close()):
            try:
                fn()
            except Exception:
                pass
        self.conn = None
        try:
            self.proc.wait(1.0)
        except Exception:
            try:
                self.proc.kill()
            except Exception:
                pass


_stream = {"s": None, "child": None}


def prepare_process(bot):
    """At Upload: start the stream process and let it connect (a few seconds)
    so Run doesn't wait for it. False (and the in-app sender at Run) when it
    is switched off or can't start / connect."""
    drop_process()
    if float(tune("stream_process")) < 0.5 or bot is None:
        return False
    try:
        _stream["child"] = _Child(getattr(bot, "ip", ""), getattr(bot, "port", 30004),
                                  getattr(bot, "user", "aubo"), getattr(bot, "pwd", "123456"))
        print("live stream: stream process ready (own controller connection)")
        return True
    except Exception as e:
        print("live stream: stream process unavailable, the app will send:", e)
        return False


def drop_process():
    ch = _stream.get("child")
    _stream["child"] = None
    if ch is not None:
        ch.send("stop")
        ch.close()


def stream_start(bot, samples):
    stream_stop()
    ch = _stream.get("child")
    _stream["child"] = None
    st = None
    if ch is not None and ch.alive():
        try:
            st = ProcStreamer(bot, samples, ch)
            st.start()
        except Exception as e:
            print("live stream: stream process failed, the app sends:", e)
            ch.close()
            st = None
    if st is None:
        st = Streamer(bot, samples)
        st.start()
    _stream["s"] = st
    _stream["last"] = st
    return st


def stream_stop():
    st = _stream.get("s")
    _stream["s"] = None
    if st is not None:
        _stream["last"] = st
        st.stop()


def stream_q():
    """The last point sent by the streamed run (the sim arm shows it), or None."""
    st = _stream.get("s")
    return None if st is None or st.q_last is None else np.asarray(st.q_last, dtype=float)


def stream_running():
    st = _stream.get("s")
    return st is not None and st.thread is not None and st.thread.is_alive()


def stream_error():
    st = _stream.get("s")
    return "" if st is None else st.error


def stream_log():
    """The last streamed run's points (for the speed export), or None."""
    st = _stream.get("last") or _stream.get("s")
    if st is None:
        return None
    return {"stats": dict(st.stats), "error": st.error, "dt": STREAM_DT, "points": list(st.log),
            "sender": "process" if isinstance(st, ProcStreamer) else "app"}


def stream_status():
    st = _stream.get("s")
    if st is None:
        return {}
    return {"err": st.error or st.aimer.err, "dev_deg": round(st.aimer.dev_deg, 1),
            "lag_deg": round(st.aimer.lag_deg, 1), "sent": st.stats["sent"]}
