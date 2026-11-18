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
import threading
import time

import numpy as np
from scipy.spatial.transform import Rotation as R

MAX_DEV_DEG = 60.0      # largest turn away from the baked aim
JUMP_MAX_DEG = 8.0      # a joint jump bigger than this between IK answers is refused
STREAM_DT = 0.02        # streamed run: 50 points/s
AIM_EVERY = 2           # streamed run: IK every 2nd point; the filters run on every point
# Smoothing (the aim used to step at a constant rate toward a goal re-made on
# every tick, and each IK answer / hold went straight to the joints: target and
# IK jitter shook the arm). The aim now turns with a speed and an acceleration
# limit (`rr._track` turn_dps / turn_acc, set in the Look At panel), starts only
# when the goal is more than AIM_DEADBAND_DEG away and settles below
# AIM_SETTLE_DEG; the joints follow the IK answers through a critically damped per-joint filter
# (time constant JOINT_SMOOTH_S, capped at JOINT_DPS / JOINT_ACC); IK answers
# changing less than IK_DEADBAND_DEG are noise. Between IK answers the answer is
# carried on at its last rate (no steps for the filter to chatter on).
AIM_DEADBAND_DEG = 0.4
AIM_SETTLE_DEG = 0.05
JOINT_DPS = 120.0
JOINT_ACC = 720.0
JOINT_SMOOTH_S = 0.08
IK_DEADBAND_DEG = 0.02


def _limits():
    rr = _rr()
    try:
        vmax = float(rr._track.get("turn_dps", rr.TURN_DPS_DEFAULT))
        amax = float(rr._track.get("turn_acc", rr.TURN_ACC_DEFAULT))
    except Exception:
        vmax, amax = 60.0, 180.0
    return math.radians(max(1.0, vmax)), math.radians(max(1.0, amax))


def _follow(x, v, goal, vmax, amax, dt, tau):
    """A critically damped follower (arrays, same units) with speed and
    acceleration caps, in substeps of at most 5 ms: steps of the goal (IK every
    other point, a held answer) come out as smooth curves, never as a stop-start
    chatter."""
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


def span_specs():
    """[(t0, t1, spec)] of the arm's spans with Look At (path time)."""
    import waypoints as wp
    bake = _bake()
    out = []
    wps = wp.waypoints
    for i in range(max(0, len(wps) - 1)):
        spec = bake._track_spec_for_seg(i)
        t0, t1 = wps[i].get("t"), wps[i + 1].get("t")
        if spec and t0 is not None and t1 is not None:
            out.append((float(t0), float(t1), dict(spec)))
    return out


class Aimer:
    def __init__(self, specs=None):
        self.specs = span_specs() if specs is None else specs
        self.reset()
        self.err = ""
        self.dev_deg = 0.0
        self.lag_deg = 0.0

    def reset(self):
        self.corr = R.identity()      # the aim's turn from the bake (world rotation)
        self.corr_ok = R.identity()   # ... at the last IK answer taken
        self.w = np.zeros(3)          # its angular velocity, rad/s (world rotvec)
        self.moving = False           # the aim is turning (else it waits for the deadband)
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
                                      math.radians(JOINT_DPS), math.radians(JOINT_ACC), dt, JOINT_SMOOTH_S)
        q = np.asarray(q_base, dtype=float) + self.delta
        self.q_prev = q
        return q

    def _turn(self, goal, dt):
        """Move corr toward goal with the speed / acceleration limits and the deadband."""
        vmax, amax = _limits()
        e = (goal * self.corr.inv()).as_rotvec()
        dist = float(np.linalg.norm(e))
        if not self.moving:
            if math.degrees(dist) < AIM_DEADBAND_DEG:
                return
            self.moving = True
        if math.degrees(dist) < AIM_SETTLE_DEG and float(np.linalg.norm(self.w)) < math.radians(2.0):
            self.corr = goal
            self.w = np.zeros(3)
            self.moving = False
            return
        if dt <= 0:
            return
        # Speed along the error, braking in time; acceleration limited (vector).
        u = e / max(dist, 1e-12)
        speed = min(vmax, math.sqrt(2.0 * amax * dist), dist / dt)
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
        model = getattr(rr, "robot_ik", None)
        if model is None:
            return self._out(q_base, dt, "no IK model")
        # The cursor where the baked joints put it.
        with rr._ik_lock:
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
            ang = np.degrees(goal.magnitude())
            if ang > MAX_DEV_DEG:
                goal = R.from_rotvec(goal.as_rotvec() * (MAX_DEV_DEG / ang))
        self._turn(goal, dt)
        corr = self.corr
        # How far the aim trails the target (deadband, speed / deviation limits).
        self.lag_deg = float(np.degrees((goal * corr.inv()).magnitude()))
        self.dev_deg = float(np.degrees(corr.magnitude()))
        if self.dev_deg < 1e-3 and not self.moving:
            # No turn from the bake: the baked joints (the filter brings them back).
            self.delta_t = np.zeros(6)
            self.rate_t = np.zeros(6)
            self.ik_d = None
            self.corr_ok = self.corr
            self.err = ""
            return self._out(q_base, dt)
        ee_pos, ee_wxyz = rr._ee_from_cursor(pos, _wxyz(corr * r_base), "sim")
        # Seed and jump check from the last answer's delta on today's base joints.
        ref = q_base + self.delta_t
        q = rr.solve_backup_ik(ee_pos, ee_wxyz, seed_q=ref, quiet=True)
        if q is None:
            return self._out(q_base, dt, "IK failed")
        q = bake._unwrap_toward(np.asarray(q, dtype=float), q_base)
        if bake._over_limit(q) is not None:
            return self._out(q_base, dt, "joint limit")
        if float(np.degrees(np.max(np.abs(q - ref)))) > JUMP_MAX_DEG:
            return self._out(q_base, dt, "joint jump")
        d = q - q_base
        if self.ik_d is not None and t > self.ik_t + 1e-6:
            self.rate_t = np.clip((d - self.ik_d) / (t - self.ik_t), -math.radians(JOINT_DPS), math.radians(JOINT_DPS))
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


class Streamer:
    """Sends the baked path (aimed live) with trackJoint, timed from its start,
    until the path's end; stop() ends it (the run's stop then stops the arm)."""

    def __init__(self, bot, samples):
        self.bot = bot
        self.samples = list(samples)
        self.t_end = float(self.samples[-1]["t"]) if self.samples else 0.0
        self.aimer = Aimer()
        self.stop_ev = threading.Event()
        self.thread = None
        self.stats = {"sent": 0, "full": 0, "cleared": 0, "late": 0, "aim_err": ""}
        self.error = ""
        # The arm stands on the first point (checked before the run): the first
        # point sent can't jump away from it either.
        self.q_last = _interp(self.samples, 0.0) if self.samples else None

    def start(self):
        mc = self.bot.mc
        for fn in (lambda: self.bot.PrepareJog() if hasattr(self.bot, "PrepareJog") else None,
                   lambda: mc.clearPath()):
            try:
                fn()
            except Exception:
                pass
        self.thread = threading.Thread(target=self._run, name="live_stream", daemon=True)
        self.thread.start()

    def stop(self):
        self.stop_ev.set()
        if self.thread is not None and self.thread is not threading.current_thread():
            self.thread.join(1.0)

    def _run(self):
        mc = self.bot.mc
        t0 = time.perf_counter()
        k = 0
        clear = False
        try:
            while not self.stop_ev.is_set():
                k += 1
                due = t0 + k * STREAM_DT
                s = min(k * STREAM_DT, self.t_end)
                if clear:
                    mc.clearPath()
                    self.stats["cleared"] += 1
                    clear = False
                q_base = _interp(self.samples, s)
                # IK every AIM_EVERY points; the aim / joint filters step on every one.
                q = self.aimer(s, q_base, solve=(AIM_EVERY <= 1 or k % AIM_EVERY == 1))
                self.stats["aim_err"] = self.aimer.err
                if self.q_last is not None:
                    # Never more than 5 deg per joint in one point (the controller trips on steps).
                    lim = math.radians(5.0)
                    q = self.q_last + np.clip(q - self.q_last, -lim, lim)
                rc = int(mc.trackJoint([float(v) for v in q], STREAM_DT, 0.5, 1.0))
                if rc == 0:
                    self.q_last = q
                    self.stats["sent"] += 1
                elif rc == 2:
                    self.stats["full"] += 1
                elif rc in (-3, -4):
                    clear = True
                elif rc == -22:
                    raise RuntimeError("the controller predicts a self-collision at t %.2f" % s)
                else:
                    raise RuntimeError("trackJoint rc %s at t %.2f" % (rc, s))
                if s >= self.t_end:
                    break  # the last point is sent: the arm comes to rest on it
                left = due - time.perf_counter()
                if left > 0:
                    time.sleep(left)
                else:
                    self.stats["late"] += 1
        except Exception as e:
            self.error = str(e)
            print("live stream stopped:", e)
            try:
                mc.stopJoint(3.0)
            except Exception:
                pass
        print("live stream done", self.stats)


_stream = {"s": None}


def stream_start(bot, samples):
    stream_stop()
    st = Streamer(bot, samples)
    _stream["s"] = st
    st.start()
    return st


def stream_stop():
    st = _stream.get("s")
    _stream["s"] = None
    if st is not None:
        st.stop()


def stream_q():
    """The last point sent by the streamed run (the sim arm shows it), or None."""
    st = _stream.get("s")
    return None if st is None or st.q_last is None else np.asarray(st.q_last, dtype=float)


def stream_status():
    st = _stream.get("s")
    if st is None:
        return {}
    return {"err": st.error or st.aimer.err, "dev_deg": round(st.aimer.dev_deg, 1),
            "lag_deg": round(st.aimer.lag_deg, 1), "sent": st.stats["sent"]}
