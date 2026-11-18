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
JUMP_MAX_DEG = 4.0      # a joint jump bigger than this between IK answers is refused
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
# Small corrections ease in: the turn's speed is also at most (what is left) /
# AIM_RESPONSE_S, so a 1 deg correction starts at 2 deg/s and slows as it
# closes (with only the speed / acceleration limits it was a quick
# accelerate-and-brake step of ~10 deg/s for every small target movement).
AIM_RESPONSE_S = 0.5
JOINT_DPS = 45.0        # the aim's joint change on top of the baked path, deg/s
JOINT_ACC = 180.0       # ... and its acceleration, deg/s^2
JOINT_SMOOTH_S = 0.08
IK_DEADBAND_DEG = 0.02


def _limits():
    rr = _rr()
    try:
        vmax = float(rr._track.get("turn_dps", rr.TURN_DPS_DEFAULT))
        amax = float(rr._track.get("turn_acc", rr.TURN_ACC_DEFAULT))
    except Exception:
        vmax, amax = 30.0, 90.0
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
        speed = min(vmax, math.sqrt(2.0 * amax * dist), dist / AIM_RESPONSE_S, dist / dt)
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


# Streamed run safety. The first version dropped a point the controller
# refused (queue full) while its clock ran on, and let any joint step up to
# 5 deg per point (250 deg/s) to catch up: after a delay the arm sped up hard.
STEP_MARGIN_DPS = 20.0   # a point may move this much faster than the path + aim limits ...
STOP_BEHIND_DEG = 3.0    # ... and the stream stops if it falls this far behind what it should send
FOLLOW_STOP_DEG = 10.0   # the arm this far from the last point sent -> stop
READ_EVERY = 5           # read the arm's joints every N points
FULL_STOP_S = 1.0        # the controller refusing points (queue full) this long -> stop
SETTLE_MAX_S = 1.5       # at the end, keep sending the last point while the aim settles
LOG_MAX = 6000           # points kept for the speed export


class Streamer:
    """Sends the baked path (aimed live) with trackJoint, a point every
    STREAM_DT; the path time moves on only when a point is taken (a refused
    point is sent again), every point stays within the joint speed the path
    and the aim allow, and the run stops (stopJoint) when it can't keep to that
    or the arm falls behind. stop() ends it (the run's stop then stops the arm)."""

    def __init__(self, bot, samples):
        self.bot = bot
        self.samples = list(samples)
        self.t_end = float(self.samples[-1]["t"]) if self.samples else 0.0
        self.aimer = Aimer()
        self.stop_ev = threading.Event()
        self.thread = None
        self.stats = {"sent": 0, "full": 0, "cleared": 0, "late": 0, "aim_err": "",
                      "max_follow_deg": 0.0, "max_behind_deg": 0.0, "max_dps": 0.0}
        self.error = ""
        self.log = []
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
        self.thread = threading.Thread(target=self._run, name="live_stream", daemon=True)
        self.thread.start()

    def stop(self):
        self.stop_ev.set()
        if self.thread is not None and self.thread is not threading.current_thread():
            self.thread.join(1.0)

    def _limit(self, q, q_base, qb_prev):
        """q held to the path's own step plus the aim's speed (and a margin)."""
        lim = np.abs(q_base - qb_prev) + math.radians(JOINT_DPS + STEP_MARGIN_DPS) * STREAM_DT
        step = np.clip(q - self.q_last, -lim, lim)
        out = self.q_last + step
        behind = float(np.degrees(np.max(np.abs(q - out))))
        self.stats["max_behind_deg"] = max(self.stats["max_behind_deg"], behind)
        if behind > STOP_BEHIND_DEG:
            raise RuntimeError("the aimed path asks for a %.1f deg jump, more than the arm may make: stopped" % behind)
        return out

    def _send(self, mc, q):
        return int(mc.trackJoint([float(v) for v in q], STREAM_DT, 0.5, 1.0))

    def _run(self):
        mc = self.bot.mc
        t_wall0 = time.perf_counter()
        due = t_wall0
        s = 0.0                      # path time of the last point taken
        qb_prev = _interp(self.samples, 0.0)
        full_since = None
        n = 0
        settle = None                # wall time the end's settling began
        try:
            while not self.stop_ev.is_set():
                n += 1
                due += STREAM_DT
                if settle is None:
                    s_new = min(s + STREAM_DT, self.t_end)
                    q_base = _interp(self.samples, s_new)
                    q = self.aimer(s_new, q_base, solve=(AIM_EVERY <= 1 or n % AIM_EVERY == 1))
                    self.stats["aim_err"] = self.aimer.err
                else:
                    # Past the end: the last point, the aim's joints settling (no new aim).
                    s_new = self.t_end
                    q_base = qb_prev
                    self.aimer.rate_t = np.zeros(6)
                    q = self._settle_q(q_base)
                q = self._limit(q, q_base, qb_prev)
                rc = self._send(mc, q)
                if rc == 0:
                    dps = float(np.degrees(np.max(np.abs(q - self.q_last)))) / STREAM_DT
                    self.stats["max_dps"] = max(self.stats["max_dps"], dps)
                    self.q_last = q
                    s, qb_prev = s_new, q_base
                    self.stats["sent"] += 1
                    full_since = None
                elif rc == 2:
                    # Queue full: the same point again next time (the path time waits).
                    self.stats["full"] += 1
                    full_since = full_since or time.perf_counter()
                    if time.perf_counter() - full_since > FULL_STOP_S:
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
                    self.log.append({"t": round(time.perf_counter() - t_wall0, 4), "s": round(s, 4), "rc": rc,
                                     "q": [round(math.degrees(v), 3) for v in self.q_last],
                                     "follow_deg": None if follow is None else round(follow, 2),
                                     "dev_deg": round(self.aimer.dev_deg, 2), "aim_err": self.aimer.err})
                if settle is None and s >= self.t_end - 1e-9:
                    settle = time.perf_counter()
                if settle is not None and (time.perf_counter() - settle > SETTLE_MAX_S
                                           or float(np.max(np.abs(self.aimer.dv))) < math.radians(0.5)):
                    break  # the last point is sent and the aim has come to rest
                left = due - time.perf_counter()
                if left > 0:
                    time.sleep(left)
                else:
                    self.stats["late"] += 1
                    due = time.perf_counter()  # late: don't send a burst to catch up
        except Exception as e:
            self.error = str(e)
            print("live stream stopped:", e)
            try:
                mc.stopJoint(3.0)
            except Exception:
                pass
        print("live stream done", self.stats)

    def _settle_q(self, q_base):
        """The aim's joints coming to rest on the last answer (filter only)."""
        a = self.aimer
        a.delta, a.dv = _follow(a.delta, a.dv, a.delta_t, math.radians(JOINT_DPS),
                                math.radians(JOINT_ACC), STREAM_DT, JOINT_SMOOTH_S)
        return np.asarray(q_base, dtype=float) + a.delta


_stream = {"s": None}


def stream_start(bot, samples):
    stream_stop()
    st = Streamer(bot, samples)
    _stream["s"] = st
    _stream["last"] = st
    st.start()
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
    return {"stats": dict(st.stats), "error": st.error, "dt": STREAM_DT, "points": list(st.log)}


def stream_status():
    st = _stream.get("s")
    if st is None:
        return {}
    return {"err": st.error or st.aimer.err, "dev_deg": round(st.aimer.dev_deg, 1),
            "lag_deg": round(st.aimer.lag_deg, 1), "sent": st.stats["sent"]}
