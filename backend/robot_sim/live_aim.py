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
#   ik_every                 streamed run: IK every Nth point (the answer is carried on between)
#   track_smooth / track_delay  trackJoint's smooth and delay scales
TUNE_DEFAULTS = {
    "smooth_s": 0.1, "deadband_mm": 1.0, "predict_s": 0.1,
    "aim_deadband_deg": 0.2, "aim_response_s": 0.15,
    "turn_dps": 30.0, "turn_acc": 90.0,
    "joint_smooth_s": 0.03, "joint_dps": 45.0, "joint_acc": 180.0,
    "ik_every": 1, "track_smooth": 0.5, "track_delay": 1.0,
}
TUNE_RANGES = {
    "smooth_s": (0.0, 5.0), "deadband_mm": (0.0, 50.0), "predict_s": (0.0, 0.5),
    "aim_deadband_deg": (0.0, 10.0), "aim_response_s": (0.0, 5.0),
    "turn_dps": (1.0, 180.0), "turn_acc": (1.0, 2000.0),
    "joint_smooth_s": (0.0, 1.0), "joint_dps": (1.0, 180.0), "joint_acc": (1.0, 2000.0),
    "ik_every": (1, 10), "track_smooth": (0.0, 1.0), "track_delay": (0.0, 2.0),
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
        ee_pos, ee_wxyz = rr._ee_from_cursor(pos, _wxyz(corr * r_base), "sim")
        # Seed and jump check from the last answer's delta on today's base joints.
        ref = q_base + self.delta_t
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
READ_EVERY = 10          # read the arm's joints every N points (each read is a network call)
QUEUE_EVERY = 3          # ask the controller's queue size every N points
FULL_STOP_S = 1.0        # the controller refusing points (queue full) this long -> stop
SETTLE_MAX_S = 0.5       # at the end, keep sending the last point while the aim settles
LEAD_POINTS = 5          # points kept queued ahead on the controller (100 ms): a late send doesn't starve it
QUEUE_MAX = 10           # with getTrajectoryQueueSize: don't send while this many are queued
RESYNC_S = 0.5           # a stall longer than this restarts the schedule (else late points go out back to back)
CATCHUP_MAX = 4          # behind the clock: up to this many points back to back in one go (each one tick)
COARSE_AFTER_S = 0.3     # only further behind than this does a point cover 2 ticks
SWITCH_S = 0.0005        # Python thread switch interval during a stream (default 5 ms): the sender
                         # waited up to 5 ms for the interpreter after each network call while the
                         # aimer / viewer / camera threads computed: ~13 points/s, 80 ms segments
AIM_DUTY = 0.5           # the aimer computes at most this share of the time (sleeps the rest)


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


def _thread_priority_high():
    """Windows: the sending thread above normal priority (it only waits and sends)."""
    import os
    if os.name != "nt":
        return
    try:
        import ctypes
        k = ctypes.windll.kernel32
        k.SetThreadPriority(k.GetCurrentThread(), 2)  # THREAD_PRIORITY_HIGHEST
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
    - the aimer (`_aim_loop`): as often as it can, the Aimer at the path time
      just ahead of the sender, on its own IK model; its answer (`aim_d`) and
      rate (`aim_v`) are what the sender's filter follows, carried on at that
      rate between answers.

    The path time moves on only when a point is taken (a refused point is
    sent again); the run stops (stopJoint) when the aim asks for more than the
    limits allow or the arm falls behind. stop() ends it."""

    def __init__(self, bot, samples):
        self.bot = bot
        self.samples = list(samples)
        self.t_end = float(self.samples[-1]["t"]) if self.samples else 0.0
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
        self.aim_v = np.zeros(6)      # its rate (rad/s)
        self.aim_at = None            # perf_counter of that answer
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
        self.aim_thread = threading.Thread(target=self._aim_loop, name="live_aim", daemon=True)
        self.aim_thread.start()
        self.thread = threading.Thread(target=self._run, name="live_stream", daemon=True)
        self.thread.start()

    def stop(self):
        self.stop_ev.set()
        for th in (self.thread, self.aim_thread):
            if th is not None and th is not threading.current_thread():
                th.join(1.0)

    # -- aimer thread -------------------------------------------------------
    def _aim_loop(self):
        last = None
        while not self.stop_ev.is_set():
            t0 = time.perf_counter()
            # Aim where the sender will be by the time this answer is used.
            s = min(self.t_end, self.s_sent + (LEAD_POINTS + 1) * STREAM_DT)
            if last is not None and s <= last + 1e-6:
                if s >= self.t_end - 1e-9:
                    time.sleep(0.005)
                    continue
                s = last + 0.002
            try:
                q_base = _interp(self.samples, s)
                self.aimer(s, q_base)
                d = np.asarray(self.aimer.delta_t, dtype=float).copy()
            except Exception as e:
                self.aimer.err = "aim failed: %s" % e
                time.sleep(0.02)
                continue
            now = time.perf_counter()
            if self.aim_at is not None and last is not None and s > last:
                cap = math.radians(tune("joint_dps"))
                self.aim_v = np.clip((d - self.aim_d) / (s - last), -cap, cap)
            if self.aimer.err:
                self.aim_v = np.zeros(6)  # held: don't carry it on
            self.aim_d, self.aim_at, last = d, now, s
            self.stats["aims"] += 1
            self.stats["max_aim_ms"] = max(self.stats["max_aim_ms"], (now - t0) * 1000.0)
            self.stats["aim_err"] = self.aimer.err
            # Not faster than the points go out, and sleeping at least as long as it
            # computed (AIM_DUTY): a busy aimer kept the sender waiting for the interpreter.
            took = time.perf_counter() - t0
            left = max(STREAM_DT - took, took * (1.0 - AIM_DUTY) / AIM_DUTY)
            if left > 0:
                time.sleep(left)

    def _delta_goal(self):
        """The aim's joint change for the next point: its last answer carried on
        at its rate (up to 0.15 s past it)."""
        if self.aim_at is None:
            return self.delta
        age = min(0.15, max(0.0, time.perf_counter() - self.aim_at))
        return self.aim_d + self.aim_v * age

    def _step_delta(self, dur, goal=None):
        self.delta, self.dv = _follow(self.delta, self.dv, self._delta_goal() if goal is None else goal,
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
        old_switch = sys.getswitchinterval()
        sys.setswitchinterval(SWITCH_S)
        _thread_priority_high()
        self._qfn = None
        t_wall0 = time.perf_counter()
        due = t_wall0 - LEAD_POINTS * STREAM_DT  # the first LEAD_POINTS go out at once
        s = 0.0                      # path time of the last point taken
        qb_prev = _interp(self.samples, 0.0)
        full_since = None
        n = 0
        settle = None                # wall time the end's settling began
        last_sent = None
        burst = 0                    # points sent back to back so far (catching up)
        qn = None
        try:
            while not self.stop_ev.is_set():
                if n % QUEUE_EVERY == 0 or (qn is not None and qn >= QUEUE_MAX):
                    qn = self._queue(mc)
                if qn is not None and qn >= QUEUE_MAX:
                    _sleep_until(time.perf_counter() + 0.004)  # plenty queued: wait
                    continue
                n += 1
                due += STREAM_DT
                if settle is None:
                    # One tick of path per point. Behind the clock, the next points go out back
                    # to back (still one tick each: even segments); only far behind does a
                    # point cover 2 ticks (80 ms points were the coarse segments on the arm).
                    behind = time.perf_counter() - t_wall0 - LEAD_POINTS * STREAM_DT - s
                    step = 2 * STREAM_DT if behind > COARSE_AFTER_S else STREAM_DT
                    s_new = min(s + step, self.t_end)
                    dur = max(STREAM_DT, s_new - s)
                    q_base = _interp(self.samples, s_new)
                    q = q_base + self._step_delta(dur)
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
                if qn is not None and rc == 0:
                    qn += 1  # our own point, until the next read
                if now - due > RESYNC_S:
                    self.stats["late"] += 1
                    due = now  # a long stall: start the schedule again (no burst)
                    burst = 0
                elif due < now and burst < CATCHUP_MAX:
                    burst += 1  # behind: the next point now
                    self.stats["catchup"] = self.stats.get("catchup", 0) + 1
                elif qn is not None and qn <= 1:
                    burst = 0
                    due = min(due, now)  # the controller is about to run dry: send the next one now
                else:
                    burst = 0
                    _sleep_until(max(due, now + 0.001) if due < now else due)
        except Exception as e:
            self.error = str(e)
            print("live stream stopped:", e)
            try:
                mc.stopJoint(3.0)
            except Exception:
                pass
        finally:
            _fine_timer(False)
            sys.setswitchinterval(old_switch)
            self.stop_ev.set()  # the aimer too
        self.stats["wall_s"] = round(time.perf_counter() - t_wall0, 3)
        self.stats["path_s"] = round(s, 3)
        print("live stream done", self.stats)


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
