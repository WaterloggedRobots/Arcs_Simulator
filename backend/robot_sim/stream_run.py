"""Stream a baked path to a real AUBO arm with ARCS trackJoint, with a live
speed / direction control (stand-alone test; not used by the sandbox yet).

The normal Run uploads the whole path as a controller pathBuffer, which can
only be stopped. Here the sandbox owns the path clock instead: a thread at a
fixed period moves a path time `s` by `rate * dt` and sends the joints at `s`
with `trackJoint(q, dt, smooth, delay)`. `rate` is path seconds per second
(1 = as baked, 0 = hold, -1 = backwards along the path as baked), ramped with
an acceleration limit and a low-pass, capped by the joint speed limit and
braked so it comes to rest at the path's ends. Changing direction is the rate
passing smoothly through 0: no stop, no re-upload. The lag is the
controller's queue (about 0.1 s) plus the ramp.

Notes on trackJoint (from another ARCS project, i5H / ARCS 0.32.1; not yet
checked on this arm): at most ~100 points/s; rc 2 = queue full (point
dropped); after any stopJoint / stopLine it answers -4 then -3 until
clearPath(); -22 = the controller predicts a self-collision. A stream that
just ends while moving is an instant velocity step (it tripped the collision
stop there), so the end of a run always ramps to 0 first.

Usage (from the repo root):
    python backend/robot_sim/stream_run.py PATH.json [--ip 192.168.2.124] [--dry]
PATH.json is an Export (File > Export) or a saved project (.arcspath) with a
baked path: its "points" (joints in rad, "t" or "dur"). --sandbox URL takes
the running sandbox's current path through /api/file/export instead.

Keys (type then Enter): f forward, r reverse, space / p hold, + / - speed
x1.25, a number sets the rate (e.g. -0.5), b bounce at the ends on / off,
q stop smoothly and quit, x hard stop (stopJoint) and quit.
"""
from __future__ import annotations

import argparse
import bisect
import csv
import json
import math
import os
import sys
import threading
import time
import urllib.request
from pathlib import Path

# trackJoint return codes (see the notes above).
RC_OK = 0
RC_QUEUE_FULL = 2
RC_PATH_STOPPED = (-3, -4)
RC_SELF_COLLISION = -22

DEFAULT_DT = 0.02           # s between points (50 Hz; stay at or below 100 Hz)
DEFAULT_ACC = 1.0           # rate change per s^2 (1 -> -1 takes 2 s)
DEFAULT_SMOOTH_S = 0.15     # low-pass on the rate (s): no corner in the speed
DEFAULT_MAX_DPS = 60.0      # joint speed cap, deg/s
STEP_MAX_DEG = 5.0          # a bigger joint step in one tick stops the run
START_TOL_DEG = 1.0         # arm further than this from the first point -> approach
APPROACH_DPS = 20.0         # moveJoint speed to the first point
READ_EVERY = 5              # read the arm's joints every N ticks (log only)
TRACK_SMOOTH = 0.5          # trackJoint smooth / delay scale (no measured effect elsewhere)
TRACK_DELAY = 1.0


# ---------------------------------------------------------------------------
# path
# ---------------------------------------------------------------------------
class Trajectory:
    """Joints over path time, cubic (Catmull-Rom on the sample times) between
    the baked samples, so the speed has no corner at each sample."""

    def __init__(self, times, joints):
        if len(times) < 2:
            raise ValueError("the path needs at least 2 baked points")
        self.t = [float(v) for v in times]
        self.q = [[float(v) for v in q[:6]] for q in joints]
        for a, b in zip(self.t, self.t[1:]):
            if b <= a:
                raise ValueError("point times must rise (%.4f then %.4f)" % (a, b))
        self.T = self.t[-1]
        n = len(self.t)
        # Tangents (dq/ds) at each sample: centred differences, one-sided at the ends.
        self.m = []
        for i in range(n):
            i0, i1 = max(0, i - 1), min(n - 1, i + 1)
            dt = self.t[i1] - self.t[i0]
            self.m.append([(b - a) / dt for a, b in zip(self.q[i0], self.q[i1])])

    @classmethod
    def from_doc(cls, doc):
        pts = doc.get("points") or []
        if not pts:
            raise ValueError("no baked points in the file (bake the path, then save or export)")
        times, joints, t = [], [], 0.0
        for i, p in enumerate(pts):
            if p.get("t") is not None:
                t = float(p["t"])
            elif i:
                t += float(p.get("dur") or 0.0)
            if times and t <= times[-1]:
                continue        # a zero-length hop (dwell sample at the same time)
            times.append(t)
            joints.append(p["joints"])
        t0 = times[0]
        return cls([v - t0 for v in times], joints)

    def _seg(self, s):
        i = bisect.bisect_right(self.t, s) - 1
        return min(max(i, 0), len(self.t) - 2)

    def at(self, s):
        s = min(max(s, 0.0), self.T)
        i = self._seg(s)
        t0, t1 = self.t[i], self.t[i + 1]
        h = t1 - t0
        u = (s - t0) / h
        h00 = 2 * u ** 3 - 3 * u ** 2 + 1
        h10 = u ** 3 - 2 * u ** 2 + u
        h01 = -2 * u ** 3 + 3 * u ** 2
        h11 = u ** 3 - u ** 2
        q0, q1, m0, m1 = self.q[i], self.q[i + 1], self.m[i], self.m[i + 1]
        return [h00 * a + h10 * h * c + h01 * b + h11 * h * d
                for a, b, c, d in zip(q0, q1, m0, m1)]

    def max_slope(self, s, span=0.25):
        """Largest |dq/ds| (rad per path second) over [s - span, s + span]."""
        i0 = self._seg(max(0.0, s - span))
        i1 = self._seg(min(self.T, s + span)) + 1
        return max(abs(v) for i in range(i0, i1 + 1) for v in self.m[i])


def load_doc(path=None, sandbox=None):
    if sandbox:
        req = urllib.request.Request(
            sandbox.rstrip("/") + "/api/file/export", data=b"{}",
            headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=30) as r:
            out = json.load(r)
        if not out.get("ok"):
            raise RuntimeError("sandbox export failed")
        return out["doc"]
    return json.loads(Path(path).read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# rate (path seconds per second)
# ---------------------------------------------------------------------------
class RateControl:
    """The wanted rate (set from any thread) shaped into the rate the stream
    uses: acceleration-limited, low-passed, capped by the joint speed limit,
    braked to rest at the path's ends."""

    def __init__(self, traj, acc=DEFAULT_ACC, smooth_s=DEFAULT_SMOOTH_S,
                 max_dps=DEFAULT_MAX_DPS, bounce=False):
        self.traj = traj
        self.acc = float(acc)
        self.smooth_s = float(smooth_s)
        self.max_rad_s = math.radians(float(max_dps))
        self.bounce = bool(bounce)
        self.want = 0.0          # set by the user / a speed algorithm
        self.ramp = 0.0          # after the acceleration limit
        self.rate = 0.0          # after the low-pass: what the stream uses
        self.capped = False
        self._lock = threading.Lock()

    def set(self, rate):
        with self._lock:
            self.want = float(rate)

    def get_want(self):
        with self._lock:
            return self.want

    def step(self, s, dt):
        want = self.get_want()
        # Joint speed cap: |rate| * |dq/ds| <= max joint speed.
        slope = self.traj.max_slope(s)
        cap = self.max_rad_s / slope if slope > 1e-9 else float("inf")
        self.capped = abs(want) > cap
        want = max(-cap, min(cap, want))
        # Brake for the path's ends: come to rest at 0 / T at the acceleration limit.
        left = (self.traj.T - s) if self.ramp > 0 else s
        brake = math.sqrt(max(0.0, 2.0 * self.acc * max(0.0, left - 0.5 * self.smooth_s * abs(self.ramp))))
        at_end = (s >= self.traj.T - 1e-6 and want > 0) or (s <= 1e-6 and want < 0)
        if self.bounce and at_end and abs(self.rate) < 1e-3:
            # At an end and stopped, still asked outwards: turn round.
            want = -want
            with self._lock:
                self.want = want
        if self.ramp > 0:
            want = min(want, brake)
        elif self.ramp < 0:
            want = max(want, -brake)
        dv = self.acc * dt
        self.ramp += max(-dv, min(dv, want - self.ramp))
        k = 1.0 if self.smooth_s <= 0 else 1.0 - math.exp(-dt / self.smooth_s)
        self.rate += (self.ramp - self.rate) * k
        if (s <= 0.0 and self.rate < 0) or (s >= self.traj.T and self.rate > 0):
            self.rate = self.ramp = 0.0
        return self.rate


# ---------------------------------------------------------------------------
# arms
# ---------------------------------------------------------------------------
class RealArm:
    """The arm over ARCS RPC (arcscontrol.RobotArcs, as the sandbox uses it)."""

    def __init__(self, ip, port):
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import arcscontrol
        self.bot = arcscontrol.RobotArcs(ip=ip or arcscontrol.robot_ip, port=port or arcscontrol.robot_port)
        if not self.bot.connect(force=True):
            raise RuntimeError("could not connect to the controller at %s" % self.bot.ip)
        self.mc = self.bot.mc

    def joints(self):
        return self.bot.GetJoints()

    def prepare(self):
        # Resume the queue, leave servo mode, 100 % speed (as for a speed jog).
        self.bot.PrepareJog()
        try:
            self.mc.clearPath()
        except Exception:
            pass

    def approach(self, q, dps):
        v = math.radians(dps)
        self.bot.MoveJAwait(list(q), jvel=v, jacc=v * 2)

    def track(self, q, dt):
        return int(self.mc.trackJoint([float(v) for v in q], float(dt), TRACK_SMOOTH, TRACK_DELAY))

    def clear_path(self):
        self.mc.clearPath()

    def queue_size(self):
        try:
            return int(self.mc.getTrajectoryQueueSize())
        except Exception:
            return None

    def hard_stop(self):
        try:
            self.mc.stopJoint(3.0)
        except Exception:
            pass

    def close(self):
        try:
            self.bot.Close()
        except Exception:
            pass


class DryArm:
    """No controller: the 'arm' follows the points ~0.1 s late (a queue of
    points played at their dt), for trying the stream and the keys."""

    def __init__(self, q0):
        self.q = list(q0)
        self.queue = []
        self.t_last = time.perf_counter()
        self.sent = 0

    def _play(self):
        now = time.perf_counter()
        while self.queue and self.queue[0][0] <= now:
            self.q = self.queue.pop(0)[1]

    def joints(self):
        self._play()
        return list(self.q)

    def prepare(self):
        pass

    def approach(self, q, dps):
        print("dry: approach to the first point")
        self.q = list(q)

    def track(self, q, dt):
        self._play()
        if len(self.queue) > 10:
            return RC_QUEUE_FULL
        due = (self.queue[-1][0] if self.queue else time.perf_counter() + 0.1) + dt
        self.queue.append((due, list(q)))
        self.sent += 1
        return RC_OK

    def clear_path(self):
        self.queue = []

    def queue_size(self):
        return len(self.queue)

    def hard_stop(self):
        self.queue = []

    def close(self):
        pass


# ---------------------------------------------------------------------------
# stream
# ---------------------------------------------------------------------------
class Streamer:
    """The fixed-period sender. `adjust(s, q) -> q` (optional) can change each
    point before it is sent (a live Look At or a moving goal, later)."""

    def __init__(self, traj, arm, rate, dt=DEFAULT_DT, log_path=None, adjust=None, s0=0.0):
        self.traj, self.arm, self.rc, self.dt = traj, arm, rate, float(dt)
        self.adjust = adjust
        self.s = float(s0)
        self.q_last = None
        self.quit = threading.Event()      # smooth stop, then end
        self.abort = threading.Event()     # hard stop now
        self.done = threading.Event()
        self.error = ""
        self.stats = {"sent": 0, "full": 0, "cleared": 0, "late": 0, "max_err_deg": 0.0}
        self._log = None
        if log_path:
            Path(log_path).parent.mkdir(parents=True, exist_ok=True)
            self._log_f = open(log_path, "w", newline="")
            self._log = csv.writer(self._log_f)
            self._log.writerow(["time", "s", "want", "rate", "rc", "queue"]
                               + ["q%d" % i for i in range(1, 7)] + ["a%d" % i for i in range(1, 7)])

    def run(self):
        k, t0 = 0, time.perf_counter()
        clear_pending = False
        try:
            while not self.abort.is_set():
                k += 1
                due = t0 + k * self.dt
                if self.quit.is_set():
                    self.rc.set(0.0)
                    if abs(self.rc.rate) < 1e-3 and abs(self.rc.ramp) < 1e-3:
                        break
                if clear_pending:
                    self.arm.clear_path()
                    self.stats["cleared"] += 1
                    clear_pending = False
                    self.q_last = None
                    self._sleep_until(due)
                    continue
                rate = self.rc.step(self.s, self.dt)
                s_new = min(max(self.s + rate * self.dt, 0.0), self.traj.T)
                q = self.traj.at(s_new)
                if self.adjust is not None:
                    q = self.adjust(s_new, q)
                if self.q_last is not None:
                    step = max(abs(a - b) for a, b in zip(q, self.q_last))
                    if math.degrees(step) > STEP_MAX_DEG:
                        raise RuntimeError("a %.1f deg joint step in one tick at s %.2f"
                                           % (math.degrees(step), s_new))
                rc = self.arm.track(q, self.dt)
                if rc == RC_OK:
                    self.s, self.q_last = s_new, q
                    self.stats["sent"] += 1
                elif rc == RC_QUEUE_FULL:
                    self.stats["full"] += 1          # dropped: resend from the same s next tick
                elif rc in RC_PATH_STOPPED:
                    clear_pending = True             # a stop left the path held
                elif rc == RC_SELF_COLLISION:
                    raise RuntimeError("the controller predicts a self-collision at s %.2f" % s_new)
                else:
                    raise RuntimeError("trackJoint rc %s at s %.2f" % (rc, s_new))
                if self._log is not None or k % READ_EVERY == 0:
                    self._sample(k, rc)
                if time.perf_counter() > due + self.dt:
                    self.stats["late"] += 1
                self._sleep_until(due)
        except Exception as e:
            self.error = str(e)
            print("stream: stopping:", e)
            self.arm.hard_stop()
        finally:
            if self.abort.is_set():
                self.arm.hard_stop()
            self.done.set()
            if self._log is not None:
                self._log_f.close()

    def _sample(self, k, rc):
        a = self.arm.joints() if k % READ_EVERY == 0 else None
        if a is not None and self.q_last is not None:
            err = max(abs(x - y) for x, y in zip(a, self.q_last))
            self.stats["max_err_deg"] = max(self.stats["max_err_deg"], math.degrees(err))
        if self._log is not None:
            qs = self.arm.queue_size() if k % READ_EVERY == 0 else None
            self._log.writerow(
                ["%.4f" % time.perf_counter(), "%.4f" % self.s, "%.3f" % self.rc.get_want(),
                 "%.4f" % self.rc.rate, rc, "" if qs is None else qs]
                + ["%.5f" % v for v in (self.q_last or [math.nan] * 6)]
                + (["%.5f" % v for v in a] if a else [""] * 6))

    @staticmethod
    def _sleep_until(t):
        while True:
            left = t - time.perf_counter()
            if left <= 0:
                return
            time.sleep(min(left, 0.005) if left > 0.002 else 0)


def _fine_timer():
    """Windows sleeps in 15.6 ms steps unless the timer is set to 1 ms."""
    if os.name == "nt":
        try:
            import ctypes
            ctypes.windll.winmm.timeBeginPeriod(1)
        except Exception:
            pass


def _keys(st, rc):
    last = 1.0
    print(__doc__.split("Keys")[1].strip().split("\n\n")[0].replace("(type then Enter):", "keys:"))
    while not st.done.is_set():
        try:
            line = input().strip().lower()
        except EOFError:
            line = "q"
        w = rc.get_want()
        if w:
            last = abs(w)
        if line == "f":
            rc.set(last)
        elif line == "r":
            rc.set(-last)
        elif line in ("", "p", " "):
            rc.set(0.0)
        elif line == "+":
            rc.set(w * 1.25 if w else last * 1.25)
        elif line == "-":
            rc.set(w / 1.25 if w else last / 1.25)
        elif line == "b":
            rc.bounce = not rc.bounce
            print("bounce", "on" if rc.bounce else "off")
        elif line == "q":
            st.quit.set()
            return
        elif line == "x":
            st.abort.set()
            return
        else:
            try:
                rc.set(float(line))
            except ValueError:
                print("?", line)
                continue
        print("want %.2f  rate %.2f  s %.2f / %.2f%s" % (
            rc.get_want(), rc.rate, st.s, st.traj.T, "  (joint speed cap)" if rc.capped else ""))


def main(argv=None):
    ap = argparse.ArgumentParser(description="Stream a baked path with trackJoint, live speed / direction.")
    ap.add_argument("path", nargs="?", help="Export .json or project .arcspath with a baked path")
    ap.add_argument("--sandbox", help="take the path from a running sandbox, e.g. http://127.0.0.1:5000")
    ap.add_argument("--ip"), ap.add_argument("--port", type=int)
    ap.add_argument("--dry", action="store_true", help="no controller: a stand-in arm")
    ap.add_argument("--dt", type=float, default=DEFAULT_DT)
    ap.add_argument("--acc", type=float, default=DEFAULT_ACC, help="rate change per s^2")
    ap.add_argument("--smooth", type=float, default=DEFAULT_SMOOTH_S, help="rate low-pass, s")
    ap.add_argument("--max-dps", type=float, default=DEFAULT_MAX_DPS)
    ap.add_argument("--rate", type=float, default=1.0, help="rate to start at (0 = wait for a key)")
    ap.add_argument("--bounce", action="store_true", help="turn round at the path's ends")
    ap.add_argument("--log", help="CSV log (default exports/stream_run_<time>.csv)")
    ap.add_argument("--script", help="rates over time instead of keys: 't:rate,t:rate,...' then quit")
    a = ap.parse_args(argv)
    if not a.path and not a.sandbox:
        ap.error("give a path file or --sandbox URL")
    if not 0.005 <= a.dt <= 0.1:
        ap.error("--dt must be 0.005-0.1 s (the controller takes up to ~100 points/s)")

    doc = load_doc(a.path, a.sandbox)
    traj = Trajectory.from_doc(doc)
    print("path: %d points, %.2f s" % (len(traj.t), traj.T))
    log = a.log
    if log is None:
        try:
            sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
            import app_paths
            log = str(app_paths.EXPORTS_DIR / time.strftime("stream_run_%Y%m%d_%H%M%S.csv"))
        except Exception:
            log = time.strftime("stream_run_%Y%m%d_%H%M%S.csv")

    q0 = traj.at(0.0)
    arm = DryArm(q0) if a.dry else RealArm(a.ip or doc.get("robot_ip"), a.port or doc.get("robot_port"))
    _fine_timer()
    try:
        cur = arm.joints()
        if cur is None:
            raise RuntimeError("cannot read the arm's joints")
        off = max(abs(x - y) for x, y in zip(cur, q0))
        if math.degrees(off) > START_TOL_DEG:
            ans = input("arm is %.1f deg from the first point: moveJoint there at %d deg/s? [y/N] "
                        % (math.degrees(off), APPROACH_DPS)) if not a.dry else "y"
            if ans.strip().lower() != "y":
                return 1
            arm.approach(q0, APPROACH_DPS)
        arm.prepare()
        rc = RateControl(traj, a.acc, a.smooth, a.max_dps, a.bounce)
        rc.set(a.rate)
        st = Streamer(traj, arm, rc, a.dt, log)
        th = threading.Thread(target=st.run, name="stream", daemon=True)
        th.start()
        if a.script:
            steps = [tuple(float(v) for v in p.split(":")) for p in a.script.split(",")]
            t_start = time.perf_counter()
            for t, r in steps:
                while time.perf_counter() - t_start < t and not st.done.is_set():
                    time.sleep(0.01)
                rc.set(r)
                print("t %.1f  want %.2f  s %.2f" % (time.perf_counter() - t_start, r, st.s))
            st.quit.set()
        else:
            threading.Thread(target=_keys, args=(st, rc), daemon=True).start()
        while th.is_alive():
            th.join(0.2)
        print("done: s %.2f / %.2f, %s%s" % (st.s, traj.T, st.stats, ("  error: " + st.error) if st.error else ""))
        print("log:", log)
        return 1 if st.error else 0
    except KeyboardInterrupt:
        arm.hard_stop()
        return 1
    finally:
        arm.close()


if __name__ == "__main__":
    sys.exit(main())
