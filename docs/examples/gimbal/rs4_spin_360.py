#!/usr/bin/env python3
"""DJI RS4: turn the camera a measured 360 deg (or any amount), closed loop.

The move this is for (sandbox set-up): the arm points the gimbal HANDLE at the
target, horizontally (handle end towards the robot, camera end towards the
target), the camera looks along the handle at the target, and the camera then
rolls a full turn around that same axis while still looking at the target.
That is DJI's 3D ROLL 360 mode: the motor on the handle axis rolls the camera
around its lens. So by default this script:

  * sets 3D ROLL 360 mode (--mode 3) and commands ROLL speed (--axis roll),
    as the reference demo did with set_roll360() and speed(roll=...);
  * reads BOTH angle sets the SDK has, every read:
      joint    - the motor angles, relative to the gimbal body (handle)
      attitude - the camera's orientation in the world (gravity)
    and, after a short probe, follows whichever angle really turned (printed),
    unwrapping it across its +/-180 jump;
  * sends a speed command every 50 ms (each lasts 0.5 s), slows near the end
    and stops on the measured angle, not on a timer;
  * stops with a message if the gimbal pulls the turn back or stops moving;
  * on exit (also Ctrl+C) stops the axes, hands speed control back and sets
    normal mode again (unless --keep-mode).

--axis yaw --mode 0 is the earlier yaw spin (it pans the camera when the
handle is horizontal, and sways back when the lens is along the yaw axis).

Examples (sandbox closed, from the repo folder):

    python docs/examples/gimbal/rs4_spin_360.py                    # roll +360 in 3D ROLL 360
    python docs/examples/gimbal/rs4_spin_360.py --turn -360        # the other way
    python docs/examples/gimbal/rs4_spin_360.py --mode 2           # ROLL 360 mode instead
    python docs/examples/gimbal/rs4_spin_360.py --axis yaw --mode 0

Uses the reference protocol code in user_files/gimbal_scripts/rs4_can.py.
Needs: pip install python-can pyusb libusb-package
"""
import argparse
import os
import struct
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.normpath(os.path.join(HERE, "..", "..", "..", "user_files", "gimbal_scripts")))
import rs4_can  # noqa: E402  (reference: DjiGimbal, BUS_KWARGS, build_packet)
from rs4_can import DjiGimbal, BUS_KWARGS, build_packet  # noqa: E402,F401

can = rs4_can.can

RESEND_S = 0.05       # speed command period
READ_EVERY_S = 0.25   # angle read period (joint + attitude)
READ_TIMEOUT_S = 0.2  # reply wait per read; a speed command is sent between the two reads
PROBE_S = 0.8         # time at the start to see which angle moves, and which way
STALL_S = 1.5         # no progress for this long -> stop
PULLBACK_DEG = 8.0    # moved this far against the command -> stop
NAMES = ("yaw", "roll", "pitch")
MODES = {0: "normal", 2: "ROLL 360", 3: "3D ROLL 360"}


def read_set(g, which, timeout=READ_TIMEOUT_S):
    """(yaw, roll, pitch) in degrees. which: 'joint' (0x02) or 'attitude' (0x01)."""
    g.seq = (g.seq + 1) & 0xFFFF
    g._send(build_packet(0x0E, 0x02, bytes([0x02 if which == "joint" else 0x01]), g.seq, 0x03))
    reply = g._read_reply(timeout)
    if reply is None or len(reply) < 10 or reply[0:2] != b"\x0e\x02" or reply[2] != 0x00:
        return None
    y, r, p = struct.unpack_from("<hhh", reply, 4)
    return y / 10, r / 10, p / 10


def wrap_step(new, old):
    """Real motion between two reads (removes the +/-180 jump)."""
    d = new - old
    if d > 180:
        d -= 360
    elif d < -180:
        d += 360
    return d


def fmt(a):
    return "  --- failed ---       " if a is None else "%7.1f %7.1f %7.1f" % a


class Tracker:
    """Speed sending plus unwrapped totals for all six angles."""

    def __init__(self, g, axis):
        self.g, self.axis = g, axis
        self.cmd = 0.0
        self.last = {"joint": None, "attitude": None}
        self.total = {(w, n): 0.0 for w in ("joint", "attitude") for n in NAMES}
        self.wraps = {k: 0 for k in self.total}
        self.read_t = 0.0
        self.send_t = None
        self.max_gap = 0.0
        self.fails = 0

    def send(self, dps=None):
        if dps is not None:
            self.cmd = dps
        now = time.time()
        if self.send_t is not None:
            self.max_gap = max(self.max_gap, now - self.send_t)
        if self.axis == "roll":
            self.g.speed(roll=self.cmd)
        else:
            self.g.speed(yaw=self.cmd)
        self.send_t = time.time()

    def _take(self, which, a):
        if a is None:
            self.fails += 1
            return
        old = self.last[which]
        if old is not None:
            for i, n in enumerate(NAMES):
                if abs(a[i] - old[i]) > 180:
                    self.wraps[(which, n)] += 1
                self.total[(which, n)] += wrap_step(a[i], old[i])
        self.last[which] = a

    def update(self, force=False):
        """Read both sets if due (speed resent between them). True if read."""
        if not force and time.time() - self.read_t < READ_EVERY_S:
            return False
        self.read_t = time.time()
        self._take("joint", read_set(self.g, "joint"))
        if self.cmd:
            self.send()
        self._take("attitude", read_set(self.g, "attitude"))
        return True

    def run_for(self, dps, seconds):
        end = time.time() + seconds
        while time.time() < end:
            self.send(dps)
            self.update()
            time.sleep(RESEND_S)


def spin(g, axis, turn, speed, tol, gain, min_speed, feedback):
    t = Tracker(g, axis)
    for _ in range(3):
        t.update(force=True)
        if t.last["joint"] and t.last["attitude"]:
            break
    if not t.last["joint"] and not t.last["attitude"]:
        raise RuntimeError("can't read the gimbal angles (no reply)")
    print("                    yaw    roll   pitch")
    print("start joint     %s" % fmt(t.last["joint"]))
    print("start attitude  %s" % fmt(t.last["attitude"]))

    # 1. Probe: which angle follows the command, and which way.
    for k in t.total:
        t.total[k] = 0.0
    probe = min(speed, 20.0) * (1 if turn >= 0 else -1)
    t.run_for(probe, PROBE_S)
    t.send(0.0)
    t.update(force=True)
    print("probe %+.0f deg/s %s for %.1f s moved:" % (probe, axis, PROBE_S))
    for which in ("joint", "attitude"):
        print("   %-9s yaw %+6.1f  roll %+6.1f  pitch %+6.1f" % (
            which, *(t.total[(which, n)] for n in NAMES)))
    if feedback == "auto":
        key = max(t.total, key=lambda k: abs(t.total[k]))
    else:
        w, _, n = feedback.partition("-")
        key = (w, n)
    moved = t.total[key]
    if abs(moved) < 2.0:
        raise RuntimeError("nothing turned during the probe (largest %s %s %+.1f deg): "
                           "speed control not taken over, or the axis is locked" % (key[0], key[1], moved))
    sign = 1.0 if (moved > 0) == (probe > 0) else -1.0
    print("following %s %s: a positive %s speed makes it go %s" % (key[0], key[1], axis, "UP" if sign > 0 else "DOWN"))
    turned0 = t.total[key]

    # 2. Closed loop on that angle (the probe's movement counts).
    t0 = time.time()
    limit_s = abs(turn) / max(min_speed, 1.0) + 10.0
    best = abs(turn - turned0)
    best_t = time.time()
    next_print = 0.0
    print("   t(s)   turned   to go   cmd    | joint  y / r / p        | attitude y / r / p")
    while True:
        turned = t.total[key]
        err = turn - turned
        if abs(err) <= tol:
            break
        want = max(min_speed, min(speed, gain * abs(err))) * (1 if err > 0 else -1)
        t.send(want * sign)
        if t.update():
            e = abs(turn - t.total[key])
            if e < best - 0.5:
                best, best_t = e, time.time()
            elif e > best + PULLBACK_DEG:
                raise RuntimeError("pulled back %.1f deg against the command at %+.1f deg turned: the gimbal "
                                   "is undoing the turn (mode not taken? check the mode read-back)"
                                   % (e - best, t.total[key]))
            if time.time() - best_t > STALL_S:
                raise RuntimeError("no progress for %.1f s at %+.1f deg turned (limit, mode or follow?)"
                                   % (STALL_S, t.total[key]))
            if time.time() - t0 >= next_print:
                next_print += 0.5
                print("  %6.2f  %+7.1f  %+6.1f  %+6.1f   | %s | %s" % (
                    time.time() - t0, t.total[key], turn - t.total[key], t.cmd,
                    fmt(t.last["joint"]), fmt(t.last["attitude"])))
        if time.time() - t0 > limit_s:
            raise RuntimeError("took longer than %.0f s; stopped at %+.1f deg" % (limit_s, t.total[key]))
        time.sleep(RESEND_S)
    elapsed = time.time() - t0
    t.cmd = 0.0
    g.speed(0, 0, 0)
    time.sleep(0.6)
    t.update(force=True)
    print("end joint       %s" % fmt(t.last["joint"]))
    print("end attitude    %s" % fmt(t.last["attitude"]))
    print("-" * 72)
    print("turned          %+.1f deg on %s %s (asked %+.1f, error %+.1f)" % (
        t.total[key], key[0], key[1], turn, t.total[key] - turn))
    print("all six totals  joint y/r/p %+.1f %+.1f %+.1f | attitude y/r/p %+.1f %+.1f %+.1f" % (
        *(t.total[("joint", n)] for n in NAMES), *(t.total[("attitude", n)] for n in NAMES)))
    wr = ", ".join("%s %s x%d" % (k[0], k[1], v) for k, v in t.wraps.items() if v) or "none"
    print("wraps (+/-180)  %s" % wr)
    print("time            %.1f s, %d failed reads, longest command gap %.0f ms" % (elapsed, t.fails, t.max_gap * 1000))


def show_mode(g, label):
    reply = g.send_command(0x0E, 0x0B, bytes([0x22]))
    print("mode %-8s read-back: %s" % (label, reply.hex(" ") if reply else None))


def main():
    ap = argparse.ArgumentParser(description="RS4 closed-loop camera turn (default: roll 360 in 3D ROLL 360 mode)")
    ap.add_argument("--axis", choices=("roll", "yaw"), default="roll", help="speed axis to command (default roll)")
    ap.add_argument("--turn", type=float, default=360.0, help="degrees to turn (default 360; negative = other way)")
    ap.add_argument("--speed", type=float, default=30.0, help="top speed in deg/s (default 30)")
    ap.add_argument("--min-speed", type=float, default=3.0, help="slowest speed near the end (default 3)")
    ap.add_argument("--gain", type=float, default=1.5, help="deg/s per degree still to go near the end (default 1.5)")
    ap.add_argument("--tol", type=float, default=1.0, help="stop within this many degrees (default 1)")
    ap.add_argument("--mode", type=int, choices=(0, 2, 3), default=None,
                    help="0 normal, 2 ROLL 360, 3 3D ROLL 360 (default: 3 for --axis roll, unchanged for yaw)")
    ap.add_argument("--keep-mode", action="store_true", help="leave the mode set at the end")
    ap.add_argument("--feedback", default="auto",
                    choices=("auto", "joint-yaw", "joint-roll", "joint-pitch", "attitude-yaw", "attitude-roll", "attitude-pitch"),
                    help="angle to stop on (default auto: whichever moved most in the probe)")
    args = ap.parse_args()
    mode = args.mode if args.mode is not None else (3 if args.axis == "roll" else None)

    print("RS4 turn %+.0f deg on %s, mode %s: interface %s, channel %s, bitrate %s" % (
        args.turn, args.axis, MODES.get(mode, "unchanged"),
        BUS_KWARGS["interface"], BUS_KWARGS["channel"], BUS_KWARGS["bitrate"]))
    with can.Bus(**BUS_KWARGS) as bus:
        g = DjiGimbal(bus)
        changed = False
        try:
            show_mode(g, "before")
            if mode is not None:
                g.set_roll360(mode)
                changed = mode != 0
                time.sleep(0.8)
                show_mode(g, "set")
            spin(g, args.axis, args.turn, args.speed, args.tol, args.gain, args.min_speed, args.feedback)
            print("DONE")
        except KeyboardInterrupt:
            print("\nstopped (Ctrl+C)")
        except RuntimeError as e:
            print("\nSTOPPED: %s" % e)
        finally:
            g.speed(0, 0, 0)
            g.speed(take_over=False)
            if changed and not args.keep_mode:
                g.set_roll360(0)
                show_mode(g, "after")


if __name__ == "__main__":
    main()
