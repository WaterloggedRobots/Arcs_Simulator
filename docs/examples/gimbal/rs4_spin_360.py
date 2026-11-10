#!/usr/bin/env python3
"""DJI RS4: spin yaw a measured 360 deg (or any amount), closed loop.

What the earlier tests showed, and how this script deals with it:

  * Absolute position moves can't go past +/-180 deg, so a full turn uses
    SPEED control (0x0E/0x01), resent every 50 ms (each command lasts 0.5 s).
  * The yaw reading WRAPS at +/-180. The script unwraps it (a jump of more
    than 180 deg between two reads is the wrap, not motion) to know how far
    the gimbal has really turned, and stops on that angle instead of on a
    timer. It slows down near the end so it lands on the target.
  * A positive speed may make the reading go DOWN. The first half second
    measures which way the reading moves for a positive speed and uses that.
  * With the lens along the yaw axis (pitch near +/-90) a yaw spin turns the
    camera around its lens; in normal mode the stabiliser pulls that back
    after ~50 deg (the gimbal sways). ROLL 360 / 3D ROLL 360 mode allows the
    full turn (--mode 2 / 3). If the gimbal stops making progress or is
    pulled back, the script stops and says so.
  * On exit (also Ctrl+C or an error) it stops the axes, hands speed control
    back and, if it changed the mode, sets normal mode again.

Examples:

    python docs/examples/gimbal/rs4_spin_360.py                  # +360 (reading goes up)
    python docs/examples/gimbal/rs4_spin_360.py --turn -360      # the other way
    python docs/examples/gimbal/rs4_spin_360.py --turn 720 --speed 45
    python docs/examples/gimbal/rs4_spin_360.py --mode 3 --pitch 90   # spin around the lens

--turn is in the gimbal's yaw reading: +360 means the reading goes up by a
full turn. Which way that is when seen from above is printed after the spin
test; check it once on your gimbal.

Uses the reference protocol code in user_files/gimbal_scripts/rs4_can.py.
Needs: pip install python-can pyusb libusb-package
"""
import argparse
import math
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.normpath(os.path.join(HERE, "..", "..", "..", "user_files", "gimbal_scripts")))
import rs4_can  # noqa: E402  (reference: DjiGimbal, BUS_KWARGS, build_packet)
from rs4_can import DjiGimbal, BUS_KWARGS, build_packet  # noqa: E402,F401

can = rs4_can.can

RESEND_S = 0.05      # speed command period
READ_EVERY_S = 0.2   # angle read period
READ_TIMEOUT_S = 0.3  # keeps a read (and the gap between speed commands) under 0.5 s
PROBE_S = 0.6        # time at the start to learn which way the reading moves
STALL_S = 1.5        # no progress for this long -> stop
PULLBACK_DEG = 8.0   # moved this far against the command -> stop


def read_angles(g, timeout=READ_TIMEOUT_S):
    """Reference get_angles with a short reply wait."""
    g.seq = (g.seq + 1) & 0xFFFF
    g._send(build_packet(0x0E, 0x02, bytes([0x02]), g.seq, 0x03))
    reply = g._read_reply(timeout)
    if reply is None or len(reply) < 10 or reply[0:2] != b"\x0e\x02" or reply[2] != 0x00:
        return None
    import struct
    yaw, roll, pitch = struct.unpack_from("<hhh", reply, 4)
    return yaw / 10, roll / 10, pitch / 10


def wrap_step(new, old):
    """Real motion between two yaw reads (removes the +/-180 jump)."""
    d = new - old
    if d > 180:
        d -= 360
    elif d < -180:
        d += 360
    return d


class Spinner:
    def __init__(self, g):
        self.g = g
        self.turned = 0.0       # unwrapped, in reading degrees
        self.last = None        # last yaw read
        self.last_read_t = 0.0
        self.last_send_t = None
        self.max_gap = 0.0
        self.wraps = 0
        self.fails = 0

    def send(self, dps):
        now = time.time()
        if self.last_send_t is not None:
            self.max_gap = max(self.max_gap, now - self.last_send_t)
        self.g.speed(yaw=dps)
        self.last_send_t = time.time()

    def update(self):
        """Read if due; returns True when a new reading came in."""
        if time.time() - self.last_read_t < READ_EVERY_S:
            return False
        self.last_read_t = time.time()
        a = read_angles(self.g)
        if a is None:
            self.fails += 1
            return False
        if self.last is not None:
            d = wrap_step(a[0], self.last)
            if abs(a[0] - self.last) > 180:
                self.wraps += 1
            self.turned += d
        self.last = a[0]
        self.angles = a
        return True

    def run_for(self, dps, seconds):
        end = time.time() + seconds
        while time.time() < end:
            self.send(dps)
            self.update()
            time.sleep(RESEND_S)


def spin(g, turn, speed, tol, gain, min_speed):
    s = Spinner(g)
    a = None
    for _ in range(3):
        a = read_angles(g, timeout=1.0)
        if a is not None:
            break
    if a is None:
        raise RuntimeError("can't read the gimbal angles (no reply)")
    s.last, s.angles = a[0], a
    start = a
    print("start          yaw %7.1f  roll %6.1f  pitch %6.1f" % a)
    if abs(a[2]) > 80:
        print("               (lens along the yaw axis: the spin turns the camera around its lens;"
              " in normal mode it sways back - use --mode 2 or 3)")

    # 1. Probe: which way does the reading go for a positive speed?
    probe = min(speed, 20.0) * (1 if turn >= 0 else -1)
    s.run_for(probe, PROBE_S)
    s.last_read_t = 0.0
    s.update()
    moved = s.turned
    if abs(moved) < 2.0:
        raise RuntimeError("the gimbal did not move during the probe (%.1f deg for %.0f deg/s): "
                           "speed control not taken over?" % (moved, probe))
    sign = 1.0 if (moved > 0) == (probe > 0) else -1.0
    print("probe          %+.0f deg/s moved the reading %+.1f deg -> positive speed makes the reading go %s"
          % (probe, moved, "UP" if sign > 0 else "DOWN"))

    # 2. Closed loop on the unwrapped angle.
    t0 = time.time()
    limit_s = abs(turn) / max(min_speed, 1.0) + 10.0
    best = abs(turn - s.turned)
    best_t = time.time()
    next_print = 0.0
    print("   t(s)   yaw read   turned   to go   cmd deg/s")
    while True:
        err = turn - s.turned
        if abs(err) <= tol:
            break
        want = max(min_speed, min(speed, gain * abs(err))) * (1 if err > 0 else -1)
        cmd = want * sign
        s.send(cmd)
        if s.update():
            e = abs(turn - s.turned)
            if e < best - 0.5:
                best, best_t = e, time.time()
            elif e > best + PULLBACK_DEG:
                raise RuntimeError("pulled back %.1f deg against the command at %+.1f deg turned: the gimbal is "
                                   "undoing the spin (try --mode 2 or 3, or check Lock mode / yaw limits)"
                                   % (e - best, s.turned))
            if time.time() - best_t > STALL_S:
                raise RuntimeError("no progress for %.1f s at %+.1f deg turned (limit, mode or follow?)"
                                   % (STALL_S, s.turned))
            if time.time() - t0 >= next_print:
                next_print += 0.5
                print("  %6.2f   %8.1f  %+7.1f  %+6.1f   %+6.1f" % (time.time() - t0, s.last, s.turned, err, cmd))
        if time.time() - t0 > limit_s:
            raise RuntimeError("took longer than %.0f s; stopped at %+.1f deg" % (limit_s, s.turned))
        time.sleep(RESEND_S)
    elapsed = time.time() - t0
    g.speed(0, 0, 0)
    time.sleep(0.6)
    s.last_read_t = 0.0
    s.update()
    end = s.angles
    print("end            yaw %7.1f  roll %6.1f  pitch %6.1f" % end)
    print("-" * 64)
    print("turned         %+.1f deg  (asked %+.1f, error %+.1f)" % (s.turned, turn, s.turned - turn))
    print("time           %.1f s, %d wraps of the reading, %d failed reads, longest command gap %.0f ms"
          % (elapsed, s.wraps, s.fails, s.max_gap * 1000))
    print("yaw reading    %.1f -> %.1f" % (start[0], end[0]))
    return s


def main():
    ap = argparse.ArgumentParser(description="RS4 closed-loop yaw spin (default one full turn)")
    ap.add_argument("--turn", type=float, default=360.0, help="degrees to turn, in the yaw reading (default 360; negative = other way)")
    ap.add_argument("--speed", type=float, default=30.0, help="top speed in deg/s (default 30)")
    ap.add_argument("--min-speed", type=float, default=3.0, help="slowest speed near the end (default 3)")
    ap.add_argument("--gain", type=float, default=1.5, help="deg/s per degree still to go, near the end (default 1.5)")
    ap.add_argument("--tol", type=float, default=1.0, help="stop within this many degrees (default 1)")
    ap.add_argument("--mode", type=int, choices=(0, 2, 3), default=None,
                    help="set before the spin: 0 normal, 2 ROLL 360, 3 3D ROLL 360 (normal again at the end)")
    ap.add_argument("--keep-mode", action="store_true", help="leave --mode set at the end")
    ap.add_argument("--pitch", type=float, default=None, help="move to this pitch (roll 0) first, e.g. 90 for lens up")
    args = ap.parse_args()

    print("RS4 spin %+.0f deg: interface %s, channel %s, bitrate %s"
          % (args.turn, BUS_KWARGS["interface"], BUS_KWARGS["channel"], BUS_KWARGS["bitrate"]))
    with can.Bus(**BUS_KWARGS) as bus:
        g = DjiGimbal(bus)
        changed_mode = False
        try:
            if args.mode is not None:
                g.set_roll360(args.mode)
                changed_mode = args.mode != 0
                time.sleep(0.5)
            if args.pitch is not None:
                ok = g.move_abs(roll=0, pitch=args.pitch, duration=3.0)
                print("pitch %.1f: %s" % (args.pitch, "ok" if ok else "no reply / refused"))
            spin(g, args.turn, args.speed, args.tol, args.gain, args.min_speed)
            print("DONE")
        except KeyboardInterrupt:
            print("\nstopped (Ctrl+C)")
        except RuntimeError as e:
            print("\nSTOPPED: %s" % e)
        finally:
            g.speed(0, 0, 0)
            g.speed(take_over=False)
            if changed_mode and not args.keep_mode:
                g.set_roll360(0)


if __name__ == "__main__":
    main()
