#!/usr/bin/env python3
"""DJI RS4: roll the camera a smooth, measured 360 deg (or any amount) with J1 only.

Joints are named by position, J1 closest to the handle:
    J1  handle motor  (SDK "joint yaw")
    J2  middle motor  (SDK "joint roll")
    J3  camera motor  (SDK "joint pitch")
The SDK's attitude angles are the camera's direction in the WORLD; they are
printed as "world yaw / roll / pitch" for reference only.

The move this is for (sandbox set-up): the arm points the gimbal handle at the
target, horizontally, the camera looks along the handle (J3 at about -90) and
the camera then rolls a full turn around that axis. Only J1 turns.

What the RS4 tests showed:
  * In 3D ROLL 360 mode the SDK's YAW command (speed or incremental) turns
    ONLY J1; a roll command pans the camera with all three joints; in normal
    mode the roll is held to about +/-30 deg; absolute yaw flips the camera.
  * Waiting for angle replies in the control loop left gaps of up to 0.25 s
    between speed commands, so the spin stuttered and overshot. Here angle
    requests are sent WITHOUT waiting and replies are picked up as they come
    in, so a speed command goes out every 50 ms, always.
  * Incremental steps work but stop between steps, and a step of a few
    degrees is ignored. --method incr keeps them (steps of at most --step,
    the last one at least --min-step).
  * Starting with J1 far from 0 (camera rolled) the gimbal swung back on its
    own when the mode changed. --home turns J1 to 0 first.
  * On the test gimbal a positive J1 turn looked ANTICLOCKWISE (as watched).

The speed method ramps up (--accel), cruises at --speed, and slows down so it
stops on the target angle of J1 (unwrapped across +/-180). It stops with a
message if J1 stops following. On exit (also Ctrl+C) it stops, hands speed
control back and sets normal mode again.

Examples (sandbox closed, from the repo folder):

    python docs/examples/gimbal/rs4_spin_360.py --home            # J1 to 0, then +360
    python docs/examples/gimbal/rs4_spin_360.py --turn -360
    python docs/examples/gimbal/rs4_spin_360.py --speed 45 --accel 60
    python docs/examples/gimbal/rs4_spin_360.py --method incr --step 90

Uses the reference protocol code in user_files/gimbal_scripts/rs4_can.py.
Needs: pip install python-can pyusb libusb-package
"""
import argparse
import math
import os
import struct
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.normpath(os.path.join(HERE, "..", "..", "..", "user_files", "gimbal_scripts")))
import rs4_can  # noqa: E402  (reference: DjiGimbal, BUS_KWARGS, build_packet, crc16, crc32)
from rs4_can import DjiGimbal, BUS_KWARGS, build_packet, crc16, crc32  # noqa: E402,F401

can = rs4_can.can

TICK_S = 0.05        # speed command period: always, nothing in the loop waits
READ_EVERY_S = 0.1   # J1/J2/J3 request period (attitude every 3rd request)
STALL_S = 1.5        # J1 not following for this long -> stop
MODES = {0: "normal", 2: "ROLL 360", 3: "3D ROLL 360"}


def wrap(d):
    while d > 180:
        d -= 360
    while d < -180:
        d += 360
    return d


def fj(a):
    return "   ---     ---     ---" if a is None else "%7.1f %7.1f %7.1f" % a


class Link:
    """Angle requests without waiting; replies parsed whenever frames arrive."""

    def __init__(self, g):
        self.g = g
        self.buf = bytearray()
        self.joint = None
        self.world = None
        self.joint_t = 0.0
        self.sent = 0
        self.got = 0

    def request(self, which):
        g = self.g
        g.seq = (g.seq + 1) & 0xFFFF
        g._send(build_packet(0x0E, 0x02, bytes([0x02 if which == "joint" else 0x01]), g.seq, 0x03))
        self.sent += 1

    def poll(self):
        """Drain received frames; True if a new joint reading came in."""
        new = False
        while True:
            msg = self.g.bus.recv(timeout=0)
            if msg is None:
                break
            if msg.arbitration_id == rs4_can.RX_ID:
                self.buf += msg.data
        buf = self.buf
        while True:
            start = buf.find(b"\xAA")
            if start < 0:
                buf.clear()
                break
            del buf[:start]
            if len(buf) < 12:
                break
            length = struct.unpack_from("<H", buf, 1)[0] & 0x03FF
            if length < 18 or crc16(bytes(buf[:10])) != struct.unpack_from("<H", buf, 10)[0]:
                del buf[:1]
                continue
            if len(buf) < length:
                break
            pkt = bytes(buf[:length])
            if crc32(pkt[:-4]) != struct.unpack("<I", pkt[-4:])[0]:
                del buf[:1]
                continue
            del buf[:length]
            data = pkt[12:-4]
            if len(data) >= 10 and data[0] == 0x0E and data[1] == 0x02 and data[2] == 0x00:
                a = tuple(v / 10 for v in struct.unpack_from("<hhh", data, 4))
                self.got += 1
                if data[3] == 0x02:
                    self.joint, self.joint_t, new = a, time.time(), True
                elif data[3] == 0x01:
                    self.world = a
        return new

    def wait_joint(self, timeout=1.0):
        end = time.time() + timeout
        self.request("joint")
        self.request("attitude")
        t0 = self.joint_t
        while time.time() < end:
            self.poll()
            if self.joint_t > t0:
                time.sleep(0.05)
                self.poll()
                return self.joint
            time.sleep(0.01)
        return None


def incr_j1(g, deg, secs):
    """Incremental SDK-yaw (J1 in 3D ROLL 360) move, other axes invalid."""
    payload = struct.pack("<hhhBB", int(round(deg * 10)), 0, 0, 0x00 | 0x04 | 0x08,
                          max(1, min(255, int(round(secs * 10)))))
    reply = g.send_command(0x0E, 0x00, payload)
    return reply is not None and len(reply) >= 3 and reply[2] == 0x00


def home(g, link, speed):
    j = link.wait_joint()
    if j is None:
        raise RuntimeError("can't read J1-J3 (no reply)")
    if abs(j[0]) < 1.0:
        print("home           J1 already at %.1f" % j[0])
        return
    secs = max(0.5, abs(j[0]) / max(speed, 1.0))
    print("home           J1 %.1f -> 0 in %.1f s" % (j[0], secs))
    incr_j1(g, -j[0], secs)
    time.sleep(secs + 0.5)
    j = link.wait_joint()
    print("home           J1 now %s" % ("?" if j is None else "%.1f" % j[0]))


def spin_speed(g, link, turn, vmax, accel, tol, min_speed):
    j = link.wait_joint()
    if j is None:
        raise RuntimeError("can't read J1-J3 (no reply)")
    print("                      J1      J2      J3")
    print("start          %s   world yaw/roll/pitch %s" % (fj(j), fj(link.world)))
    if abs(j[0]) > 90:
        print("               (J1 is %.0f deg from 0: the gimbal may swing back on its own; --home first)" % j[0])
    d = 1.0 if turn >= 0 else -1.0
    sign = 1.0           # J1 change per + command; corrected once it has moved
    total = 0.0          # unwrapped J1 since start
    last = j[0]
    v = 0.0              # commanded J1 speed (deg/s, along d)
    t0 = time.time()
    next_tick = t0
    next_read = t0
    reads = 0
    best, best_t = abs(turn), t0
    max_gap, last_send = 0.0, None
    next_print = 0.0
    print("   t(s)   J1 turned   to go   speed   |      J1      J2      J3")
    while True:
        now = time.time()
        if link.poll():
            total += wrap(link.joint[0] - last)
            last = link.joint[0]
            if abs(total) > 4 and sign > 0 and (total > 0) != (d > 0) and best >= abs(turn) - 1:
                sign = -1.0
                print("        (J1 moves against the command: reversing the command sign)")
            e = abs(turn - total)
            if e < best - 0.3:
                best, best_t = e, now
            if now - t0 >= next_print:
                next_print += 0.5
                print("  %6.2f  %+8.1f  %+7.1f  %+6.1f   | %s" % (now - t0, total, turn - total, v * d, fj(link.joint)))
        # J1 estimated now: last reading plus what the command added since.
        est = total + v * d * sign * max(0.0, min(0.3, now - link.joint_t))
        rem = (turn - est) * d
        if rem <= tol:
            break
        dt = TICK_S if last_send is None else min(0.2, now - last_send)
        v_stop = math.sqrt(max(0.0, 2.0 * accel * max(0.0, rem - tol)))
        v = max(min_speed, min(vmax, v + accel * dt, v_stop))
        g.speed(yaw=v * d * sign)
        if last_send is not None:
            max_gap = max(max_gap, now - last_send)
        last_send = now
        if now >= next_read:
            reads += 1
            link.request("attitude" if reads % 3 == 0 else "joint")
            next_read += READ_EVERY_S
        if now - best_t > STALL_S:
            raise RuntimeError("J1 stopped following for %.1f s at %+.1f deg turned (mode, limit or follow?)"
                               % (STALL_S, total))
        if now - t0 > abs(turn) / max(min_speed, 1.0) + 15:
            raise RuntimeError("took too long; stopped at %+.1f deg" % total)
        next_tick += TICK_S
        time.sleep(max(0.0, next_tick - time.time()))
    elapsed = time.time() - t0
    g.speed(0, 0, 0)
    time.sleep(0.5)
    end = time.time() + 0.6
    while time.time() < end:
        link.request("joint")
        time.sleep(0.1)
        if link.poll():
            total += wrap(link.joint[0] - last)
            last = link.joint[0]
    print("end            %s   world yaw/roll/pitch %s" % (fj(link.joint), fj(link.world)))
    print("-" * 72)
    print("J1 turned      %+.1f deg (asked %+.1f, error %+.1f) in %.1f s" % (total, turn, total - turn, elapsed))
    print("speed commands longest gap %.0f ms; angle replies %d of %d requests" % (max_gap * 1000, link.got, link.sent))


def spin_incr(g, link, turn, speed, step, min_step, tol):
    j = link.wait_joint()
    if j is None:
        raise RuntimeError("can't read J1-J3 (no reply)")
    print("                      J1      J2      J3")
    print("start          %s" % fj(j))
    total, last = 0.0, j[0]
    print("   step   asked   J1 turned   to go   |      J1      J2      J3")
    t0 = time.time()
    for i in range(1, 100):
        err = turn - total
        if abs(err) <= tol:
            break
        s = max(-step, min(step, err))
        if abs(s) < min_step:
            s = math.copysign(min_step, s)  # tiny steps are ignored by the gimbal
        secs = max(0.2, abs(s) / max(speed, 1.0))
        if not incr_j1(g, s, secs):
            raise RuntimeError("step %d (%+.1f) refused or no reply" % (i, s))
        time.sleep(secs + 0.3)
        j = link.wait_joint()
        if j is None:
            raise RuntimeError("no reading after step %d" % i)
        moved = wrap(j[0] - last)
        total += moved
        last = j[0]
        print("  %4d  %+6.1f  %+9.1f  %+7.1f   | %s" % (i, s, total, turn - total, fj(j)))
        if abs(moved) < 0.3 * abs(s):
            raise RuntimeError("step %d asked %+.1f but J1 moved %+.1f" % (i, s, moved))
    print("-" * 72)
    print("J1 turned      %+.1f deg (asked %+.1f, error %+.1f) in %.1f s" % (total, turn, total - turn, time.time() - t0))


def show_mode(g, label):
    reply = g.send_command(0x0E, 0x0B, bytes([0x22]))
    print("mode %-8s read-back: %s" % (label, reply.hex(" ") if reply else None))


def main():
    ap = argparse.ArgumentParser(description="RS4: smooth camera roll with J1 only (3D ROLL 360)")
    ap.add_argument("--turn", type=float, default=360.0, help="J1 degrees to turn (default 360; negative = other way)")
    ap.add_argument("--method", choices=("speed", "incr"), default="speed", help="speed (smooth, default) or incr steps")
    ap.add_argument("--speed", type=float, default=30.0, help="cruise speed deg/s (default 30)")
    ap.add_argument("--accel", type=float, default=40.0, help="speed method: ramp in deg/s^2 (default 40)")
    ap.add_argument("--min-speed", type=float, default=2.0, help="speed method: slowest speed at the end (default 2)")
    ap.add_argument("--step", type=float, default=90.0, help="incr method: largest step (default 90)")
    ap.add_argument("--min-step", type=float, default=8.0, help="incr method: smallest step (default 8)")
    ap.add_argument("--tol", type=float, default=1.0, help="stop within this many degrees (default 1)")
    ap.add_argument("--home", action="store_true", help="turn J1 to 0 first")
    ap.add_argument("--mode", type=int, choices=(0, 2, 3), default=3, help="0 normal, 2 ROLL 360, 3 3D ROLL 360 (default)")
    ap.add_argument("--keep-mode", action="store_true", help="leave the mode set at the end")
    args = ap.parse_args()

    print("RS4 J1 turn %+.0f deg (%s), mode %s: interface %s, channel %s, bitrate %s" % (
        args.turn, args.method, MODES[args.mode], BUS_KWARGS["interface"], BUS_KWARGS["channel"], BUS_KWARGS["bitrate"]))
    with can.Bus(**BUS_KWARGS) as bus:
        g = DjiGimbal(bus)
        link = Link(g)
        changed = False
        try:
            show_mode(g, "before")
            g.set_roll360(args.mode)
            changed = args.mode != 0
            time.sleep(0.8)
            show_mode(g, "set")
            if args.home:
                home(g, link, args.speed)
            if args.method == "incr":
                spin_incr(g, link, args.turn, args.speed, args.step, args.min_step, args.tol)
            else:
                spin_speed(g, link, args.turn, args.speed, args.accel, args.tol, args.min_speed)
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
