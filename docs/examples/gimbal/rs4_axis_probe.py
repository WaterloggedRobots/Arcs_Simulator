#!/usr/bin/env python3
"""DJI RS4: which command turns ONE joint only? (small, safe test moves)

The 360 roll test showed that SPEED commands aim the camera in world terms
(attitude): the gimbal works out the motor moves itself, so in the
"handle pointing at the target" pose all three joints moved. The camera roll
around the lens needs only the handle-axis motor (joint yaw) to turn.

This script tries each candidate command with a small move (default 20 deg),
prints how much every joint and attitude angle changed, says whether only
one joint moved, then undoes the move and shows what is left over:

  speed   yaw / roll            0x0E/0x01 for --secs seconds
  incr    yaw / roll            0x0E/0x00 incremental position (ctrl bit0 = 0)
  abs     yaw                   0x0E/0x00 absolute position (only with --with-abs:
                                in 3D ROLL 360 it flipped the camera upside down)

each in normal mode and in 3D ROLL 360 mode, optionally with the gimbal set
to Lock mode first (--lock: 0x0E/0x0E, no follow). Set the gimbal up as in
the sandbox (handle pointing at a target, camera looking along it) and run
it with the sandbox closed:

    python docs/examples/gimbal/rs4_axis_probe.py
    python docs/examples/gimbal/rs4_axis_probe.py --lock
    python docs/examples/gimbal/rs4_axis_probe.py --only incr-yaw-3 --step 30

Send back the summary table at the end.
"""
import argparse
import os
import struct
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.normpath(os.path.join(HERE, "..", "..", "..", "user_files", "gimbal_scripts")))
import rs4_can  # noqa: E402
from rs4_can import DjiGimbal, BUS_KWARGS, build_packet  # noqa: E402,F401

can = rs4_can.can
NAMES = ("yaw", "roll", "pitch")
MODES = {0: "normal", 3: "3D ROLL 360"}
MOVED = 4.0     # deg: a joint changed this much -> it moved
STILL = 2.0     # deg: the others stayed within this -> "only one joint"


def read_set(g, which, timeout=0.3):
    g.seq = (g.seq + 1) & 0xFFFF
    g._send(build_packet(0x0E, 0x02, bytes([0x02 if which == "joint" else 0x01]), g.seq, 0x03))
    reply = g._read_reply(timeout, want=(0x0E, 0x02))
    if reply is None or len(reply) < 10 or reply[2] != 0x00:
        return None
    y, r, p = struct.unpack_from("<hhh", reply, 4)
    return y / 10, r / 10, p / 10


def read_both(g, tries=3):
    for _ in range(tries):
        j, a = read_set(g, "joint"), read_set(g, "attitude")
        if j and a:
            return j, a
    raise RuntimeError("can't read the gimbal angles")


def wrap(d):
    while d > 180:
        d -= 360
    while d < -180:
        d += 360
    return d


def diff(a, b):
    return tuple(wrap(b[i] - a[i]) for i in range(3))


def speed_for(g, axis, dps, secs):
    end = time.time() + secs
    while time.time() < end:
        g.speed(**{axis: dps})
        time.sleep(0.05)
    g.speed(0, 0, 0)
    g.speed(take_over=False)


def position(g, axis, deg, secs, absolute):
    """0x0E/0x00 with only `axis` valid. absolute=False: incremental."""
    ctrl = 0x01 if absolute else 0x00
    vals = {"yaw": 0, "roll": 0, "pitch": 0}
    vals[axis] = int(round(deg * 10))
    for i, n in enumerate(NAMES):
        if n != axis:
            ctrl |= 1 << (i + 1)  # bit1 yaw, bit2 roll, bit3 pitch invalid
    payload = struct.pack("<hhhBB", vals["yaw"], vals["roll"], vals["pitch"], ctrl, max(1, min(255, int(secs * 10))))
    reply = g.send_command(0x0E, 0x00, payload)
    ok = reply is not None and len(reply) >= 3 and reply[2] == 0x00
    time.sleep(secs + 0.4)
    return ok


def set_mode(g, mode):
    g.set_roll360(mode)
    time.sleep(0.6)


def set_lock(g):
    reply = g.send_command(0x0E, 0x0E, bytes([0x00, 0x00]))
    print("lock mode reply:", reply.hex(" ") if reply else None)
    time.sleep(0.5)


def tests(step, secs, with_abs=False):
    dps = step / secs
    out = []
    for mode in (0, 3):
        for axis in ("yaw", "roll"):
            out.append(("speed-%s-%d" % (axis, mode), mode,
                        lambda g, s, a=axis: speed_for(g, a, s * dps, secs)))
        for axis in ("yaw", "roll"):
            out.append(("incr-%s-%d" % (axis, mode), mode,
                        lambda g, s, a=axis: position(g, a, s * step, secs, absolute=False)))
        if with_abs:
            out.append(("abs-yaw-%d" % mode, mode, None))  # needs the start angle: handled below
    return out


def run_one(g, name, mode, fn, step, secs):
    print("\n--- %s  (mode %s)" % (name, MODES[mode]))
    set_mode(g, mode)
    j0, a0 = read_both(g)
    if name.startswith("abs-yaw"):
        target = j0[0] + step
        if target > 180:
            target -= 360
        fwd = lambda: position(g, "yaw", target, secs, absolute=True)
        back = lambda: position(g, "yaw", j0[0], secs, absolute=True)
    else:
        fwd = lambda: fn(g, +1)
        back = lambda: fn(g, -1)
    fwd()
    time.sleep(0.3)
    j1, a1 = read_both(g)
    dj, da = diff(j0, j1), diff(a0, a1)
    moved = [n for i, n in enumerate(NAMES) if abs(dj[i]) >= MOVED]
    only = len(moved) == 1 and all(abs(dj[i]) <= STILL for i, n in enumerate(NAMES) if n != moved[0])
    verdict = ("ONLY joint %s" % moved[0]) if only else ("joints moved: %s" % (", ".join(moved) or "none"))
    print("   joint    d yaw %+6.1f  roll %+6.1f  pitch %+6.1f" % dj)
    print("   attitude d yaw %+6.1f  roll %+6.1f  pitch %+6.1f" % da)
    print("   -> %s" % verdict)
    back()
    time.sleep(0.3)
    j2, _ = read_both(g)
    left = diff(j0, j2)
    print("   after undo, joints off start by yaw %+.1f roll %+.1f pitch %+.1f" % left)
    return {"name": name, "dj": dj, "da": da, "verdict": verdict, "left": left}


def main():
    ap = argparse.ArgumentParser(description="RS4: which command moves one joint only")
    ap.add_argument("--step", type=float, default=20.0, help="test move size in degrees (default 20)")
    ap.add_argument("--secs", type=float, default=1.0, help="time per test move (default 1 s)")
    ap.add_argument("--lock", action="store_true", help="set Lock mode (no follow) before the tests")
    ap.add_argument("--only", default=None, help="run one test by name, e.g. incr-yaw-3")
    ap.add_argument("--with-abs", action="store_true",
                    help="also absolute yaw (in 3D ROLL 360 it flipped the RS4 camera upside down)")
    args = ap.parse_args()

    print("RS4 axis probe: step %.0f deg in %.1f s, interface %s" % (args.step, args.secs, BUS_KWARGS["interface"]))
    results = []
    with can.Bus(**BUS_KWARGS) as bus:
        g = DjiGimbal(bus)
        try:
            j, a = read_both(g)
            print("start joint     yaw %7.1f roll %7.1f pitch %7.1f" % j)
            print("start attitude  yaw %7.1f roll %7.1f pitch %7.1f" % a)
            if args.lock:
                set_lock(g)
            for name, mode, fn in tests(args.step, args.secs, args.with_abs or bool(args.only and args.only.startswith("abs"))):
                if args.only and name != args.only:
                    continue
                try:
                    results.append(run_one(g, name, mode, fn, args.step, args.secs))
                except RuntimeError as e:
                    print("   FAILED: %s" % e)
                    results.append({"name": name, "verdict": "failed: %s" % e})
                time.sleep(0.5)
        except KeyboardInterrupt:
            print("\nstopped (Ctrl+C)")
        finally:
            g.speed(0, 0, 0)
            g.speed(take_over=False)
            g.set_roll360(0)

    print("\n" + "=" * 78)
    print("SUMMARY (send this back)%s" % ("  [Lock mode]" if args.lock else ""))
    print("%-14s %-24s %-24s %s" % ("test", "joint d y / r / p", "attitude d y / r / p", "verdict"))
    for r in results:
        if "dj" in r:
            print("%-14s %+6.1f %+6.1f %+6.1f     %+6.1f %+6.1f %+6.1f     %s" % (
                r["name"], *r["dj"], *r["da"], r["verdict"]))
        else:
            print("%-14s %s" % (r["name"], r["verdict"]))
    print("=" * 78)


if __name__ == "__main__":
    main()
