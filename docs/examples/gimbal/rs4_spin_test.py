#!/usr/bin/env python3
"""RS4 360° yaw spin test (reference code), with diagnostic prints.

Spins yaw with speed commands (CmdSet 0x0E, CmdID 0x01), resent every 50 ms,
for speed x seconds degrees (30 °/s x 12 s = 360° by default), then stops,
releases speed control and levels the pitch. It prints what the sandbox needs
to know before it can drive multi-turn yaw:

  1. WRAP: does the read yaw jump from about +180 to about -180 (or back)?
  2. DIRECTION: does a positive speed turn clockwise or anticlockwise seen
     from above? (Software can't see this: watch the gimbal; you are asked at
     the end.)
  3. RESEND: is every speed command sent within the SDK's 0.5 s window?
  4. ACCURACY: how far the gimbal really turned (unwrapped) versus speed x time.

Run from a terminal on the PC with the CANalyst-II adapter:

    python docs/examples/gimbal/rs4_spin_test.py
    python docs/examples/gimbal/rs4_spin_test.py --reverse      # then the other way
    python docs/examples/gimbal/rs4_spin_test.py --speed 20 --seconds 18
    python docs/examples/gimbal/rs4_spin_test.py --roll360 2    # ROLL 360 mode for the spin
    python docs/examples/gimbal/rs4_spin_test.py --roll360 3    # 3D ROLL 360 mode
    python docs/examples/gimbal/rs4_spin_test.py --pitch 0      # level the camera first

With the lens along the yaw axis (pitch near +/-90) a yaw spin turns the
camera around its lens. In normal mode the stabiliser keeps the image level
and pulls that back after ~50 deg, so the spin sways. ROLL 360 / 3D ROLL 360
mode (--roll360 2 / 3, as the reference's set_roll360) lets it turn all the
way. The test sets normal mode again at the end unless --keep-mode.

Needs: pip install python-can pyusb libusb-package
"""
import argparse
import struct
import time
import can
import functools, usb.core, libusb_package


usb.core.find = functools.partial(usb.core.find,
                                  backend=libusb_package.get_libusb1_backend())

BUS_KWARGS = dict(interface="canalystii", channel=0, bitrate=1_000_000)

TX_ID = 0x223   # PC -> gimbal
RX_ID = 0x222   # gimbal -> PC

RESEND_S = 0.05        # speed command period (reference: 50 ms)
READ_EVERY_S = 0.25    # how often the spin loop reads the angles
READ_TIMEOUT_S = 0.3   # reply wait for reads inside the spin loop
SDK_WINDOW_S = 0.5     # a speed command lasts 0.5 s (SDK 2.3.4.2)


# ---------------- CRC (parameters from SDK section 3.2) -----------------------
# SDK lists init 0xC55C / 0xC55C0000 in non-reflected form; in a reflected
# (LSB-first) implementation that equals 0x3AA3. Verified against the SDK's
# sample packet (CRC16 = 0x42A2, CRC32 = 0xBE97407B).
def _crc(data: bytes, width: int, poly_reflected: int, init: int) -> int:
    crc = init
    for byte in data:
        crc ^= byte
        for _ in range(8):
            crc = (crc >> 1) ^ poly_reflected if crc & 1 else crc >> 1
    return crc & ((1 << width) - 1)


def crc16(data: bytes) -> int:
    return _crc(data, 16, 0xA001, 0x3AA3)


def crc32(data: bytes) -> int:
    return _crc(data, 32, 0xEDB88320, 0x3AA3)


# ---------------- packet building ---------------------------------------------
def build_packet(cmd_set: int, cmd_id: int, payload: bytes, seq: int,
                 cmd_type: int = 0x03) -> bytes:
    """cmd_type 0x03 = command frame, reply required; 0x00 = no reply."""
    data = bytes([cmd_set, cmd_id]) + payload
    total_len = 12 + len(data) + 4
    header = (
        bytes([0xAA])
        + struct.pack("<H", total_len)          # version 0 | 10-bit length, LSB first
        + bytes([cmd_type, 0x00])               # CmdType, ENC (unencrypted)
        + b"\x00\x00\x00"                       # reserved
        + struct.pack("<H", seq & 0xFFFF)       # sequence number
    )
    pkt = header + struct.pack("<H", crc16(header)) + data
    return pkt + struct.pack("<I", crc32(pkt))


# ---------------- gimbal class (reference) --------------------------------------
class DjiGimbal:
    def __init__(self, bus: can.BusABC):
        self.bus = bus
        self.seq = 0x1122

    def _send(self, pkt: bytes):
        for i in range(0, len(pkt), 8):                      # split into 8-byte CAN frames
            self.bus.send(can.Message(arbitration_id=TX_ID,
                                      data=pkt[i:i + 8],
                                      is_extended_id=False))

    def _read_reply(self, timeout: float = 1.0, want=None):
        """DATA segment of the next valid packet (of `want` = (cmd_set, cmd_id) if given).

        Changed from the reference: an 0xAA inside data, a late reply or a
        packet cut in half used to end in "bad CRC" (None) or a struct.error
        when the length read was too short. Now a start is only trusted when
        its length is possible and its header CRC16 is right; otherwise the
        search goes on to the next 0xAA. Replies to other commands are
        skipped, until timeout.
        """
        buf = bytearray()
        end = time.time() + timeout
        while time.time() < end:
            msg = self.bus.recv(timeout=0.05)
            if msg is None or msg.arbitration_id != RX_ID:
                continue
            buf += msg.data
            while True:
                start = buf.find(b"\xAA")
                if start < 0:
                    buf.clear()
                    break
                del buf[:start]
                if len(buf) < 3:
                    break
                length = struct.unpack_from("<H", buf, 1)[0] & 0x03FF
                if length < 18:                              # header 12 + CmdSet/CmdID 2 + CRC32 4
                    del buf[:1]                              # not a packet start: look further
                    continue
                if len(buf) < 12:
                    break
                if crc16(bytes(buf[:10])) != struct.unpack_from("<H", buf, 10)[0]:
                    del buf[:1]                              # header CRC16 wrong: not a packet start
                    continue
                if len(buf) < length:
                    break                                    # wait for the rest
                pkt = bytes(buf[:length])
                if crc32(pkt[:-4]) != struct.unpack("<I", pkt[-4:])[0]:
                    del buf[:1]                              # bad CRC: not a packet start
                    continue
                del buf[:length]
                data = pkt[12:-4]                            # DATA segment
                if want is None or (len(data) >= 2 and data[0] == want[0] and data[1] == want[1]):
                    return data
        return None

    def send_command(self, cmd_set, cmd_id, payload, want_reply=True, timeout=1.0):
        self.seq = (self.seq + 1) & 0xFFFF
        self._send(build_packet(cmd_set, cmd_id, payload, self.seq,
                                0x03 if want_reply else 0x00))
        return self._read_reply(timeout, want=(cmd_set, cmd_id)) if want_reply else None

    # --- CmdSet 0x0E, CmdID 0x00: position control (SDK 2.3.4.1) ---
    def move_abs(self, yaw=None, roll=None, pitch=None, duration=2.0):
        """Absolute move (degrees; None leaves an axis). Waits for the move."""
        ctrl = 0x01                                 # bit0 = 1 -> absolute control
        if yaw is None:
            ctrl |= 1 << 1                          # yaw axis invalid (ignored)
        if roll is None:
            ctrl |= 1 << 2
        if pitch is None:
            ctrl |= 1 << 3
        to_raw = lambda a: int(round((a or 0) * 10))   # 0.1 degree units
        payload = struct.pack("<hhhBB",
                              to_raw(yaw), to_raw(roll), to_raw(pitch),
                              ctrl, min(255, int(duration * 10)))
        reply = self.send_command(0x0E, 0x00, payload)
        # reply DATA = [CmdSet, CmdID, return_code]; 0x00 means success
        ok = reply is not None and len(reply) >= 3 and reply[2] == 0x00
        time.sleep(duration + 0.3)                  # wait for the move to finish
        return ok

    # --- CmdSet 0x0E, CmdID 0x02: read angles (SDK 2.3.4.3) ---
    def get_angles(self, joint=True, timeout=1.0):
        reply = self.send_command(0x0E, 0x02, bytes([0x02 if joint else 0x01]), timeout=timeout)
        if reply is None or len(reply) < 10 or reply[2] != 0x00:
            return None
        yaw, roll, pitch = struct.unpack_from("<hhh", reply, 4)
        return yaw / 10, roll / 10, pitch / 10

    # --- CmdSet 0x0E, CmdID 0x0C / 0x0B: operating mode (reference set_roll360) ---
    def set_roll360(self, mode=2):          # 0 = normal, 2 = ROLL 360, 3 = 3D ROLL 360
        reply = self.send_command(0x0E, 0x0C, bytes([0x22, 0x01, mode << 3]))
        print("set mode %d reply:" % mode, reply.hex(" ") if reply else None)
        return reply

    def read_mode(self):
        """Raw reply to the reference's mode read-back (0x0E 0x0B, 0x22)."""
        reply = self.send_command(0x0E, 0x0B, bytes([0x22]))
        print("mode read-back:", reply.hex(" ") if reply else None)
        return reply

    def speed(self, yaw=0, roll=0, pitch=0, take_over=True):
        ctrl = 0x80 if take_over else 0x00   # bit 7 = take over speed control
        payload = struct.pack("<hhhB", int(yaw * 10), int(roll * 10),
                              int(pitch * 10), ctrl | 0x08)
        self.send_command(0x0E, 0x01, payload, want_reply=False)


# ---------------- test ----------------------------------------------------------
def fmt(a):
    return "None" if a is None else "yaw %7.1f  roll %6.1f  pitch %6.1f" % a


def read_settled(g, label, tries=3):
    for _ in range(tries):
        a = g.get_angles()
        if a is not None:
            print("%-14s %s" % (label, fmt(a)))
            return a
    print("%-14s READ FAILED (no reply %d times)" % (label, tries))
    return None


def spin(g, speed_dps, seconds):
    """The reference spin, plus wrap / unwrap / resend bookkeeping."""
    print()
    print("=" * 72)
    print("SPIN  yaw %+.1f deg/s for %.1f s  -> expected %+.1f deg" % (speed_dps, seconds, speed_dps * seconds))
    print("Watch the gimbal FROM ABOVE and note which way it turns.")
    print("=" * 72)
    start = read_settled(g, "before:")
    prev = start[0] if start else None
    unwrapped = 0.0
    wraps = []
    reads = fails = 0
    sends = 0
    max_gap = 0.0
    late = 0
    last_send = None
    next_read = 0.0
    print("   t(s)    yaw read   step   unwrapped total   roll   pitch   note")
    t0 = time.time()
    while time.time() - t0 < seconds:
        now = time.time()
        if last_send is not None:
            gap = now - last_send
            max_gap = max(max_gap, gap)
            if gap > SDK_WINDOW_S:
                late += 1
                print("  !! %.2f s since the last speed command (> %.1f s): the gimbal may have stopped" % (gap, SDK_WINDOW_S))
        g.speed(yaw=speed_dps)
        last_send = time.time()
        sends += 1
        if now - t0 >= next_read:
            next_read += READ_EVERY_S
            r0 = time.time()
            a = g.get_angles(timeout=READ_TIMEOUT_S)
            read_ms = (time.time() - r0) * 1000
            reads += 1
            if a is None:
                fails += 1
                print("  %6.2f   READ FAILED (%.0f ms)" % (now - t0, read_ms))
            else:
                yaw = a[0]
                note = ""
                step = 0.0
                if prev is not None:
                    step = yaw - prev
                    # A jump of more than 180 deg between two reads is a wrap, not real motion.
                    if step > 180:
                        step -= 360
                        wraps.append((now - t0, prev, yaw))
                        note = "<-- WRAP (%+.1f -> %+.1f)" % (prev, yaw)
                    elif step < -180:
                        step += 360
                        wraps.append((now - t0, prev, yaw))
                        note = "<-- WRAP (%+.1f -> %+.1f)" % (prev, yaw)
                    unwrapped += step
                prev = yaw
                print("  %6.2f   %8.1f  %+6.1f   %+10.1f     %6.1f  %6.1f   %s%s" % (
                    now - t0, yaw, step, unwrapped, a[1], a[2], note,
                    ("  (read %.0f ms)" % read_ms) if read_ms > 100 else ""))
        time.sleep(RESEND_S)
    g.speed(0, 0, 0)
    g.speed(take_over=False)
    elapsed = time.time() - t0
    time.sleep(1.0)  # let it settle
    end = read_settled(g, "after (1 s):")
    if end is not None and prev is not None:
        step = end[0] - prev
        if step > 180:
            step -= 360
        elif step < -180:
            step += 360
        unwrapped += step

    expected = speed_dps * elapsed
    print("-" * 72)
    print("speed commands sent : %d  (every %.0f ms planned, longest gap %.0f ms)" % (sends, RESEND_S * 1000, max_gap * 1000))
    print("gaps over 0.5 s     : %d  %s" % (late, "OK" if late == 0 else "<-- the spin was interrupted; reads are too slow"))
    print("angle reads         : %d  (%d failed)" % (reads, fails))
    print("wraps seen          : %d  %s" % (len(wraps), "-> yaw reading WRAPS at +/-180" if wraps else "-> no wrap seen (see the note below)"))
    for t, a, b in wraps:
        print("                      at %.2f s: %+.1f -> %+.1f" % (t, a, b))
    print("turned (unwrapped)  : %+.1f deg  (reading increased: %s)" % (unwrapped, "yes" if unwrapped > 0 else "no"))
    print("expected            : %+.1f deg  (speed x %.2f s)" % (expected, elapsed))
    print("error               : %+.1f deg" % (unwrapped - expected))
    if not wraps and abs(unwrapped) > 300:
        print("note                : turned past a full circle without a jump: yaw may NOT wrap (it counts on).")
    return {"speed": speed_dps, "turned": unwrapped, "wraps": len(wraps), "late": late, "fails": fails}


def ask_direction(res):
    try:
        ans = input("\nSeen FROM ABOVE, did it turn (c)lockwise or (a)nticlockwise? [c/a/enter=skip] ").strip().lower()
    except EOFError:
        ans = ""
    if ans[:1] in ("c", "a"):
        res["seen"] = "clockwise" if ans[0] == "c" else "anticlockwise"
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="RS4 360 deg yaw spin test")
    ap.add_argument("--speed", type=float, default=30.0, help="yaw speed in deg/s (default 30)")
    ap.add_argument("--seconds", type=float, default=12.0, help="spin time in s (default 12 -> 360 deg at 30 deg/s)")
    ap.add_argument("--reverse", action="store_true", help="after the spin, spin back with the opposite speed")
    ap.add_argument("--no-level", action="store_true", help="don't level the pitch at the end")
    ap.add_argument("--roll360", type=int, choices=(0, 2, 3), default=None,
                    help="set this mode before spinning: 0 normal, 2 ROLL 360, 3 3D ROLL 360")
    ap.add_argument("--keep-mode", action="store_true", help="don't set normal mode (0) again at the end")
    ap.add_argument("--pitch", type=float, default=None,
                    help="move to this pitch (and roll 0) before spinning, e.g. 0 for a level camera")
    args = ap.parse_args()

    print("RS4 spin test: interface %(interface)s, channel %(channel)s, bitrate %(bitrate)s" % BUS_KWARGS)
    results = []
    with can.Bus(**BUS_KWARGS) as bus:
        g = DjiGimbal(bus)
        start = read_settled(g, "start:")
        g.read_mode()
        if start is not None and abs(start[2]) > 80 and args.roll360 in (None, 0):
            print("NOTE: pitch is %.1f deg: the lens points along the yaw axis, so the spin turns the camera "
                  "around its lens. In normal mode it sways back; try --roll360 2 or 3." % start[2])
        try:
            if args.roll360 is not None:
                g.set_roll360(args.roll360)
                time.sleep(0.5)
                g.read_mode()
            if args.pitch is not None:
                ok = g.move_abs(roll=0, pitch=args.pitch, duration=3.0)
                print("moved to pitch %.1f, roll 0: %s" % (args.pitch, "ok" if ok else "no reply / refused"))
                read_settled(g, "after pitch:")
            results.append(ask_direction(spin(g, args.speed, args.seconds)))
            if args.reverse:
                time.sleep(1.0)
                results.append(ask_direction(spin(g, -args.speed, args.seconds)))
        finally:
            # Also on Ctrl+C: stop, hand speed control back, normal mode again.
            g.speed(0, 0, 0)
            g.speed(take_over=False)
            if args.roll360 not in (None, 0) and not args.keep_mode:
                g.set_roll360(0)
                g.read_mode()
        if not args.no_level:
            print("\nlevelling pitch:", "ok" if g.move_abs(pitch=0, duration=3.0) else "no reply / refused")
        read_settled(g, "end:")

    print("\n" + "=" * 72)
    print("SUMMARY (send this back)")
    for r in results:
        print("speed %+6.1f deg/s: turned %+7.1f deg, wraps %d, late resends %d, failed reads %d, seen: %s" % (
            r["speed"], r["turned"], r["wraps"], r["late"], r["fails"], r.get("seen", "not given")))
    print("start pitch %s, --pitch %s, --roll360 %s" % ("%.1f" % start[2] if start else "?", args.pitch, args.roll360))
    print("=" * 72)
