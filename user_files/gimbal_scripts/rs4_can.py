#!/usr/bin/env python3
"""DJI RS4 over CAN: the reference script (rs4_can_demo.py), used by the sandbox.

Everything down to "Sandbox driver" is the reference code. Changes:
  - a check that `import can` found python-can (see _check_python_can);
  - move_abs(..., wait=True): the reference sleeps until the move ends;
    the sandbox passes wait=False so its gimbal thread isn't blocked;
  - _read_reply resyncs on a false 0xAA / bad CRC instead of crashing
    (struct.error) or giving up, and returns only the reply to the command
    just sent (late replies are skipped).

Joints: the SDK's joint yaw / roll / pitch are the motors J1 / J2 / J3, J1
closest to the handle. The sandbox shows them as J1-J3 (data keys stay
yaw / roll / pitch). get_angles(joint=False) gives the camera's WORLD
(attitude) yaw / roll / pitch instead.

The "Sandbox driver" section at the end is what the sandbox calls (from one
thread only): connect(cfg), get_angles(), move_to(yaw, roll, pitch, duration),
close(), and optionally get_world() (camera attitude, for Calibrate) and
track_begin() / track_speed(dps) / track_end() (J1 along the track during a Run). Running this file directly runs the reference demo.

Needs, in the Python that runs the sandbox:  pip install python-can pyusb libusb-package
"""
import struct
import sys
import time
import can
import functools, usb.core, libusb_package


def _check_python_can():
    """`import can` can pick up something that isn't python-can: a folder or
    file named `can` on the path while python-can is not installed in this
    Python. Then can.Bus is missing ("module 'can' has no attribute 'Bus'")."""
    if hasattr(can, "Bus"):
        return
    where = getattr(can, "__file__", None) or list(getattr(can, "__path__", []) or [])
    raise ImportError(
        "'import can' did not load python-can: it found %r. Install python-can in "
        "this Python (%s -m pip install python-can pyusb libusb-package) and "
        "remove or rename that 'can' folder/file." % (where, sys.executable))


_check_python_can()

usb.core.find = functools.partial(usb.core.find,
                                  backend=libusb_package.get_libusb1_backend())

BUS_KWARGS = dict(interface="canalystii", channel=0, bitrate=1_000_000)

TX_ID = 0x223   # PC -> gimbal
RX_ID = 0x222   # gimbal -> PC


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


# ---------------- gimbal class ------------------------------------------------
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
    def move_abs(self, yaw=None, roll=None, pitch=None, duration=2.0, wait=True):
        """
        Absolute move. Angles in degrees; pass None to leave an axis untouched.
        Limits: yaw +-180, roll +-30, pitch -56..+146.
        duration: seconds for the move (0.1 s resolution, max 25.5 s).
        wait: sleep until the move is done (reference behaviour); the sandbox passes False.
        """
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
        if wait:
            time.sleep(duration + 0.3)              # wait for the move to finish
        return ok

    # --- CmdSet 0x0E, CmdID 0x02: read angles (SDK 2.3.4.3) ---
    def get_angles(self, joint=True):
        reply = self.send_command(0x0E, 0x02, bytes([0x02 if joint else 0x01]))
        if reply is None or len(reply) < 10 or reply[2] != 0x00:
            return None
        yaw, roll, pitch = struct.unpack_from("<hhh", reply, 4)
        return yaw / 10, roll / 10, pitch / 10

    def set_roll360(self, mode=2):          # 0 = normal, 2 = ROLL 360, 3 = 3D ROLL 360
        reply = self.send_command(0x0E, 0x0C, bytes([0x22, 0x01, mode << 3]))
        print("set mode reply:", reply.hex(" ") if reply else None)

    def speed(self, yaw=0, roll=0, pitch=0, take_over=True):
        ctrl = 0x80 if take_over else 0x00   # bit 7 = take over speed control
        payload = struct.pack("<hhhB", int(yaw * 10), int(roll * 10),
                              int(pitch * 10), ctrl | 0x08)
        self.send_command(0x0E, 0x01, payload, want_reply=False)

    def roll_turns(self, deg_per_s=45, seconds=8):
        end = time.time() + seconds
        while time.time() < end:
            self.speed(roll=deg_per_s)
            time.sleep(0.1)                  # must be resent more often than every 0.5 s
        self.speed(0, 0, 0)                  # stop
        self.speed(take_over=False)          # release speed control


# ---------------- Sandbox driver ------------------------------------------------
# The sandbox's gimbal thread calls these four. cfg holds the Gimbal object's
# Connection settings (interface, channel, bitrate); missing ones use BUS_KWARGS.
_bus = None
_gimbal = None


def connect(cfg=None):
    global _bus, _gimbal
    close()
    cfg = dict(cfg or {})
    kwargs = dict(BUS_KWARGS)
    for k in ("interface", "channel", "bitrate"):
        if cfg.get(k) not in (None, ""):
            kwargs[k] = cfg[k]
    kwargs["channel"] = int(kwargs["channel"])
    kwargs["bitrate"] = int(kwargs["bitrate"])
    _bus = can.Bus(**kwargs)
    _gimbal = DjiGimbal(_bus)


def get_angles():
    """(J1, J2, J3) joint angles in degrees (SDK joint yaw, roll, pitch); raises if the read fails."""
    if _gimbal is None:
        raise RuntimeError("not connected")
    a = _gimbal.get_angles()
    if a is None:
        raise RuntimeError("no reply from gimbal (or error reply)")
    return a


def get_world():
    """(yaw, roll, pitch) of the camera in the WORLD (attitude), in degrees;
    pitch and roll are against gravity. Optional: the sandbox's Calibrate
    uses it. Raises if the read fails."""
    if _gimbal is None:
        raise RuntimeError("not connected")
    a = _gimbal.get_angles(joint=False)
    if a is None:
        raise RuntimeError("no reply from gimbal (or error reply)")
    return a


def move_to(yaw=None, roll=None, pitch=None, duration=2.0):
    """Absolute move; returns at once (True if the gimbal accepted it)."""
    if _gimbal is None:
        raise RuntimeError("not connected")
    return _gimbal.move_abs(yaw=yaw, roll=roll, pitch=pitch, duration=duration, wait=False)


# Run drive (optional): during a Run the sandbox turns J1 along the track with
# speed commands, for any number of turns, and sends J2 / J3 as absolute moves
# with yaw left out. In 3D ROLL 360 mode the SDK's yaw speed turns only J1 (RS4
# tests, docs/examples/gimbal/rs4_spin_360.py); the sandbox resends the speed
# every tick (it must come more often than every 0.5 s) and works out its sign.
RUN_MODE = 3  # 3D ROLL 360


def track_begin():
    """Start of a run: 3D ROLL 360 mode for the whole run."""
    if _gimbal is None:
        raise RuntimeError("not connected")
    _gimbal.set_roll360(RUN_MODE)


def track_speed(dps):
    """J1 speed in deg/s (taken over from the joystick); no reply is awaited."""
    if _gimbal is None:
        raise RuntimeError("not connected")
    _gimbal.speed(yaw=float(dps))


def track_end():
    """End of a run (or pause): stop, hand speed control back, normal mode."""
    if _gimbal is None:
        return
    _gimbal.speed(0, 0, 0)
    _gimbal.speed(take_over=False)
    _gimbal.set_roll360(0)


def close():
    global _bus, _gimbal
    bus, _bus, _gimbal = _bus, None, None
    if bus is not None:
        try:
            bus.shutdown()
        except Exception:
            pass


# ---------------- demo (reference) ----------------------------------------------
if __name__ == "__main__":
    with can.Bus(**BUS_KWARGS) as bus:
        g = DjiGimbal(bus)
        # g.set_roll360(0)                          # back to normal 3-axis mode first

        # 1. Point the lens along the yaw axis. SDK pitch range is -56..+146 deg.
        #    Try +90 (lens up). If the lens points the wrong way, check the sign.
        # print("pitch move ok:", g.move_abs(pitch=90, duration=3.0))
        print("angles:", g.get_angles())

        # 2. Spin yaw continuously with speed commands, resent every 50 ms.
        t0 = time.time()
        while time.time() - t0 < 12:             # 12 s x 30 deg/s = 360 deg
            g.speed(yaw=30)
            if int((time.time() - t0) * 20) % 10 == 0:
                print("yaw:", (g.get_angles() or [None])[0])
            time.sleep(0.05)
        g.speed(0, 0, 0); g.speed(take_over=False)

        g.move_abs(pitch=0, duration=3.0)        # return to level
