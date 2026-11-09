"""DJI RS4 gimbal over CAN (DJI R SDK v2.5), the driver script for a Gimbal object.

The sandbox loads this file for every Gimbal object whose Script field is empty
(or points here) and calls the functions below from its gimbal thread. They are
never called from two threads at once.

    connect(cfg)                       open the CAN bus; raise on failure
    get_angles() -> (yaw, roll, pitch) joint angles in degrees; raise on failure
    move_to(yaw, roll, pitch, duration) absolute move, returns at once; True if
                                       the gimbal accepted it
    close()                            release the bus

`cfg` holds the Gimbal object's settings: interface (python-can interface name,
"canalystii" for the CANalyst-II USB adapter), channel and bitrate.

Based on rs4_can_demo.py and the SDK's "DJI R SDK Protocol and User Interface"
(section 2.3: frame layout, CRC parameters, CmdSet 0x0E commands).

Needs: pip install python-can pyusb libusb-package
"""
import struct
import time

TX_ID = 0x223  # PC -> gimbal
RX_ID = 0x222  # gimbal -> PC

# Joint limits from SDK 2.3.4.1 (degrees).
LIMITS = {"yaw": (-180.0, 180.0), "roll": (-30.0, 30.0), "pitch": (-56.0, 146.0)}
REPLY_TIMEOUT_S = 0.25


# ---------------- CRC (SDK section 3.2) ---------------------------------------
# The SDK gives init 0xC55C / 0xC55C0000 in non-reflected form; in this
# reflected (LSB-first) implementation that is 0x3AA3. Checked against the
# SDK's sample packet (CRC16 = 0x42A2, CRC32 = 0xBE97407B).
def _crc(data, width, poly_reflected, init):
    crc = init
    for byte in data:
        crc ^= byte
        for _ in range(8):
            crc = (crc >> 1) ^ poly_reflected if crc & 1 else crc >> 1
    return crc & ((1 << width) - 1)


def crc16(data):
    return _crc(data, 16, 0xA001, 0x3AA3)


def crc32(data):
    return _crc(data, 32, 0xEDB88320, 0x3AA3)


def build_packet(cmd_set, cmd_id, payload, seq, cmd_type=0x03):
    """cmd_type 0x03 = command frame, reply required; 0x00 = no reply."""
    data = bytes([cmd_set, cmd_id]) + payload
    total_len = 12 + len(data) + 4
    header = (
        bytes([0xAA])
        + struct.pack("<H", total_len)      # version 0 | 10-bit length, LSB first
        + bytes([cmd_type, 0x00])           # CmdType, ENC (unencrypted)
        + b"\x00\x00\x00"                   # reserved
        + struct.pack("<H", seq & 0xFFFF)   # sequence number
    )
    pkt = header + struct.pack("<H", crc16(header)) + data
    return pkt + struct.pack("<I", crc32(pkt))


# ---------------- bus ---------------------------------------------------------
_bus = None
_seq = 0x1122
_rx = bytearray()


def _patch_libusb():
    """CANalyst-II goes through pyusb; on Windows it needs libusb-package's DLL."""
    try:
        import functools
        import usb.core
        import libusb_package
        if not getattr(usb.core.find, "_rs4_patched", False):
            find = functools.partial(usb.core.find, backend=libusb_package.get_libusb1_backend())
            find._rs4_patched = True
            usb.core.find = find
    except Exception:
        pass


def connect(cfg=None):
    global _bus
    cfg = dict(cfg or {})
    close()
    import can  # python-can
    interface = str(cfg.get("interface") or "canalystii")
    if interface == "canalystii":
        _patch_libusb()
    _bus = can.Bus(interface=interface, channel=int(cfg.get("channel") or 0),
                   bitrate=int(cfg.get("bitrate") or 1_000_000))
    _rx.clear()


def close():
    global _bus
    bus, _bus = _bus, None
    if bus is not None:
        try:
            bus.shutdown()
        except Exception:
            pass


def _send(pkt):
    import can
    for i in range(0, len(pkt), 8):  # 8-byte CAN frames
        _bus.send(can.Message(arbitration_id=TX_ID, data=pkt[i:i + 8], is_extended_id=False))


def _next_packet(deadline):
    """Next whole, CRC-checked packet from the gimbal, or None at the deadline."""
    while True:
        start = _rx.find(b"\xAA")
        if start < 0:
            _rx.clear()
        elif start:
            del _rx[:start]
        if len(_rx) >= 3:
            length = struct.unpack_from("<H", _rx, 1)[0] & 0x03FF
            if length < 16:
                del _rx[:1]  # not a header: look for the next 0xAA
                continue
            if len(_rx) >= length:
                pkt = bytes(_rx[:length])
                if crc32(pkt[:-4]) != struct.unpack("<I", pkt[-4:])[0]:
                    del _rx[:1]
                    continue
                del _rx[:length]
                return pkt
        left = deadline - time.time()
        if left <= 0:
            return None
        msg = _bus.recv(timeout=min(0.05, left))
        if msg is not None and msg.arbitration_id == RX_ID:
            _rx.extend(msg.data)


def send_command(cmd_set, cmd_id, payload, want_reply=True, timeout=REPLY_TIMEOUT_S):
    """Send one command. With want_reply, return the reply's DATA segment
    (starting with CmdSet, CmdID) or None on timeout."""
    global _seq
    if _bus is None:
        raise RuntimeError("not connected")
    _seq = (_seq + 1) & 0xFFFF
    _send(build_packet(cmd_set, cmd_id, payload, _seq, 0x03 if want_reply else 0x00))
    if not want_reply:
        return None
    deadline = time.time() + timeout
    while True:
        pkt = _next_packet(deadline)
        if pkt is None:
            return None
        data = pkt[12:-4]
        # Skip pushes and late replies to other commands.
        if len(data) >= 2 and data[0] == cmd_set and data[1] == cmd_id:
            return data


# ---------------- commands (CmdSet 0x0E) --------------------------------------
def get_angles():
    """Joint angles (yaw, roll, pitch) in degrees (SDK 2.3.4.3)."""
    reply = send_command(0x0E, 0x02, bytes([0x02]))
    if reply is None:
        raise RuntimeError("no reply from gimbal")
    if len(reply) < 10:
        raise RuntimeError("short reply")
    if reply[2] != 0x00:
        raise RuntimeError("gimbal return code 0x%02X" % reply[2])
    if reply[3] == 0x00:
        raise RuntimeError("angle data not ready")
    yaw, roll, pitch = struct.unpack_from("<hhh", reply, 4)
    return yaw / 10.0, roll / 10.0, pitch / 10.0


def _clamp(axis, v):
    lo, hi = LIMITS[axis]
    return max(lo, min(hi, float(v)))


def move_to(yaw=None, roll=None, pitch=None, duration=2.0):
    """Absolute move (SDK 2.3.4.1); None leaves an axis where it is.
    duration: seconds for the move, 0.1 s steps, 0.1 to 25.5 s. Does not wait."""
    ctrl = 0x01  # bit0: absolute control
    if yaw is None:
        ctrl |= 1 << 1
    if roll is None:
        ctrl |= 1 << 2
    if pitch is None:
        ctrl |= 1 << 3

    def raw(axis, a):
        return 0 if a is None else int(round(_clamp(axis, a) * 10))

    t = max(1, min(255, int(round(float(duration) * 10))))
    payload = struct.pack("<hhhBB", raw("yaw", yaw), raw("roll", roll), raw("pitch", pitch), ctrl, t)
    reply = send_command(0x0E, 0x00, payload)
    return reply is not None and len(reply) >= 3 and reply[2] == 0x00


def stop():
    """Stop all three axes (zero speed, then hand speed control back)."""
    for ctrl in (0x80 | 0x08, 0x08):
        send_command(0x0E, 0x01, struct.pack("<hhhB", 0, 0, 0, ctrl), want_reply=False)


if __name__ == "__main__":
    # Stand-alone check: read the angles, nudge yaw and come back.
    connect({"interface": "canalystii", "channel": 0, "bitrate": 1_000_000})
    try:
        print("angles (yaw, roll, pitch):", get_angles())
        print("yaw +20:", move_to(yaw=20, duration=1.5))
        time.sleep(1.8)
        print("angles:", get_angles())
        print("yaw 0:", move_to(yaw=0, duration=1.5))
        time.sleep(1.8)
        print("angles:", get_angles())
    finally:
        close()
