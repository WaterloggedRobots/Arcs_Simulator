"""The Look At Live run's sender as its own process (started by live_aim.ProcStreamer).

Python runs one thread at a time, so inside the app the sender waited for its
turn after every network call while the aimer, the 3D view, the camera script
and the web server computed: ~14 points/s instead of 50. Here it has a Python
of its own and its own RPC connection to the controller (a second TCP session
to the same address, as a second client), and does nothing but send.

Started as `python stream_proc.py <port> <authkey hex>`; talks to the app over
multiprocessing.connection on 127.0.0.1:
  app -> here:  ("start", {samples, tune}), ("aim", s, delta, dev_deg, err), ("tune", {...}),
                ("hb",), ("stop",)
  here -> app:  ("ready", ok, error), ("sent", s, q), ("log", text),
                ("done", stats, error, log)
Before "start" it connects (ip, port, user, pwd from argv) and says "ready".
It stops the arm itself if the app goes quiet for APP_LOST_S.
"""
from __future__ import annotations

import os
import sys
import threading
import time

APP_LOST_S = 1.0


def main():
    port = int(sys.argv[1])
    key = bytes.fromhex(sys.argv[2])
    ip, rport, user, pwd = sys.argv[3], int(sys.argv[4]), sys.argv[5], sys.argv[6]
    here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, here)
    from multiprocessing.connection import Client
    conn = Client(("127.0.0.1", port), authkey=key)
    send_lock = threading.Lock()

    def send(*msg):
        try:
            with send_lock:
                conn.send(msg)
        except Exception:
            pass

    import builtins
    _print = builtins.print

    def print_to_app(*a, **k):
        send("log", " ".join(str(x) for x in a))
    builtins.print = print_to_app

    try:
        import numpy as np
        import arcscontrol
        import live_aim
        bot = arcscontrol.RobotArcs(ip=ip, port=rport, user=user, pwd=pwd)
        ok = bool(bot.connect(force=True, timeout_s=8.0))
        send("ready", ok, "" if ok else "the controller did not accept a second connection")
    except Exception as e:
        send("ready", False, "stream process failed to start: %s" % e)
        return
    if not ok:
        return

    # Wait for the run.
    msg = None
    while msg is None:
        if not conn.poll(30.0):
            return  # no run came: give up
        m = conn.recv()
        if m and m[0] == "start":
            msg = m
        elif m and m[0] == "stop":
            return
    cfg = msg[1]
    live_aim.set_tuning(cfg.get("tune") or {})
    st = live_aim.Streamer(bot, cfg["samples"], external=True)
    st.on_sent = lambda s, q: send("sent", float(s), [float(v) for v in q])
    last_msg = [time.perf_counter()]
    lost = [False]

    def receive():
        while not st.stop_ev.is_set():
            try:
                if conn.poll(0.05):
                    m = conn.recv()
                    last_msg[0] = time.perf_counter()
                    kind = m[0]
                    if kind == "aim":
                        st.feed_answer(m[1], np.asarray(m[2], dtype=float), *m[3:5])
                    elif kind == "tune":
                        live_aim.set_tuning(m[1])
                    elif kind == "stop":
                        st.stop_ev.set()
                elif time.perf_counter() - last_msg[0] > APP_LOST_S:
                    lost[0] = True
                    st.error = "lost the app for %.1f s: stopped" % APP_LOST_S
                    st.stop_ev.set()
            except (EOFError, OSError):
                lost[0] = True
                st.error = "the app closed the connection: stopped"
                st.stop_ev.set()

    rx = threading.Thread(target=receive, name="stream_rx", daemon=True)
    rx.start()
    st.start()
    st.thread.join()
    if lost[0]:
        # Nobody else will stop the arm.
        try:
            bot.mc.stopJoint(3.0)
        except Exception:
            pass
    send("done", dict(st.stats), st.error, list(st.log))
    builtins.print = _print
    time.sleep(0.2)


if __name__ == "__main__":
    try:
        main()
    finally:
        os._exit(0)  # don't log the RPC session out (arcscontrol: that can crash after a drop)
