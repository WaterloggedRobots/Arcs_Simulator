
import math
import time
import threading
import numpy as np
import pyaubo_sdk

robot_ip ="192.168.2.124"#"192.168.0.101""192.168.1.102"
robot_port = 30004

class RobotArcs:
    def __init__(self, ip=robot_ip, port=robot_port, user="aubo", pwd="123456", block=False):
        self.ip = ip
        self.port = port
        self.user = user
        self.pwd = pwd
        self.client = None
        self.name = None
        self.interface = None
        self.mc = None
        self._dead = True
        self._connecting = False
        self.initPos = [-1.4687596998706127, 0.6089765683721227, 1.9923472065558696, -0.1984839510809548, 1.57996713558883, 0.1695107834780384]
        self.loadPosJ = [0, -15*math.pi/180, 100*math.pi/180, 25*math.pi/180, 90*math.pi/180, 0]
        if block:
            while not self.connect():
                print("arcscontrol: RPC connect failed, retrying...")
                time.sleep(2)
        else:
            # Do not touch the controller on the viewer thread. Offline UI
            # must come up even when 192.168.x.x is unreachable; the
            # runtime updater calls reconnect() later.
            print("arcscontrol: deferred RPC connect", self.ip)

    def _drop_client(self):
        """Abandon a dead RpcClient. Do not logout/disconnect — that AVs after RPC drop."""
        self.client = None
        self.mc = None
        self.interface = None
        self.name = None
        self._dead = True

    def hasConnected(self):
        if self._dead or self.client is None or self.mc is None:
            return False
        try:
            return bool(self.client.hasConnected())
        except Exception:
            self._drop_client()
            return False

    def _connect_once(self):
        self._drop_client()
        self.client = pyaubo_sdk.RpcClient()
        try:
            self.client.setRequestTimeout(1500)
        except Exception:
            pass
        self.client.connect(self.ip, self.port)
        if not self.client.hasConnected():
            print("arcscontrol: RPC connect failed")
            self._drop_client()
            return False
        self.client.login(self.user, self.pwd)
        names = self.client.getRobotNames()
        if not names:
            print("arcscontrol: no robots on controller")
            self._drop_client()
            return False
        self.name = names[0]
        self.interface = self.client.getRobotInterface(self.name)
        self.mc = self.interface.getMotionControl()
        cfg = self.interface.getRobotConfig()
        self.mc.setSpeedFraction(0.75)
        cfg.setTcpOffset([0, 0, 0, 0, 0, 0])
        self._dead = False
        print("arcscontrol: RPC connected", self.ip)
        return True

    def connect(self, force=False, timeout_s=8.0):
        """One shot connect+login. Returns True if ready for motion."""
        if self.client is not None and not self._dead:
            try:
                if self.client.hasConnected() and self.mc is not None:
                    return True
            except Exception:
                if not force:
                    self._drop_client()
        if self._connecting:
            return bool(self.hasConnected())
        box = {"ok": False, "err": None}
        self._connecting = True

        def _run():
            try:
                box["ok"] = bool(self._connect_once())
            except Exception as e:
                box["err"] = e
                try:
                    self._drop_client()
                except Exception:
                    pass
            self._connecting = False

        t = threading.Thread(target=_run, daemon=True)
        t.start()
        t.join(max(1.0, float(timeout_s)))
        if t.is_alive():
            print("arcscontrol: RPC connect still running", self.ip, "wait", timeout_s)
            return False
        self._connecting = False
        if box["err"] is not None:
            print("arcscontrol:", box["err"])
            return False
        return bool(box["ok"])

    def set_address(self, ip=None, port=None):
        """Point at another controller (Robot Settings in the UI). The next
        reconnect() from the runtime updater uses the new address."""
        ip = str(ip or "").strip() or self.ip
        port = int(port) if port not in (None, "") else self.port
        if ip == self.ip and port == self.port:
            return False
        print("arcscontrol: robot address", self.ip, "->", ip, port)
        self.ip, self.port = ip, port
        self._drop_client()
        return True

    def reconnect(self):
        if self.hasConnected():
            return True
        if self._connecting:
            return False
        return self.connect(force=True, timeout_s=2.0)

    def wait_arrival(self, start_timeout_s=2.0):
        t0 = time.time()
        while self.mc.getExecId() == -1:
            if time.time() - t0 > start_timeout_s:
                return -1
            time.sleep(0.05)
        while self.mc.getExecId() != -1:
            time.sleep(0.05)
        return 0

    def MoveLAwait(self, pose_xyzrpy, vel=1.2, acc=0.25, blend_r=0.0, blend_type=0):
        self.mc.moveLine(pose_xyzrpy, float(vel), float(acc), float(blend_r), int(blend_type))
        return self.wait_arrival()

    def MoveLAsync(self, pose_xyzrpy, vel=1.2, acc=0.25, blend_r=0.0, blend_type=0.02):
        print("moving")
        self.mc.moveLine(pose_xyzrpy, float(vel), float(acc), float(blend_r), int(blend_type))
        return None

    def MoveJAwait(self, q, jvel=80*math.pi/180, jacc=60*math.pi/180, blend_r=0.0, blend_type=0):    #input as rad
        self.mc.moveJoint(q, float(jvel), float(jacc), float(blend_r), int(blend_type))
        return self.wait_arrival()

    def MoveJAsync(self, q, jvel=0.4, jacc=0.2, blend_r=0.0, blend_type=0):
        self.mc.moveJoint(
            [float(v) for v in q[:6]],
            float(jvel), float(jacc), float(blend_r), int(blend_type),
        )
        return None

    def GetTcpPose(self):  # base->TCP
        if not self.hasConnected():
            return None
        try:
            return self.interface.getRobotState().getTcpPose()
        except Exception:
            self._drop_client()
            return None

    def GetToolPose(self):  # base->Tool
        if not self.hasConnected():
            return None
        try:
            return self.interface.getRobotState().getToolPose()
        except Exception:
            self._drop_client()
            return None

    def GetJoints(self):    #joints as rad
        if not self.hasConnected():
            return None
        try:
            return self.interface.getRobotState().getJointPositions()
        except Exception:
            self._drop_client()
            return None

    def StopMove(self):
        """Hard stop: stopMove is only a pause. Also stopJoint + clearPath."""
        try:
            self.mc.stopMove(True, True)
        except Exception:
            pass
        try:
            self.mc.stopJoint(8.0)
        except Exception:
            pass
        try:
            self.mc.stopLine(4.0, 8.0)
        except Exception:
            pass
        try:
            self.mc.clearPath()
        except Exception:
            pass

    def PauseMove(self):
        try:
            self.mc.stopMove(True, True)
        except Exception:
            pass

    def AbortPathBuffer(self, name="rec"):
        """Stop executing movePathBuffer and drop the named buffer."""
        name = str(name or "rec")
        self.StopMove()
        try:
            self.mc.pathBufferFree(name)
        except Exception:
            pass
        try:
            t0 = time.time()
            while self.mc.getExecId() != -1 and time.time() - t0 < 0.25:
                try:
                    self.mc.stopJoint(8.0)
                except Exception:
                    pass
                time.sleep(0.02)
        except Exception:
            pass

    def StartMove(self):
        self.mc.clearPath()
        self.mc.startMove()

    # Speed jogging (ARCS MotionControl speedLine / speedJoint). The arm keeps
    # the commanded speed until a new one arrives, t runs out, or StopSpeed.
    # Call again every control tick with a short t, so a lost caller stops it.
    def SpeedLine(self, xd, acc=0.5, t=0.2):
        """TCP velocity [vx, vy, vz, wx, wy, wz] (m/s, rad/s) in the base frame."""
        return self.mc.speedLine([float(v) for v in xd[:6]], float(acc), float(t))

    def SpeedJoint(self, qd, acc=1.0, t=0.2):
        """Joint velocities (rad/s), acc in rad/s^2."""
        return self.mc.speedJoint([float(v) for v in qd[:6]], float(acc), float(t))

    def StopSpeed(self, acc=1.0, rot_acc=2.0, joint_acc=3.0):
        """End a speed jog: decelerate (line and joint), no clearPath."""
        try:
            self.mc.stopLine(float(acc), float(rot_acc))
        except Exception:
            pass
        try:
            self.mc.stopJoint(float(joint_acc))
        except Exception:
            pass

    def Close(self):
        c = self.client
        self._drop_client()
        if c is None:
            return
        try:
            c.logout()
        except Exception:
            pass
        try:
            c.disconnect()
        except Exception:
            pass

    def GetFK(self, q):
        pose, err = self.interface.getRobotAlgorithm(q) # rad, J1..J6
        if not err:             # err  = 0 on success
            return pose         # pose = [x, y, z, rx, ry, rz]   meters + rad (RPY)
        else:
            return -1

    def GetIK(self, targetPose, seed=None):
        if not self.hasConnected():
            return None
        pose = [float(v) for v in targetPose]
        if seed is None:
            seed = self.GetJoints()
        if seed is None:
            return None
        q_near = [float(v) for v in np.asarray(seed, dtype=float).reshape(-1)[:6]]
        try:
            q, code = self.interface.getRobotAlgorithm().inverseKinematics(q_near, pose)
        except Exception:
            self._drop_client()
            return None
        print("IK code", code, "q", q)
        if int(code) != 0:
            return None
        q = [float(v) for v in q]
        # Reject only an echo of the input pose (all six values), not a real
        # solution whose first three radians happen to match x, y, z metres.
        if len(q) != 6 or (len(pose) >= 6 and np.allclose(q, pose[:6], atol=1e-6)):
            return None
        return q

    def SetSpeed(self,new_speed):
        self.mc.setSpeedFraction(new_speed)

    # pathBuffer types from AUBO pathBufferAlloc:
    # 1 toppra, 2 cubic_spline (recorded traj), 3 joint B-spline
    PATH_BUF_TOPPRA = 1
    PATH_BUF_CUBIC_SPLINE = 2
    PATH_BUF_JOINT_BSPLINE = 3

    def _as_joint_list(self, points):
        out = []
        for p in points or []:
            if isinstance(p, dict):
                q = p.get("q") or p.get("joints") or p.get("pose")
            else:
                q = p
            if q is None:
                continue
            q = [float(v) for v in list(q)[:6]]
            if len(q) == 6:
                out.append(q)
        return out

    def _unwrap_traj(self, traj):
        out = []
        prev = None
        for q in traj:
            q = [float(v) for v in q[:6]]
            if prev is not None:
                for j in range(6):
                    d = q[j] - prev[j]
                    while d > math.pi:
                        q[j] -= 2.0 * math.pi
                        d = q[j] - prev[j]
                    while d < -math.pi:
                        q[j] += 2.0 * math.pi
                        d = q[j] - prev[j]
            out.append(q)
            prev = q
        return out

    def _resample_traj(self, traj, times=None, dt=0.01, src_interval=None, max_n=4000):
        traj = self._unwrap_traj(traj)
        if times is None or len(times) != len(traj):
            step = src_interval if src_interval and float(src_interval) > 1e-6 else dt
            times = [i * float(step) for i in range(len(traj))]
        times = [float(t) for t in times]
        t_end = max(times[-1] - times[0], dt)
        n = max(2, int(round(t_end / dt)) + 1)
        if n > max_n:
            dt = t_end / float(max_n - 1)
            n = max_n
        t0 = times[0]
        out = []
        j = 0
        for i in range(n):
            t = t0 + min(i * dt, t_end)
            while j + 1 < len(times) and times[j + 1] < t:
                j += 1
            if j + 1 >= len(times):
                out.append(list(traj[-1]))
                continue
            span = times[j + 1] - times[j]
            a, b = traj[j], traj[j + 1]
            if max(abs(a[k] - b[k]) for k in range(6)) < 1e-9:
                out.append(list(a))
                continue
            u = 0.0 if span < 1e-9 else (t - times[j]) / span
            u = max(0.0, min(1.0, u))
            out.append([(1.0 - u) * a[k] + u * b[k] for k in range(6)])
        return out, dt

    def prepare_path_buffer(self, points, name="rec", interval=0.01,
                            path_type=None, chunk=10, times=None):
        """Load joint waypoints into a controller pathBuffer. Does NOT execute.

        Matches Aubo demo: cubic_spline + ~10 ms samples.
        pathBufferEval -5 = invalid args; any non-zero return fails the upload.
        """
        raw = self._unwrap_traj(self._as_joint_list(points))
        if len(raw) < 2:
            raise ValueError("pathBuffer needs at least 2 joint points")
        # Keep same-q samples that span real time (dwells). Collapsing them
        # to one knot makes _resample_traj lerp the previous move across the
        # hold, which is the type-2 crawl at pose B.
        if times is not None and len(times) == len(raw):
            kept_q, kept_t = [raw[0]], [float(times[0])]
            for q, t in zip(raw[1:], times[1:]):
                t = float(t)
                prev = kept_q[-1]
                dt = t - kept_t[-1]
                same = max(abs(q[k] - prev[k]) for k in range(6)) < 1e-4
                if same and dt < 1e-3:
                    kept_t[-1] = t
                    continue
                kept_q.append(q)
                kept_t.append(t)
            raw, times = kept_q, kept_t
        if len(raw) < 2:
            raise ValueError("pathBuffer needs at least 2 joint points")
        dt = 0.01
        src_dt = float(interval) if interval and float(interval) > 1e-6 else dt
        traj, dt = self._resample_traj(
            raw, times=times, dt=dt, src_interval=src_dt, max_n=4000
        )
        play_s = (len(traj) - 1) * dt
        expect_s = None
        if times and len(times) == len(raw):
            expect_s = float(times[-1]) - float(times[0])
        elif src_dt:
            expect_s = src_dt * (len(raw) - 1)
        print(
            "pathBuffer resample",
            "in", len(raw),
            "src_dt", src_dt,
            "out", len(traj),
            "dt", dt,
            "play_s", play_s,
            "expect_s", expect_s,
        )
        if expect_s and expect_s > 1.0 and play_s < 0.25 * expect_s:
            raise ValueError(
                "refusing pathBuffer: play time %.2fs would be far shorter than %.2fs"
                % (play_s, expect_s)
            )
        # Always cubic spline (type 2). Type 1 TOPPRA / type 3 B-spline unused.
        path_type = self.PATH_BUF_CUBIC_SPLINE
        name = str(name or "rec")
        n = len(traj)
        chunk = max(int(chunk or 10), 40)
        prev_to = 1500
        try:
            self.client.setRequestTimeout(120000)
        except Exception:
            pass
        vmax = [3.892, 3.892, 3.892, 4.136, 4.136, 4.136]
        amax = [8.65, 8.65, 8.65, 10.37, 10.37, 10.37]
        wait_s = max(20.0, min(120.0, 10.0 + n * 0.008))
        try:
            try:
                self.mc.pathBufferFree(name)
            except Exception:
                pass
            self.mc.pathBufferAlloc(name, path_type, n)
            i = 0
            while i < n:
                chunk_pts = [list(map(float, q)) for q in traj[i:i + chunk]]
                self.mc.pathBufferAppend(name, chunk_pts)
                i += chunk
            # Firmware rejects empty limit arrays (-5 = invalid args), so pass
            # the joint vmax/amax directly. Any non-zero return is a failed
            # upload; never treat a rejected eval as success.
            ev = self.mc.pathBufferEval(name, vmax, amax, dt)
            print("pathBufferEval", name, "type", path_type, "n", n, "dt", dt, "ret", ev)
            if ev not in (0, None):
                raise RuntimeError("pathBufferEval rejected %s (ret=%s)" % (name, ev))
            t0 = time.time()
            while not self.mc.pathBufferValid(name):
                if time.time() - t0 > wait_s:
                    raise TimeoutError(
                        "pathBufferEval timed out for %s (n=%s type=%s ret=%s wait=%.1fs)"
                        % (name, n, path_type, ev, wait_s)
                    )
                time.sleep(0.05)
            return {"ok": True, "name": name, "count": n, "interval": dt, "type": path_type}
        finally:
            try:
                self.client.setRequestTimeout(prev_to)
            except Exception:
                pass

    def run_path_buffer(self, name="rec", wait=True):
        """Execute a prepared pathBuffer. Call only when ready to move."""
        try:
            self.mc.startMove()
        except Exception:
            pass
        try:
            self.SetSpeed(1.0)
        except Exception:
            pass
        self.mc.movePathBuffer(name)
        if not wait:
            return 0
        while self.mc.getExecId() == -1:
            time.sleep(0.05)
        while self.mc.getExecId() != -1:
            time.sleep(0.05)
        return 0

    def plan_joint_path(self, points, interval=0.01, name="rec", times=None):
        """Validate + upload buffer only. Never starts motion."""
        return self.prepare_path_buffer(
            points, name=name, interval=interval, times=times
        )

    def estimate_move_duration(self, current_q, target_q, speed_fraction=0.3, max_joint_speed=100):
        """
        Estimate how long a joint move will take on the real robot.

        Parameters
        ----------
        current_q : list or np.array
            Current joint angles (rad)
        target_q : list or np.array
            Target joint angles (rad)
        speed_fraction : float
            Speed percentage used on the real robot (0.0 ~ 1.0)
        max_joint_speed : float
            Maximum joint speed of the robot (rad/s)

        Returns
        -------
        duration : float
            Estimated time in seconds
        """
        current_q = np.array(current_q)
        target_q = np.array(target_q)

        # Largest joint angle change
        max_delta = np.max(np.abs(target_q - current_q))

        # Effective speed

        effective_speed = max_joint_speed * speed_fraction

        # Avoid division by zero
        if effective_speed < 1e-6:
            return 10.0

        duration = max_delta / effective_speed

        # Add a small margin (real robot has acceleration/deceleration)
        duration *= 1.15

        return max(duration, 0.3)  # minimum 0.3

def main():
    try:

        print(robot.mc.getExecId())

        robot.MoveJAwait(robot.initPos)
        print("robot at init pos")

        pos1 = robot.GetToolPose()
        print(pos1)
        pos1[1]-=0.1
        robot.MoveLAwait(pos1)
        print("robot at pos 1: ",robot.GetToolPose())

        pos2 = robot.GetToolPose()
        pos2[0]-=0.15
        robot.MoveLAsync(pos2)
        print("robot moving to ", pos2)
        t0 = time.time()
        """ """
        while robot.mc.getExecId() == -1 and (time.time() - t0) < 2.0:  #wait for robot to start moving
            time.sleep(0.1)

        robot.StopMove()    #interrupt
        print("interrupt")
        time.sleep(0.5)

        robot.StartMove()
        print("robot moving to init pos")
        robot.MoveJAwait(robot.initPos)
        """ """

    except KeyboardInterrupt:
        try:
            robot.Close()
        except Exception:
            pass

if __name__ == "__main__":
    robot = RobotArcs(robot_ip, robot_port, block=True)
    print("RPC客户端连接成功!" if robot.hasConnected() else "RPC客户端连接失败!")
    try:
        main()
    finally:
        try:
            robot.Close()
        except Exception:
            pass
