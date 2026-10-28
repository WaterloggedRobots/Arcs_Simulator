# version: 0.11.0
import time
import math
import threading
from pathlib import Path
import numpy as np
from enum import Enum, auto
from skrobot.model import RobotModel
import arcscontrol
import arms
from scipy.spatial.transform import Rotation as R
import waypoints as wp

_cursor_sim = None
_arrow_sim = None
_arrow_sim_side = None
_sim_rot0 = None
_sim_selected = False
_last_good_q = None
_sim_wxyz = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)

CURSOR_POS_OFFSET = {
    "real": np.array([0.0, 0.0, 0.05], dtype=float),
    "sim": np.array([0.0, 0.0, 0.05], dtype=float),
}
CURSOR_RPY_OFFSET = {
    "real": np.array([0.0, 0.0, 0.0], dtype=float),
    "sim": np.array([0.0, 0.0, 0.0], dtype=float),
}

viewer = None
robot_actual = None
robot_sim = None
robot_ik = None
robot = None

IK_POS_TOL = 0.03
IK_ANG_TOL = 0.45
IK_RECONNECT_POS = 0.05
IK_REJECT_POS = 0.12
CURSOR_R = 0.03
ARROW_LEN = 0.07
_GIZMO_SCALE_ON = 0.15
_cursors_ready = False
_cursor_real = None
_grid_dimmed = None
_GRID_OPACITY_FULL = 1.0
_GRID_OPACITY_DIM = 0.5
_arrow_real = None
_arrow_real_side = None
_sim_gizmo = None
_gizmo_serial = 0
_sim_ik_pending = False
_ik_lock = threading.Lock()
_track = {"on": False, "name": "", "pick": False, "valid": False, "added": False, "offset": [0.0, 0.0, 0.0], "pos_offset": [0.0, 0.0, 0.0], "pin_pos": None,
          "tool": "", "tool_pick": False}
_hold_sim = 0
_start_snap_done = False
_rpc_probe_ready = False

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
_URDF_DIRS = [
    _HERE / "aubo-i5-full-master" / "aubo_description" / "urdf",
    _ROOT / "aubo-i5-full-master" / "aubo_description" / "urdf",
]
_URDF_DIR = next((p for p in _URDF_DIRS if p.is_dir()), _URDF_DIRS[0])


def list_robot_models():
    rows = []
    seen = set()
    if _URDF_DIR.is_dir():
        for p in sorted(_URDF_DIR.glob("aubo_i*.urdf")):
            if "gripper" in p.name.lower() or " " in p.name:
                continue
            stem = p.stem
            if stem.endswith("_sim") or stem.endswith("_robot"):
                continue
            if stem in seen:
                continue
            seen.add(stem)
            rows.append({"id": stem, "name": stem.replace("_", " "), "path": str(p)})
    return rows


def resolve_urdf(model_id=None):
    model_id = str(model_id or "aubo_i5").strip() or "aubo_i5"
    if model_id.lower().endswith(".urdf"):
        model_id = Path(model_id).stem
    path = _URDF_DIR / ("%s.urdf" % model_id)
    if not path.is_file():
        path = _URDF_DIR / "aubo_i5.urdf"
        model_id = "aubo_i5"
    sim = path.with_name(path.stem + "_sim.urdf")
    return model_id, path, (sim if sim.is_file() else path)


_MODEL_ID, _URDF_P, _URDF_SIM_P = resolve_urdf("aubo_i5")
URDF_PATH = str(_URDF_P)
URDF_SIM_PATH = str(_URDF_SIM_P)
if _URDF_SIM_P == _URDF_P:
    print("%s_sim.urdf missing next to real URDF; sim uses %s" % (_MODEL_ID, _URDF_P.name))


def set_robot_model(model_id):
    global URDF_PATH, URDF_SIM_PATH, _MODEL_ID
    mid, path, sim = resolve_urdf(model_id)
    URDF_PATH = str(path)
    URDF_SIM_PATH = str(sim)
    changed = mid != _MODEL_ID
    _MODEL_ID = mid
    print("robot model", mid, URDF_PATH)
    return {"id": mid, "path": URDF_PATH, "sim_path": URDF_SIM_PATH, "changed": changed}


def current_robot_model():
    return _MODEL_ID


def _drop_viser_robot(model):
    if model is None:
        return
    if viewer is not None:
        for name in ("remove", "delete"):
            fn = getattr(viewer, name, None)
            if callable(fn):
                try:
                    fn(model)
                    return
                except Exception:
                    pass
        scene = getattr(viewer, "scene", None) or getattr(getattr(viewer, "_server", None), "scene", None)
        if scene is not None and hasattr(scene, "remove_by_name"):
            try:
                scene.remove_by_name(getattr(model, "name", "") or "")
            except Exception:
                pass
    try:
        for link in getattr(model, "link_list", []) or []:
            mesh = getattr(link, "visual_mesh", None)
            h = getattr(mesh, "visual", None)
            if h is not None and hasattr(h, "remove"):
                try:
                    h.remove()
                except Exception:
                    pass
    except Exception:
        pass


def reload_robot_models():
    """Load the URDF currently selected by set_robot_model into Viser."""
    global robot_actual, robot_sim, robot_ik, _last_good_q
    if viewer is None:
        print("reload robots: no viewer yet")
        return False
    old_a, old_s, old_k = robot_actual, robot_sim, robot_ik
    _drop_viser_robot(old_a)
    _drop_viser_robot(old_s)
    import_robots(URDF_PATH, URDF_SIM_PATH)
    if robot_connected():
        set_hide_real_robot(False)
    else:
        set_hide_real_robot(True)
    reset_sim_to_home()
    print("reloaded robots", _MODEL_ID, URDF_PATH)
    return True

SIM_OPACITY = 255*0.6  # 0.40 — Viser ignores URDF rgba A; set on mesh handles
SIM_RGB = (90, 150, 230)  # sim tint (blue)
# Cursor colors follow the arm they belong to: sim = blue, real = orange.
SIM_CURSOR_RGB = (80, 200, 255)
REAL_CURSOR_RGB = (255, 180, 80)
AUBO_I5_MAX_VEL_DEG = {
    "shoulder_joint": 223.0,
    "upperArm_joint": 223.0,
    "foreArm_joint": 223.0,
    "wrist1_joint": 267.0,
    "wrist2_joint": 267.0,
    "wrist3_joint": 267.0,
}
AUBO_I5_MAX_ACCEL_DEG = {
    "shoulder_joint": 496.0,
    "upperArm_joint": 496.0,
    "foreArm_joint": 496.0,
    "wrist1_joint": 594.0,
    "wrist2_joint": 594.0,
    "wrist3_joint": 594.0,
}


class SimMode(Enum):
    FOLLOW = auto()
    PLAY_PATH = auto()


sim_mode = SimMode.FOLLOW
sim_path = None
sim_path_duration = 2.0
return_delay = 1.5
stop_flag = threading.Event()
mode_lock = threading.Lock()
HOLD_MOVE_FRACTION = 0.6
HOLD_MOVE_SPEED = math.radians(223.0) * HOLD_MOVE_FRACTION  # rad/s, 60% of J1–J3 max
HOLD_MOVE_ACCEL = math.radians(496.0) * HOLD_MOVE_FRACTION
HOLD_LIN_MPS = 0.05
HOLD_REACH_M = 0.85


def set_hold_lin_mps(mps):
    global HOLD_LIN_MPS
    try:
        HOLD_LIN_MPS = max(0.001, float(mps))
    except Exception:
        return HOLD_LIN_MPS
    print("hold lin speed", HOLD_LIN_MPS, "m/s")
    return HOLD_LIN_MPS
_hold_move = False
_hold_target_ik = None
_hold_target_q = None
_hold_arrived = False
_hold_mode = "ik"
_t_real = None
_t_sim = None
TOOL_OFFSET = np.array([0.0, 0.0, 0.0])
WORLD_YAW_CCW = 0.0


def _apply_world_yaw(model):
    if model is None or abs(WORLD_YAW_CCW) < 1e-9:
        return
    try:
        model.rotate(WORLD_YAW_CCW, "z")
    except Exception:
        try:
            model.coords.rotate(WORLD_YAW_CCW, "z")
        except Exception as e:
            print("world yaw failed:", e)


# Aubo Arm: where the robot base sits in the world. Both arms (real, sim) and
# the hidden IK model stand on it; viewer poses are world, AUBO poses (IK,
# MoveL, the IK fields) are relative to this base.
_base_pos = np.zeros(3, dtype=float)
_base_wxyz = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)


def arm_base_pose():
    return _base_pos.copy(), _base_wxyz.copy()


def _base_rot():
    return R.from_quat(wxyz_to_xyzw(_base_wxyz))


def _apply_base(model):
    if model is None:
        return
    try:
        from skrobot.coordinates import Coordinates
        model.newcoords(Coordinates(pos=[float(v) for v in _base_pos], rot=[float(v) for v in _base_wxyz]))
    except Exception as e:
        print("arm base failed:", e)
        return
    _apply_world_yaw(model)


def world_to_base(pos, wxyz):
    inv = _base_rot().inv()
    p = inv.apply(np.asarray(pos, dtype=float).reshape(3) - _base_pos)
    q = (inv * R.from_quat(wxyz_to_xyzw(wxyz))).as_quat()
    return p, np.array([q[3], q[0], q[1], q[2]], dtype=float)


def base_to_world(pos, wxyz):
    rot = _base_rot()
    p = _base_pos + rot.apply(np.asarray(pos, dtype=float).reshape(3))
    q = (rot * R.from_quat(wxyz_to_xyzw(wxyz))).as_quat()
    return p, np.array([q[3], q[0], q[1], q[2]], dtype=float)


def set_arm_base(pos=None, wxyz=None):
    """Move the Aubo Arm. The sim cursor rides along (same joints, same pose
    relative to the base); the real cursor follows the real arm's FK."""
    global _base_pos, _base_wxyz, _sim_wxyz
    if edits_locked():
        return False
    new_p = _base_pos.copy() if pos is None else np.asarray(pos, dtype=float).reshape(3)
    new_q = _base_wxyz.copy() if wxyz is None else np.asarray(wxyz, dtype=float).reshape(4)
    n = float(np.linalg.norm(new_q)) or 1.0
    new_q = new_q / n
    if not (np.all(np.isfinite(new_p)) and np.all(np.isfinite(new_q))):
        return False
    try:
        cur_p, cur_q = get_sim_cursor_pose()
        loc_p, loc_q = world_to_base(cur_p, cur_q)
    except Exception:
        loc_p = loc_q = None
    _base_pos, _base_wxyz = new_p, new_q
    for model in (robot_actual, robot_sim, robot_ik):
        _apply_base(model)
    if loc_p is not None:
        p, q = base_to_world(loc_p, loc_q)
        _sim_wxyz = q
        pin = tuple(float(v) for v in p)
        wp._suspend_gizmo_cb = True
        try:
            if _cursor_sim is not None:
                _cursor_sim.position = pin
            if _sim_gizmo is not None:
                _sim_gizmo.position = pin
                _sim_gizmo.wxyz = tuple(float(v) for v in q)
            if _track.get("pin_pos") is not None:
                _track["pin_pos"] = np.asarray(p, dtype=float).reshape(3).copy()
            _track["_ik_key"] = None
        finally:
            # Always back to False, like every other writer: restoring a value
            # read earlier could leave it stuck True when another thread set
            # it meanwhile, and then no gizmo responds any more.
            wp._suspend_gizmo_cb = False
        _draw_sim_dir(p)
    return True


def _flat(x, n, default=None):
    if x is None:
        return None if default is None else [float(v) for v in default]
    if isinstance(x, np.ndarray) and x.dtype == object:
        parts = [np.asarray(v, dtype=np.float64).reshape(-1) for v in x.tolist()]
        x = np.concatenate(parts)
    else:
        x = np.asarray(x, dtype=np.float64).reshape(-1)
    return [float(v) for v in x[:n]]


def wxyz_to_xyzw(wxyz):
    w, x, y, z = np.asarray(wxyz, dtype=float).reshape(4)
    return np.array([x, y, z, w], dtype=float)


def wxyz_to_rpy(wxyz):
    w, x, y, z = np.asarray(wxyz, dtype=float).reshape(4)
    return R.from_quat([x, y, z, w]).as_euler("xyz", degrees=False)


def apply_tool(pos, wxyz, offset=TOOL_OFFSET):
    rot = R.from_quat(wxyz_to_xyzw(wxyz))
    return np.asarray(pos, dtype=float).reshape(3) + rot.apply(offset), wxyz


def aubo_pose_from_viewer(pos, wxyz):
    pos, wxyz = world_to_base(_flat(pos, 3), wxyz)
    p = _flat(pos, 3)
    rx, ry, rz = wxyz_to_rpy(wxyz)
    return [float(p[0]), float(p[1]), float(p[2]), float(rx), float(ry), float(rz)]


def viewer_pose_from_aubo(ik):
    ik = [float(v) for v in list(ik)[:6]]
    pos = np.array(ik[:3], dtype=float)
    quat_xyzw = R.from_euler("xyz", ik[3:6], degrees=False).as_quat()
    wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]], dtype=float)
    return base_to_world(pos, wxyz)


def viser_to_aubo_pose(pos, wxyz, as_quat=True):
    return aubo_pose_from_viewer(pos, wxyz)


def robot_connected():
    if not _rpc_probe_ready:
        return False
    try:
        if robot is None or not robot.hasConnected():
            return False
        q = robot.GetJoints()
        if q is None or len(list(q)) < 6:
            return False
        # A powered-off controller still answers RPC but reports all-zero
        # joints. Do not treat that as a live pose (the viewer would copy it).
        if all(abs(float(v)) < 1e-9 for v in list(q)[:6]):
            return False
        return True
    except Exception:
        return False


# Robot Settings (UI) override for the controller address; None keeps the
# defaults in arcscontrol.py, which a script run on its own still uses.
_robot_addr = {"ip": None, "port": None}


def set_robot_address(ip=None, port=None):
    """This arm's controller address. The default arm falls back to
    arcscontrol.robot_ip / robot_port; an added arm has no controller (sim
    only) until it gets an IP."""
    global robot
    _robot_addr["ip"] = str(ip).strip() if ip not in (None, "") else None
    _robot_addr["port"] = int(port) if port not in (None, "") else None
    if arms.is_default():
        if robot is not None and hasattr(robot, "set_address"):
            robot.set_address(_robot_addr["ip"] or arcscontrol.robot_ip, _robot_addr["port"] or arcscontrol.robot_port)
    elif _robot_addr["ip"] is None:
        if robot is not None:
            print("robot address cleared:", arms.name_of(), "is sim only")
        robot = None
    elif robot is None:
        robot = arcscontrol.RobotArcs(
            ip=_robot_addr["ip"], port=_robot_addr["port"] or arcscontrol.robot_port, block=False)
    elif hasattr(robot, "set_address"):
        robot.set_address(_robot_addr["ip"], _robot_addr["port"] or arcscontrol.robot_port)
    return robot_address()


def robot_address():
    if arms.is_default():
        default_ip = str(getattr(arcscontrol, "robot_ip", ""))
    else:
        default_ip = ""
    default_port = int(getattr(arcscontrol, "robot_port", 30004))
    ip = getattr(robot, "ip", None) or _robot_addr["ip"] or default_ip
    port = getattr(robot, "port", None) or _robot_addr["port"] or default_port
    return {"ip": str(ip or ""), "port": int(port), "default_ip": default_ip,
            "default_port": default_port, "arm": arms.current(), "arm_name": arms.name_of(),
            "set_ip": _robot_addr["ip"] or "", "set_port": _robot_addr["port"] or ""}


def init(viser_viewer, urdf_path=URDF_PATH):
    global viewer, robot
    arms.capture_defaults()
    viewer = viser_viewer
    robot = arcscontrol.RobotArcs(
        ip=_robot_addr["ip"] or arcscontrol.robot_ip,
        port=_robot_addr["port"] or arcscontrol.robot_port,
        block=False)
    import_robots(urdf_path)
    if robot_connected():
        set_hide_real_robot(False)
        reset_sim_to_real()
    else:
        print("no robot at startup — viewer only, sim at home")
        set_hide_real_robot(True)
        reset_sim_to_home()
    start_threads()
    bind_default_camera()


def setup_extra_arm(pos=None, wxyz=None, model=None, ip=None, port=None):
    """Build an added Aubo Arm, run as that arm (arms.add_arm). Its robots,
    cursors and saved points start like the default arm's at launch."""
    global _base_pos, _base_wxyz
    if model:
        set_robot_model(model)
    if pos is not None:
        _base_pos = np.asarray(pos, dtype=float).reshape(3)
    if wxyz is not None:
        q = np.asarray(wxyz, dtype=float).reshape(4)
        _base_wxyz = q / (float(np.linalg.norm(q)) or 1.0)
    set_robot_address(ip, port)
    import_robots(URDF_PATH, URDF_SIM_PATH)
    set_hide_real_robot(True)
    reset_sim_to_home()
    srv = getattr(wp, "server", None)
    init_cursors(srv)
    hide_sim_gizmo()
    set_hide_real_robot(True)
    try:
        if hasattr(wp, "ensure_builtin_sim_poses"):
            wp.ensure_builtin_sim_poses()
        wp.refresh_pose_markers()
    except Exception as e:
        print("extra arm poses", e)
    try:
        wp.ensure_sim_bot_marker()
    except Exception as e:
        print("extra arm marker", e)


def teardown_arm():
    """Remove everything an added arm drew (run as that arm)."""
    global robot, _cursors_ready
    try:
        exit_simulation_mode()
    except Exception:
        pass
    try:
        if hasattr(wp, "clear_waypoints"):
            wp.clear_waypoints()
    except Exception as e:
        print("teardown waypoints", e)
    try:
        import path_bake as bake
        bake.invalidate_from(0, drop_all=True)
        bake.clear_sample_dots()
    except Exception:
        pass
    hide_sim_gizmo()
    for h in (_cursor_sim, _arrow_sim, _arrow_sim_side, _cursor_real, _arrow_real, _arrow_real_side, _burn_line):
        if h is None:
            continue
        try:
            h.remove()
        except Exception:
            pass
    for pack in list((getattr(wp, "_pose_markers", None) or {}).values()):
        for h in (pack.values() if isinstance(pack, dict) else [pack]):
            try:
                h.remove()
            except Exception:
                pass
    try:
        wp._pose_markers.clear()
    except Exception:
        pass
    try:
        import simulatorview as sv
        sv.drop_sim_bot_gizmo()
        for h in list(sv._sim_bot_viz.values()):
            h.remove()
        sv._sim_bot_viz.clear()
    except Exception:
        pass
    for model in (robot_actual, robot_sim):
        _drop_viser_robot(model)
    robot = None
    _cursors_ready = False


CAM_LOOK = (0.35, 0.0, 0.30)
CAM_DIST = 1.85
CAM_VIEWS = {
    "front":  (1.0,  0.0,  0.12),
    "back":   (-1.0, 0.0,  0.12),
    "right":  (0.0,  1.0,  0.12),
    "left":   (0.0, -1.0,  0.12),
    "top":    (0.08, 0.0,  1.0),
    "bottom": (0.08, 0.0, -1.0),
    "iso":    (0.85, -0.75, 0.55),
}


def _camera_pose(name="front"):
    key = str(name or "front").lower()
    if key not in CAM_VIEWS and "," in key:
        parts = [float(x) for x in key.split(",")[:3]]
        vec = np.array(parts, dtype=float)
    else:
        vec = np.array(CAM_VIEWS.get(key, CAM_VIEWS["front"]), dtype=float)
    n = float(np.linalg.norm(vec)) or 1.0
    look = np.array(CAM_LOOK, dtype=float)
    pos = look + vec / n * CAM_DIST
    if key == "bottom":
        pos[2] = min(pos[2], look[2] - 0.9)
    return tuple(float(v) for v in pos), tuple(float(v) for v in look)


def camera_view_angles():
    """Yaw/pitch of the first client camera about the look-at point, degrees."""
    srv = getattr(viewer, "_server", None) if viewer is not None else None
    if srv is None:
        return {"yaw": -32.0, "pitch": 28.0}
    try:
        clients = list(srv.get_clients().values())
    except Exception:
        clients = []
    if not clients:
        return {"yaw": -32.0, "pitch": 28.0}
    cam = clients[0].camera
    try:
        p = np.asarray(cam.position, dtype=float).reshape(3)
        look = np.asarray(getattr(cam, "look_at", CAM_LOOK), dtype=float).reshape(3)
    except Exception:
        return {"yaw": -32.0, "pitch": 28.0}
    d = p - look
    yaw = math.degrees(math.atan2(float(d[1]), float(d[0])))
    pitch = math.degrees(math.atan2(float(d[2]), math.hypot(float(d[0]), float(d[1])) + 1e-9))
    return {"yaw": float(yaw), "pitch": float(pitch)}


def apply_camera_view(name="front"):
    pos, look = _camera_pose(name)
    srv = getattr(viewer, "_server", None) if viewer is not None else None
    if srv is None:
        return {"ok": False, "error": "no viewer"}
    n = 0
    try:
        clients = dict(srv.get_clients())
    except Exception:
        clients = {}
    ids = list(clients)
    world = _world_cids.intersection(ids) if _world_cids else set()
    if not world and ids:
        try:
            world = {min(ids, key=lambda k: int(k))}
        except Exception:
            world = {ids[0]}
    targets = [clients[i] for i in world if i in clients] or list(clients.values())[:1]
    for client in targets:
        try:
            client.camera.position = pos
            client.camera.look_at = look
            n += 1
        except Exception as e:
            print("camera set failed", e)
    print("camera", name, "pos", pos, "world_clients", n)
    return {"ok": True, "view": name, "position": list(pos), "look_at": list(look)}


_preview_cid = None
_world_cids = set()


def apply_scene_camera_preview(pos, wxyz, vfov=60.0, hfov=90.0, client_id=None):
    """Drive only extra Viser clients. The first/solo client is the world view."""
    global _preview_cid, _world_cids
    srv = getattr(viewer, "_server", None) if viewer is not None else None
    if srv is None:
        return False
    try:
        clients = dict(srv.get_clients())
    except Exception:
        return False
    ids = set(clients)
    if len(ids) <= 1:
        _world_cids = set(ids)
        _preview_cid = None
        return False
    if not _world_cids.intersection(ids):
        try:
            _world_cids = {min(ids, key=lambda k: int(k))}
        except Exception:
            _world_cids = {next(iter(ids))}
    preview_ids = [cid for cid in ids if cid not in _world_cids]
    if not preview_ids:
        return False
    if client_id in preview_ids:
        _preview_cid = client_id
    elif _preview_cid not in preview_ids:
        _preview_cid = preview_ids[-1]
    client = clients[_preview_cid]
    p = np.asarray(pos, dtype=float).reshape(3)
    q = np.asarray(wxyz if wxyz is not None else [1, 0, 0, 0], dtype=float).reshape(4)
    n = float(np.linalg.norm(q)) or 1.0
    q = q / n
    # Object frustum looks along local −Z. Euler XYZ 0,180,180 = 180° about X
    # maps local +Z → −Z so the preview matches the blue gizmo.
    # q_off = (0, 1, 0, 0); q' = q ⊗ q_off
    q = np.array([-q[1], q[0], q[3], -q[2]], dtype=float)
    n = float(np.linalg.norm(q)) or 1.0
    q = q / n
    fwd = np.array([
        2 * (q[1] * q[3] + q[0] * q[2]),
        2 * (q[2] * q[3] - q[0] * q[1]),
        1 - 2 * (q[1] * q[1] + q[2] * q[2]),
    ], dtype=float)
    try:
        client.camera.position = (float(p[0]), float(p[1]), float(p[2]))
        if hasattr(client.camera, "wxyz"):
            client.camera.wxyz = (float(q[0]), float(q[1]), float(q[2]), float(q[3]))
        else:
            look = p + fwd
            client.camera.look_at = (float(look[0]), float(look[1]), float(look[2]))
        if hasattr(client.camera, "fov"):
            try:
                client.camera.fov = float(vfov) * math.pi / 180.0
            except Exception:
                pass
        for attr in ("enable_rotate", "enable_zoom", "enable_pan", "enable_look_controls"):
            if hasattr(client.camera, attr):
                try:
                    setattr(client.camera, attr, False)
                except Exception:
                    pass
        return True
    except Exception as e:
        print("scene camera preview failed", e)
        return False


def apply_scene_camera_previews(targets):
    """Assign each extra Viser client a target (pos, wxyz, vfov, hfov)."""
    global _world_cids
    srv = getattr(viewer, "_server", None) if viewer is not None else None
    if srv is None or not targets:
        return False
    try:
        clients = dict(srv.get_clients())
    except Exception:
        return False
    ids = set(clients)
    if len(ids) <= 1:
        return False
    if not _world_cids.intersection(ids):
        try:
            _world_cids = {min(ids, key=lambda k: int(k))}
        except Exception:
            _world_cids = {next(iter(ids))}
    preview_ids = sorted(
        [cid for cid in ids if cid not in _world_cids],
        key=lambda k: int(k) if str(k).isdigit() else 0,
    )
    ok = False
    for i, cid in enumerate(preview_ids):
        if i >= len(targets):
            break
        rec = targets[i]
        if apply_scene_camera_preview(rec[0], rec[1], rec[2], rec[3], client_id=cid):
            ok = True
    return ok


def bind_default_camera():
    global _world_cids, _preview_cid
    srv = getattr(viewer, "_server", None) if viewer is not None else None
    if srv is None:
        return
    try:
        @srv.on_client_connect
        def _on_cam(client):
            global _world_cids, _preview_cid
            cid = getattr(client, "client_id", None)
            if cid is None:
                try:
                    for k, v in dict(srv.get_clients()).items():
                        if v is client:
                            cid = k
                            break
                except Exception:
                    cid = None
            if cid is None:
                cid = id(client)
            if not _world_cids:
                _world_cids = {cid}
                try:
                    pos, look = _camera_pose("iso")
                    client.camera.position = pos
                    client.camera.look_at = look
                except Exception:
                    pass
                return
            # Extra Viser clients are camera-pane previews — do not iso them.
            _preview_cid = cid
    except Exception as e:
        print("bind camera connect failed", e)



def _sync_sim_to_real():
    global _last_good_q
    try:
        q = _flat(robot.GetJoints(), 6)
    except Exception:
        q = None
    if q is None or len(q) < 6:
        return
    q = np.array(q, dtype=float)
    try:
        robot_actual.angle_vector(q)
        robot_sim.angle_vector(q)
        _last_good_q = q.copy()
        print("startup sim = real", q)
    except Exception as e:
        print("startup sync failed:", e)


def reset_sim_to_real():
    """Snap sim arm + orange cursor to the current real-robot pose."""
    global _last_good_q, _sim_wxyz, _sim_rot0, _sim_ik_pending, sim_mode, sim_path, _hold_sim
    with mode_lock:
        sim_mode = SimMode.FOLLOW
        sim_path = None
    q = None
    try:
        if robot is not None:
            q = _flat(robot.GetJoints(), 6)
    except Exception as e:
        print("reset GetJoints failed:", e)
    if (q is None or len(q) < 6) and robot_actual is not None:
        try:
            q = _flat(robot_actual.angle_vector(), 6)
        except Exception:
            q = None
    if q is None or len(q) < 6:
        print("reset: no real pose")
        return False
    q = np.array(q, dtype=float).reshape(-1)[:6]
    try:
        if robot_actual is not None:
            robot_actual.angle_vector(q)
        if robot_sim is not None:
            robot_sim.angle_vector(q)
    except Exception as e:
        print("reset angle_vector failed:", e)
        return False
    _last_good_q = q.copy()
    _sim_rot0 = None
    _sim_ik_pending = False
    _hold_sim = 12
    _snap_sim_cursor_to_ee()
    print("reset sim = real", q)
    return True


HOME_Q_DEG = [0.0, 0.0, 90.0, 0.0, 90.0, 0.0]
# always_aubo: Aubo GetIK only; offline sim IK features disabled.
# backup: local skrobot IK only while disconnected.
IK_PREF = "backup"


def set_ik_pref(mode):
    global IK_PREF
    m = str(mode or "backup").strip().lower()
    if m in ("always_aubo", "aubo", "always"):
        IK_PREF = "always_aubo"
    else:
        IK_PREF = "backup"
    print("ik pref", IK_PREF)
    return IK_PREF


def ik_pref():
    return IK_PREF


def ik_allowed():
    if robot_connected():
        return True
    return IK_PREF == "backup"


def use_local_ik():
    return (not robot_connected()) and IK_PREF == "backup"


def reset_sim_to_home():
    """Teleport sim arm + cursor to FK home (deg): 0, 0, 90, 0, 90, 0."""
    global _last_good_q, _sim_rot0, _sim_ik_pending, sim_mode, sim_path, _hold_sim
    with mode_lock:
        sim_mode = SimMode.FOLLOW
        sim_path = None
    q = np.deg2rad(np.array(HOME_Q_DEG, dtype=float))
    try:
        if robot_sim is not None:
            robot_sim.angle_vector(q)
    except Exception as e:
        print("home angle_vector failed:", e)
        return False
    _last_good_q = q.copy()
    _sim_rot0 = None
    _sim_ik_pending = False
    _hold_sim = 0
    try:
        if robot_ik is not None:
            _set_q(robot_ik, q)
    except Exception:
        pass
    _snap_sim_cursor_to_ee()
    print("reset sim = home", q)
    return True


def current_joint_q(source="sim"):
    """Return 6-vector in radians, or None."""
    try:
        if source == "real":
            q = None
            if robot is not None:
                try:
                    q = _flat(robot.GetJoints(), 6)
                except Exception:
                    q = None
            if (q is None or len(q) < 6) and robot_actual is not None:
                q = _flat(robot_actual.angle_vector(), 6)
            return q
        if robot_sim is not None:
            return _flat(robot_sim.angle_vector(), 6)
    except Exception:
        return None
    return None


def reset_sim_to_q(q, units="rad"):
    """Teleport sim arm + cursor to an FK vector."""
    global _last_good_q, _sim_rot0, _sim_ik_pending, sim_mode, sim_path, _hold_sim
    if q is None or len(list(q)) < 6:
        return False
    q = np.array([float(v) for v in list(q)[:6]], dtype=float)
    if str(units).lower().startswith("deg"):
        q = np.deg2rad(q)
    with mode_lock:
        sim_mode = SimMode.FOLLOW
        sim_path = None
    try:
        if robot_sim is not None:
            robot_sim.angle_vector(q)
    except Exception as e:
        print("pose angle_vector failed:", e)
        return False
    _last_good_q = q.copy()
    _sim_rot0 = None
    _sim_ik_pending = False
    _hold_sim = 12
    _snap_sim_cursor_to_ee()
    print("reset sim = custom q", q)
    return True


def _snap_sim_cursor_to_ee():
    """0.11.0 reset snap: move the orange handle onto current sim EE+offset."""
    if robot_sim is None:
        return
    pos, wxyz = _cursor_pose_from_ee(robot_sim, "sim")
    _force_sim_heading(wxyz)
    try:
        _track["pin_pos"] = np.asarray(pos, dtype=float).reshape(3).copy()
    except Exception:
        pass
    xyz = tuple(float(v) for v in np.asarray(pos, dtype=float).reshape(3))
    had_gizmo = _sim_gizmo is not None
    try:
        wp._suspend_gizmo_cb = True
        if _cursor_sim is not None:
            try:
                _cursor_sim.position = xyz
            except Exception:
                pass
        _draw_sim_dir(pos)
    finally:
        try:
            wp._suspend_gizmo_cb = False
        except Exception:
            pass
    srv = _sim_scene_server()
    if had_gizmo and srv is not None:
        _make_sim_gizmo(srv, pos, wxyz, resume=False)
    try:
        if hasattr(wp, "_propagate_children"):
            wp._propagate_children(arms.cid("sim_cursor"))
    except Exception:
        pass


_JOINT_ORDER = (
    "shoulder_joint", "upperArm_joint", "foreArm_joint",
    "wrist1_joint", "wrist2_joint", "wrist3_joint",
)


def set_joint_limits(vel_deg=None, acc_deg=None):
    """Preferences > Speed limits (deg/s, deg/s^2). The speed check reads these.

    Returns True when a limit changed, so the caller can re-score timing.
    """
    changed = False
    for table, vals in ((AUBO_I5_MAX_VEL_DEG, vel_deg), (AUBO_I5_MAX_ACCEL_DEG, acc_deg)):
        if not vals:
            continue
        for name, v in zip(_JOINT_ORDER, list(vals)[:6]):
            try:
                v = float(v)
            except (TypeError, ValueError):
                continue
            if v <= 0:
                continue
            if abs(table.get(name, 0.0) - v) > 1e-9:
                table[name] = v
                changed = True
    if changed:
        for model in (robot_actual, robot_sim, robot_ik):
            if model is not None:
                apply_aubo_i5_joint_limits(model)
    return changed


def joint_limits():
    """This arm's speed and accel limits (deg/s, deg/s^2), J1..J6."""
    return ([float(AUBO_I5_MAX_VEL_DEG.get(n, 0.0)) for n in _JOINT_ORDER],
            [float(AUBO_I5_MAX_ACCEL_DEG.get(n, 0.0)) for n in _JOINT_ORDER])


def apply_aubo_i5_joint_limits(model):
    for joint in model.joint_list:
        name = getattr(joint, "name", None)
        if name not in AUBO_I5_MAX_VEL_DEG:
            continue
        joint.max_joint_velocity = math.radians(AUBO_I5_MAX_VEL_DEG[name])


def set_opacity_only(model, opacity=153, rgb=None):
    alpha_u8 = int(np.clip(opacity, 0, 255))
    alpha_f = alpha_u8 / 255.0
    rgb_u8 = None if rgb is None else [int(np.clip(c, 0, 255)) for c in list(rgb)[:3]]
    rgb_f = None if rgb_u8 is None else [c / 255.0 for c in rgb_u8]
    meshes = []
    for link in model.link_list:
        m = getattr(link, "visual_mesh", None)
        if m is not None:
            meshes.append(m)
        extra = getattr(link, "visuals", None) or []
        for v in extra:
            gm = getattr(v, "geometry", None)
            if gm is not None:
                meshes.append(gm)
    for mesh in meshes:
        vis = getattr(mesh, "visual", None)
        if vis is None:
            continue
        for attr in ("vertex_colors", "face_colors"):
            cols = getattr(vis, attr, None)
            if cols is None:
                continue
            arr = np.asarray(cols)
            if arr.ndim != 2 or arr.shape[1] < 3:
                continue
            if arr.shape[1] == 3:
                if arr.dtype in (np.float32, np.float64):
                    a = np.full((arr.shape[0], 1), alpha_f, dtype=arr.dtype)
                else:
                    a = np.full((arr.shape[0], 1), alpha_u8, dtype=arr.dtype)
                arr = np.hstack([arr, a])
            else:
                arr = arr.copy()
                arr[:, 3] = alpha_u8 if arr.dtype not in (np.float32, np.float64) else alpha_f
            if rgb_u8 is not None:
                if arr.dtype in (np.float32, np.float64):
                    arr[:, 0], arr[:, 1], arr[:, 2] = rgb_f
                else:
                    arr[:, 0], arr[:, 1], arr[:, 2] = rgb_u8
            setattr(vis, attr, arr)
        mat = getattr(vis, "material", None)
        if mat is None:
            continue
        for attr in ("baseColorFactor", "mainColor", "diffuse", "ambient"):
            if not hasattr(mat, attr):
                continue
            val = getattr(mat, attr)
            if val is None:
                continue
            val = list(val)
            rgb = val[:3]
            if len(rgb) < 3:
                continue
            if max(float(c) for c in rgb) <= 1.01:
                while len(val) < 4:
                    val.append(1.0)
                if rgb_f is not None:
                    val[0], val[1], val[2] = rgb_f
                val[3] = alpha_f
            else:
                while len(val) < 4:
                    val.append(255)
                if rgb_u8 is not None:
                    val[0], val[1], val[2] = rgb_u8
                val[3] = alpha_u8
            try:
                setattr(mat, attr, val)
            except Exception:
                pass


def _link_wxyz_pos(link):
    pos = np.asarray(link.worldpos(), dtype=float).reshape(3)
    rot = np.asarray(link.worldrot(), dtype=float).reshape(3, 3)
    xyzw = R.from_matrix(rot).as_quat()
    wxyz = np.array([xyzw[3], xyzw[0], xyzw[1], xyzw[2]], dtype=float)
    return wxyz, pos


_hide_real_robot = False


def set_hide_real_robot(hide):
    global _hide_real_robot
    _hide_real_robot = bool(hide)
    vis = not _hide_real_robot
    if viewer is not None and robot_actual is not None:
        handles = getattr(viewer, "_linkid_to_handle", None) or {}
        for link in getattr(robot_actual, "link_list", None) or []:
            h = handles.get(str(id(link)))
            if h is None:
                continue
            try:
                h.visible = vis
            except Exception:
                pass
    for h in (_cursor_real, _arrow_real, _arrow_real_side):
        if h is None:
            continue
        try:
            h.visible = vis
        except Exception:
            pass
    return vis


def hide_real_robot_status():
    return {"hidden": bool(_hide_real_robot), "connected": robot_connected()}


def apply_sim_viewer_opacity(model, opacity=None, rgb=None):
    """Viser add_mesh_trimesh drops vertex/URDF alpha. Set handle.opacity,
    and replace GLB meshes with add_mesh_simple when needed."""
    if viewer is None or model is None:
        return 0
    if opacity is None:
        opacity = SIM_OPACITY
    alpha = float(opacity)
    if alpha > 1.0:
        alpha = alpha / 255.0
    alpha = float(np.clip(alpha, 0.05, 1.0))
    rgb = tuple(int(c) for c in (rgb or SIM_RGB)[:3])
    handles = getattr(viewer, "_linkid_to_handle", None)
    server = getattr(viewer, "_server", None)
    if not handles or server is None:
        return 0
    n = 0
    for link in model.link_list:
        lid = str(id(link))
        h = handles.get(lid)
        if h is None:
            continue
        applied = False
        if hasattr(h, "opacity"):
            try:
                h.opacity = alpha
                applied = True
            except Exception:
                applied = False
        kind = type(h).__name__
        if (not applied) or ("Glb" in kind) or ("glb" in kind.lower()):
            mesh = getattr(link, "concatenated_visual_mesh", None)
            if mesh is None:
                mesh = getattr(link, "visual_mesh", None)
            if mesh is None:
                if applied:
                    n += 1
                continue
            try:
                verts = np.asarray(mesh.vertices, dtype=np.float32)
                faces = np.asarray(mesh.faces, dtype=np.uint32)
            except Exception:
                if applied:
                    n += 1
                continue
            try:
                h.remove()
            except Exception:
                pass
            wxyz, pos = _link_wxyz_pos(link)
            try:
                new_h = server.scene.add_mesh_simple(
                    lid,
                    vertices=verts,
                    faces=faces,
                    color=rgb,
                    opacity=alpha,
                    wxyz=tuple(float(v) for v in wxyz),
                    position=tuple(float(v) for v in pos),
                    flat_shading=False,
                )
                handles[lid] = new_h
                applied = True
            except Exception as e:
                print("sim mesh opacity replace failed", lid, e)
        if applied:
            n += 1
    print("sim mesh opacity", alpha, "on", n, "links")
    return n


def _ee_link(model):
    if hasattr(model, "wrist3_Link"):
        return model.wrist3_Link
    moving = [
        j for j in model.joint_list
        if getattr(j, "joint_type", "revolute") not in ("fixed", "fixed_joint")
    ]
    if moving:
        child = getattr(moving[-1], "child_link", None)
        if child is not None:
            return child
    for lnk in reversed(list(model.link_list)):
        name = (getattr(lnk, "name", "") or "").lower()
        if name not in ("world", "base", "base_link"):
            return lnk
    return model.link_list[-1]


def _ee_frame(model):
    ee = _ee_link(model)
    pos = np.asarray(ee.worldpos(), dtype=np.float32).reshape(3)
    Rm = np.asarray(ee.worldrot(), dtype=np.float64)
    wxyz = np.array(R.from_matrix(Rm).as_quat()[[3, 0, 1, 2]], dtype=np.float32)
    return pos, wxyz


def _ee_from_cursor(pos, wxyz, which="sim"):
    Rm = R.from_quat(wxyz_to_xyzw(wxyz)).as_matrix()
    ee = np.asarray(pos, dtype=float).reshape(3) - Rm @ CURSOR_POS_OFFSET[which]
    return ee, wxyz


def _cursor_pose_from_ee(model, which):
    pos, wxyz = _ee_frame(model)
    Rm = R.from_quat(wxyz_to_xyzw(wxyz))
    extra = R.from_euler("xyz", CURSOR_RPY_OFFSET[which])
    pos = np.asarray(pos, dtype=float).reshape(3) + Rm.as_matrix() @ CURSOR_POS_OFFSET[which]
    wxyz_out = np.array((extra * Rm).as_quat()[[3, 0, 1, 2]], dtype=float)
    return pos, wxyz_out


def get_ee_pose(which="sim"):
    model = robot_sim if which == "sim" else robot_actual
    return _ee_frame(model)


def get_real_cursor_pose():
    return _cursor_pose_from_ee(robot_actual, "real")


def restore_sim_cursor(rec):
    """Apply saved cursor ik/joints onto the sim arm and its handle."""
    if not rec:
        return False
    joints = rec.get("joints") or rec.get("q")
    ik = rec.get("ik") or rec.get("pose")
    pos = wxyz = None
    if joints is not None and robot_sim is not None:
        try:
            q = np.asarray(joints, dtype=float).reshape(-1)[:6]
            robot_sim.angle_vector(q)
            pos, wxyz = _cursor_pose_from_ee(robot_sim, "sim")
        except Exception:
            pos = wxyz = None
    if pos is None and ik is not None:
        try:
            import path_bake as bake
            pos, wxyz = bake._pos_wxyz_from_ik(ik)
        except Exception:
            pos = wxyz = None
    if pos is None:
        return False
    _set_sim_heading(wxyz)
    if _cursor_sim is not None:
        try:
            _cursor_sim.position = tuple(float(v) for v in pos)
        except Exception:
            pass
    if wp.server is not None:
        _make_sim_gizmo(wp.server, pos, _sim_wxyz)
    _draw_sim_dir(pos)
    return True


def get_sim_cursor_pose():
    if _pb["active"] and _cursor_sim is not None:
        # Playback drives the cursor; a Track pin must not hold it (or its children) still.
        return np.asarray(_cursor_sim.position, dtype=float).reshape(3), np.array(_sim_wxyz, dtype=float)
    if _sim_gizmo is not None:
        pos = np.asarray(_sim_gizmo.position, dtype=float).reshape(3)
        return pos, np.array(_sim_wxyz, dtype=float)
    if _track.get("on") and _track.get("pin_pos") is not None:
        return np.asarray(_track["pin_pos"], dtype=float).reshape(3), np.array(_sim_wxyz, dtype=float)
    if _cursor_sim is not None:
        pos = np.asarray(_cursor_sim.position, dtype=float).reshape(3)
        return pos, np.array(_sim_wxyz, dtype=float)
    return _cursor_pose_from_ee(robot_sim, "sim")


def _make_real_cursor(server):
    sph = server.scene.add_icosphere(
        name="/cursor_real%s/sphere" % arms.ns(),
        radius=CURSOR_R,
        color=REAL_CURSOR_RGB,
        position=(0.35, 0.0, 0.35),
    )
    try:
        sph.opacity = 0.4
    except Exception:
        pass
    return sph


def apply_gizmo_space(mode="local"):
    global _sim_gizmo, _sim_wxyz
    if _sim_gizmo is None:
        return
    pos = tuple(float(v) for v in np.asarray(_sim_gizmo.position, dtype=float).reshape(3))
    stored = np.array(_sim_wxyz, dtype=float)
    _make_sim_gizmo(wp.server, pos, stored)


def hide_sim_gizmo():
    global _sim_gizmo, _sim_selected, _sim_rot0
    if _track.get("on") and not _pb["active"]:
        try:
            if _sim_gizmo is not None:
                _track["pin_pos"] = np.asarray(_sim_gizmo.position, dtype=float).reshape(3).copy()
            elif _cursor_sim is not None:
                _track["pin_pos"] = np.asarray(_cursor_sim.position, dtype=float).reshape(3).copy()
        except Exception:
            pass
    _sim_selected = False
    _sim_rot0 = None
    if _sim_gizmo is None:
        return
    try:
        _sim_gizmo.remove()
    except Exception:
        pass
    _sim_gizmo = None


def _look_at_wxyz(eye, target, up=(0.0, 0.0, 1.0)):
    eye = np.asarray(eye, dtype=float).reshape(3)
    target = np.asarray(target, dtype=float).reshape(3)
    z = target - eye
    n = float(np.linalg.norm(z))
    if n < 1e-8:
        return np.array(_sim_wxyz, dtype=float)
    z = z / n
    upv = np.asarray(up, dtype=float).reshape(3)
    x = np.cross(upv, z)
    if float(np.linalg.norm(x)) < 1e-6:
        x = np.cross(np.array([0.0, 1.0, 0.0]), z)
    x = x / (float(np.linalg.norm(x)) or 1.0)
    y = np.cross(z, x)
    y = y / (float(np.linalg.norm(y)) or 1.0)
    Rm = np.column_stack((x, y, z))
    q_xyzw = R.from_matrix(Rm).as_quat()
    return np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]], dtype=float)


def _aim_wxyz(pos, tgt, offset=None, pos_offset=None):
    """Look At heading for a cursor at pos.

    The aiming frame sits at pos_offset (cursor-local) and the cursor is that
    frame's look-at turned by the rx/ry/rz rot offset. With no pos offset this
    is the original look-at-then-rotate.
    """
    pos = np.asarray(pos, dtype=float).reshape(3)
    tgt = np.asarray(tgt, dtype=float).reshape(3)
    off = list(offset or [0.0, 0.0, 0.0])
    while len(off) < 3:
        off.append(0.0)
    lp = np.asarray(list(pos_offset or [0.0, 0.0, 0.0])[:3] + [0.0] * 3, dtype=float)[:3]
    try:
        r_off = R.from_euler("xyz", off[:3])
    except Exception:
        r_off = R.identity()
    r_obj = None
    eye = pos
    # The eye moves with the heading; a few passes settle it.
    for _ in range(6 if float(np.linalg.norm(lp)) > 1e-9 else 1):
        look = _look_at_wxyz(eye, tgt)
        r_obj = R.from_quat([look[1], look[2], look[3], look[0]]) * r_off
        eye = pos + r_obj.apply(lp)
    q = r_obj.as_quat()
    wxyz = np.array([q[3], q[0], q[1], q[2]], dtype=float)
    return wxyz / (float(np.linalg.norm(wxyz)) or 1.0)


def look_at_wxyz_offset(pos, name, offset=None, pos_offset=None):
    tgt = _track_target_pos(name)
    if tgt is None or pos is None:
        return None
    return _aim_wxyz(pos, tgt, offset, pos_offset)


def offset_to_look_at(rec):
    """Offset component -> Look At (pos offset, rx/ry/rz) that aims the same way.

    Look At turns the look-at frame by the rot offset to get the cursor, while
    the Offset frame is the cursor turned by its local rotation, so the rot
    offset is the inverse of the Offset's local rotation.
    """
    lp = [float(v) for v in list(rec.get("pos") or [0, 0, 0])[:3]]
    lq = np.asarray(rec.get("wxyz") or [1, 0, 0, 0], dtype=float).reshape(4)
    eu = R.from_quat([lq[1], lq[2], lq[3], lq[0]]).inv().as_euler("xyz")
    return lp, [float(v) for v in eu]


def _offset_state(rec):
    """'ok' | 'disabled' | 'missing' for the Look At warning label."""
    if not rec:
        return "missing"
    return "ok" if rec.get("enabled", True) else "disabled"


# Look At name that follows the camera script's primary target (red sphere).
PRIMARY_TARGET_NAME = "Primary target"


def _track_target_pos(name):
    key = str(name or "").strip().lower()
    if not key:
        return None
    if key == PRIMARY_TARGET_NAME.lower() and hasattr(wp, "primary_target_pos"):
        return wp.primary_target_pos()
    for obj in getattr(wp, "scene_objects", None) or []:
        if str(obj.get("name") or "").strip().lower() == key:
            return np.asarray(obj.get("pos"), dtype=float).reshape(3)
    poses = list(getattr(wp, "sim_poses", None) or []) + list(getattr(wp, "BUILTIN_SIM_POSES", ()) or [])
    for rec in poses:
        if str(rec.get("name") or rec.get("id") or "").strip().lower() != key:
            continue
        if rec.get("id") == "real" or str(rec.get("name") or "").lower() == "real pose":
            try:
                p, _ = get_real_cursor_pose()
                return np.asarray(p, dtype=float).reshape(3)
            except Exception:
                return None
        ik = rec.get("ik")
        if ik is not None and len(ik) >= 3:
            return np.asarray(ik[:3], dtype=float).reshape(3)
        q = rec.get("q")
        if q is None and rec.get("q_deg") is not None:
            q = np.deg2rad(np.asarray(rec["q_deg"], dtype=float))
        if q is not None and robot_sim is not None:
            try:
                hold = np.array(robot_sim.angle_vector(), dtype=float)
                robot_sim.angle_vector(np.asarray(q, dtype=float).reshape(-1)[:6])
                p, _ = _ee_frame(robot_sim)
                robot_sim.angle_vector(hold)
                return np.asarray(p, dtype=float).reshape(3)
            except Exception:
                return None
    return None


def _cursor_children():
    """Objects parented straight to the sim cursor: the swappable EE tools."""
    return [o for o in (getattr(wp, "scene_objects", None) or []) if str(o.get("parent_id") or "") == arms.cid("sim_cursor")]


def _tool_offsets(obj):
    """Look At (pos offset, rx/ry/rz) from a cursor child's local transform."""
    rec = {
        "pos": [float(v) for v in np.asarray(obj.get("local_pos", [0, 0, 0]), dtype=float).reshape(3)],
        "wxyz": [float(v) for v in np.asarray(obj.get("local_wxyz", [1, 0, 0, 0]), dtype=float).reshape(4)],
    }
    return offset_to_look_at(rec)


def _use_tool(key):
    """Copy a cursor child's local transform into the Look At offsets."""
    key = str(key or "").strip()
    if not key:
        _track["tool"] = ""
        return True
    for obj in _cursor_children():
        if key in (str(obj.get("id")), str(obj.get("name") or "")):
            _track["pos_offset"], _track["offset"] = _tool_offsets(obj)
            _track["tool"] = str(obj.get("name") or obj.get("id"))
            return True
    return False


def set_track_object(on=None, name=None, pick=None, added=None, offset=None, pos_offset=None, copy_offset=None,
                     tool=None, tool_pick=None):
    _track["_ik_key"] = None
    # The page sends both offsets with every change; hand edits that differ
    # from the chosen tool's values drop the tool label.
    if tool is None and _track.get("tool"):
        for key, val in (("pos_offset", pos_offset), ("offset", offset)):
            if val is None:
                continue
            try:
                if not np.allclose(np.asarray(val, dtype=float).reshape(-1)[:3],
                                   np.asarray(_track.get(key) or [0, 0, 0], dtype=float).reshape(-1)[:3], atol=1e-5):
                    _track["tool"] = ""
            except Exception:
                pass
    if tool_pick is not None:
        _track["tool_pick"] = bool(tool_pick)
    if pos_offset is not None:
        try:
            _track["pos_offset"] = [float(pos_offset[0]), float(pos_offset[1]), float(pos_offset[2])]
        except Exception:
            pass
    if on is not None:
        _track["on"] = on is True or str(on).lower() in ("1", "true", "on")
    if name is not None:
        _track["name"] = str(name)
    if pick is not None:
        _track["pick"] = bool(pick)
    if added is not None:
        if added and not _track.get("added") and hasattr(wp, "note_component"):
            wp.note_component("sim", "track")
        was = bool(_track.get("added"))
        _track["added"] = bool(added)
        if was != _track["added"] and hasattr(wp, "sync_camera_tracks"):
            try:
                wp.sync_camera_tracks()
            except Exception:
                pass
    if _track.get("on") and _track.get("pin_pos") is None:
        try:
            p, _ = get_sim_cursor_pose()
            if p is not None:
                _track["pin_pos"] = np.asarray(p, dtype=float).reshape(3).copy()
        except Exception:
            pass
    if on is False:
        _track["pin_pos"] = None
    if offset is not None:
        try:
            _track["offset"] = [float(offset[0]), float(offset[1]), float(offset[2])]
        except Exception:
            pass
    if tool is not None:
        # One-shot copy of the child's local transform; later moves of the child do not follow.
        _use_tool(tool)
    if copy_offset:
        # One-shot copy into Look At's own fields; later Offset edits do not follow.
        rec = getattr(wp, "sim_offset", None)
        if rec and rec.get("enabled", True):
            _track["pos_offset"], _track["offset"] = offset_to_look_at(rec)
    tgt = _track_target_pos(_track["name"])
    _track["valid"] = tgt is not None
    if _track.get("on"):
        try:
            tick_track_object()
        except Exception:
            pass
    return track_status()


def waypoint_track_record():
    """Look At record a waypoint added from the sim cursor keeps.

    Look At off still gives an explicit off record, so bake can tell these
    waypoints apart from older ones that never recorded anything (None).
    """
    st = track_status() or {}
    if not (st.get("on") and st.get("added") and st.get("name")):
        return {"on": False, "added": False, "name": "",
                "offset": [0.0, 0.0, 0.0], "pos_offset": [0.0, 0.0, 0.0]}
    return {
        "on": True,
        "added": True,
        "name": st.get("name") or "",
        "tool": st.get("tool") or "",
        "offset": list(st.get("offset") or [0.0, 0.0, 0.0]),
        "pos_offset": list(st.get("pos_offset") or [0.0, 0.0, 0.0]),
    }


def begin_track_pick():
    _track["pick"] = True
    _track["added"] = True
    return track_status()


def consume_tool_pick(oid):
    """Tool pick armed: a click on a cursor child (viewer or list) uses it."""
    if not _track.get("tool_pick"):
        return False
    _track["tool_pick"] = False
    _track["_ik_key"] = None
    return _use_tool(oid)


def consume_track_pick(name):
    if not _track.get("pick"):
        return False
    _track["pick"] = False
    if name:
        _track["name"] = str(name)
        _track["valid"] = _track_target_pos(_track["name"]) is not None
        return True
    return False


def track_status():
    tgt = _track_target_pos(_track.get("name"))
    _track["valid"] = tgt is not None
    pos = None if tgt is None else [float(tgt[0]), float(tgt[1]), float(tgt[2])]
    off = list(_track.get("offset") or [0.0, 0.0, 0.0])
    while len(off) < 3:
        off.append(0.0)
    return {
        "on": bool(_track.get("on")),
        "name": _track.get("name") or "",
        "pick": bool(_track.get("pick")),
        "valid": bool(_track.get("valid")),
        "added": bool(_track.get("added")),
        "error": bool(_track.get("on") and (_track.get("name") or "") and not _track.get("valid")),
        "pos": pos,
        "offset": [float(off[0]), float(off[1]), float(off[2])],
        "pos_offset": [float(v) for v in list(_track.get("pos_offset") or [0.0, 0.0, 0.0])[:3]],
        "offset_state": _offset_state(getattr(wp, "sim_offset", None)),
        "tool": _track.get("tool") or "",
        "tool_pick": bool(_track.get("tool_pick")),
        "tools": [{"id": str(o.get("id")), "name": str(o.get("name") or o.get("id"))} for o in _cursor_children()],
    }


def tick_track_object():
    if not _track.get("on"):
        return
    tgt = _track_target_pos(_track.get("name"))
    _track["valid"] = tgt is not None
    if tgt is None:
        return
    if _sim_gizmo is not None:
        pos = np.asarray(_sim_gizmo.position, dtype=float).reshape(3)
        _track["pin_pos"] = pos.copy()
    elif _track.get("pin_pos") is not None:
        pos = np.asarray(_track["pin_pos"], dtype=float).reshape(3)
    elif _cursor_sim is not None:
        pos = np.asarray(_cursor_sim.position, dtype=float).reshape(3)
        _track["pin_pos"] = pos.copy()
    else:
        pos, _ = get_sim_cursor_pose()
    if pos is None:
        return
    frozen = np.asarray(pos, dtype=float).reshape(3).copy()
    _track["pin_pos"] = frozen.copy()
    wxyz = _aim_wxyz(frozen, tgt, _track.get("offset"), _track.get("pos_offset"))
    n = float(np.linalg.norm(wxyz)) or 1.0
    global _sim_wxyz
    _sim_wxyz = wxyz / n
    # Re-solve only when the pin or the look-at changed. Backup IK costs
    # ~0.1 s per attempt, and this runs every viewer tick.
    key = tuple(np.round(frozen, 4)) + tuple(np.round(_sim_wxyz, 4))
    if key == _track.get("_ik_key"):
        return
    _track["_ik_key"] = key
    try:
        wp._suspend_gizmo_cb = True
        pin = (float(frozen[0]), float(frozen[1]), float(frozen[2]))
        if _cursor_sim is not None:
            _cursor_sim.position = pin
        if _sim_gizmo is not None:
            _sim_gizmo.position = pin
            _sim_gizmo.wxyz = tuple(float(v) for v in _sim_wxyz)
        _ik_sim_cursor()
        if _cursor_sim is not None:
            _cursor_sim.position = pin
        if _sim_gizmo is not None:
            _sim_gizmo.position = pin
    except Exception:
        pass
    finally:
        try:
            wp._suspend_gizmo_cb = False
        except Exception:
            pass
    _draw_sim_dir(frozen)
    # Children of the cursor follow the new heading now, not on their next move.
    try:
        if hasattr(wp, "_propagate_children"):
            wp._propagate_children(arms.cid("sim_cursor"))
        if hasattr(wp, "refresh_sim_camera_viz"):
            wp.refresh_sim_camera_viz()
    except Exception:
        pass


def _force_sim_heading(q):
    global _sim_wxyz
    q = np.asarray(q, dtype=float).reshape(4)
    n = float(np.linalg.norm(q)) or 1.0
    _sim_wxyz = q / n


def _sim_scene_server():
    srv = getattr(wp, "server", None)
    if srv is not None:
        return srv
    if viewer is None:
        return None
    return getattr(viewer, "_server", None) or getattr(viewer, "server", None)


def _set_sim_heading(q):
    global _sim_wxyz
    q = np.asarray(q, dtype=float).reshape(4)
    n = float(np.linalg.norm(q)) or 1.0
    q = q / n
    old = np.asarray(_sim_wxyz, dtype=float).reshape(4)
    if hasattr(wp, "_keep_heading"):
        _sim_wxyz = wp._keep_heading(old, q)
    else:
        _sim_wxyz = q


def _resume_sim_ik():
    """Same path as a gizmo update: solve EE to the current cursor, or hold."""
    return _ik_sim_cursor()


def _make_sim_gizmo(server, pos, wxyz, resume=True):
    global _sim_gizmo, _sim_selected, _sim_wxyz, _gizmo_serial
    heading = np.asarray(wxyz, dtype=float).reshape(4)
    hide_sim_gizmo()
    xyz = tuple(float(v) for v in np.asarray(pos, dtype=float).reshape(3))
    _force_sim_heading(heading)
    import waypoints as _wp
    space = getattr(_wp, "gizmo_space", "local")
    giz_wxyz = tuple(float(v) for v in _sim_wxyz) if space == "local" else (1.0, 0.0, 0.0, 0.0)
    _gizmo_serial += 1
    kwargs = dict(
        name="/cursor_sim%s/gizmo_%s_%d" % (arms.ns(), space, _gizmo_serial),
        position=xyz,
        scale=_GIZMO_SCALE_ON,
        wxyz=giz_wxyz,
    )
    try:
        _sim_gizmo = server.scene.add_transform_controls(
            **kwargs, space=("local" if space == "local" else "world")
        )
    except TypeError:
        _sim_gizmo = server.scene.add_transform_controls(**kwargs)
    if hasattr(_wp, "_try_set_space"):
        _wp._try_set_space(_sim_gizmo, "local" if space == "local" else "world")

    def _on_sim_gizmo(_=None):
        global _sim_ik_pending, _sim_rot0
        if getattr(wp, "_suspend_gizmo_cb", False):
            return
        if not ik_allowed():
            return
        _sim_ik_pending = True
        if _sim_gizmo is not None:
            if not (_track.get("on") and _track.get("valid")):
                if getattr(wp, "gizmo_space", "local") == "local":
                    _set_sim_heading(_sim_gizmo.wxyz)
                else:
                    if _sim_rot0 is None:
                        _sim_rot0 = np.array(_sim_wxyz, dtype=float)
                    if hasattr(wp, "_orientation_from_gizmo"):
                        _set_sim_heading(wp._orientation_from_gizmo(_sim_gizmo, _sim_wxyz, _sim_rot0))
            if _cursor_sim is not None:
                _cursor_sim.position = tuple(float(v) for v in _sim_gizmo.position)
            if hasattr(wp, "_mark_gizmo_moved"):
                wp._mark_gizmo_moved()
        _ik_sim_cursor()
        try:
            if hasattr(wp, "refresh_sim_camera_viz"):
                wp.refresh_sim_camera_viz()
        except Exception:
            pass
        try:
            if hasattr(wp, "_tick_sim_previews"):
                wp._tick_sim_previews()
        except Exception:
            pass
        try:
            if hasattr(wp, "_propagate_children"):
                wp._propagate_children(arms.cid("sim_cursor"))
        except Exception:
            pass

    if hasattr(_sim_gizmo, "on_update"):
        _sim_gizmo.on_update(_on_sim_gizmo)
    _sim_selected = True
    if resume:
        _resume_sim_ik()


def init_cursors(server):
    global _cursors_ready, _cursor_real, _arrow_real, _cursor_sim, _sim_wxyz
    if server is None:
        print("init_cursors: server is None")
        return
    try:
        _cursor_real = _make_real_cursor(server)
        pos, wxyz = _cursor_pose_from_ee(robot_sim, "sim")
        _sim_wxyz = np.asarray(wxyz, dtype=float).reshape(4)
        xyz = tuple(float(v) for v in pos)

        _cursor_sim = server.scene.add_icosphere(
            name="/cursor_sim%s/sphere" % arms.ns(),
            radius=CURSOR_R,
            color=SIM_CURSOR_RGB,
            position=xyz,
        )
        try:
            _cursor_sim.opacity = 0.4
        except Exception:
            pass

        def _toggle_sim(_=None):
            mode = "replace"
            if getattr(wp, "_select_ctrl", False):
                mode = "add"
            elif getattr(wp, "_select_shift", False):
                mode = "add"
            if hasattr(wp, "select_item"):
                wp.select_item("sim", None, mode=mode)
                return
            if _sim_gizmo is None:
                p, w = get_sim_cursor_pose()
                if _cursor_sim is not None:
                    p = np.asarray(_cursor_sim.position, dtype=float)
                _make_sim_gizmo(wp.server, p, w)
                wp.selection_kind = "sim"
            else:
                hide_sim_gizmo()
                if getattr(wp, "selection_kind", None) == "sim":
                    wp.selection_kind = None

        if hasattr(_cursor_sim, "on_click"):
            _cursor_sim.on_click(_toggle_sim)

        _make_sim_gizmo(server, pos, _sim_wxyz)
        _cursors_ready = True
        print("cursors ready EE", getattr(_ee_link(robot_sim), "name", None), "at", pos)
        try:
            wp.refresh_pose_markers()
        except Exception as e:
            print("pose markers", e)
    except Exception as e:
        _cursors_ready = False
        print("init_cursors FAILED:", repr(e))
        import traceback
        traceback.print_exc()


def _dir_from_wxyz(wxyz):
    Rm = R.from_quat(wxyz_to_xyzw(wxyz)).as_matrix()
    d = Rm[:, 2]
    n = float(np.linalg.norm(d)) or 1.0
    return d / n


def _right_from_wxyz(wxyz):
    Rm = R.from_quat(wxyz_to_xyzw(wxyz)).as_matrix()
    d = Rm[:, 0]
    n = float(np.linalg.norm(d)) or 1.0
    return d / n


_dir_lock = threading.Lock()


def _line_points(pos, wxyz, axis="fwd", scale=1.0):
    pos = np.asarray(pos, dtype=np.float32).reshape(3)
    vec = _dir_from_wxyz(wxyz) if axis == "fwd" else _right_from_wxyz(wxyz)
    tip = pos + np.asarray(vec, dtype=np.float32) * (ARROW_LEN * float(scale))
    return np.array([[pos, tip]], dtype=np.float32)


def _set_line(handle, name, pos, wxyz, color, axis="fwd", scale=1.0):
    pts = _line_points(pos, wxyz, axis=axis, scale=scale)
    if handle is not None:
        try:
            handle.points = pts
            return handle
        except Exception:
            handle = None
    server = wp.server
    if server is None:
        return handle
    try:
        return server.scene.add_line_segments(
            name=name,
            points=pts,
            colors=np.array(color, dtype=np.uint8).reshape(3),
            line_width=4.0,
        )
    except Exception:
        return handle


def _draw_sim_dir(pos=None):
    global _arrow_sim, _arrow_sim_side
    if _cursor_sim is None:
        return
    if pos is None:
        pos = np.asarray(_cursor_sim.position, dtype=float).reshape(3)
    else:
        pos = np.asarray(pos, dtype=float).reshape(3)
        try:
            _cursor_sim.position = (float(pos[0]), float(pos[1]), float(pos[2]))
        except Exception:
            pass
    with _dir_lock:
        _arrow_sim = _set_line(_arrow_sim, "/cursor_sim%s/dir" % arms.ns(), pos, _sim_wxyz, SIM_CURSOR_RGB)
        _arrow_sim_side = _set_line(
            _arrow_sim_side, "/cursor_sim%s/dir_side" % arms.ns(), pos, _sim_wxyz, SIM_CURSOR_RGB,
            axis="right", scale=0.70,
        )


def update_cursors():
    global _arrow_real, _arrow_real_side, _hold_sim
    if not _cursors_ready:
        return
    server = wp.server
    if server is None:
        return

    if robot_actual is not None and _cursor_real is not None and not _hide_real_robot:
        pos_r, wxyz_r = get_real_cursor_pose()
        back_r = _dir_from_wxyz(wxyz_r)
        _cursor_real.position = (float(pos_r[0]), float(pos_r[1]), float(pos_r[2]))
        with _dir_lock:
            _arrow_real = _set_line(_arrow_real, "/cursor_real%s/dir" % arms.ns(), pos_r, wxyz_r, REAL_CURSOR_RGB)
            _arrow_real_side = _set_line(
                _arrow_real_side, "/cursor_real%s/dir_side" % arms.ns(), pos_r, wxyz_r, REAL_CURSOR_RGB,
                axis="right", scale=0.70,
            )

    tracking = bool(_track.get("on") and _track.get("valid"))
    if _sim_gizmo is not None:
        pos_s = np.asarray(_sim_gizmo.position, dtype=float).reshape(3)
        if tracking:
            _track["pin_pos"] = pos_s.copy()
        if not tracking:
            if getattr(wp, "gizmo_space", "local") == "local" and not getattr(wp, "_suspend_gizmo_cb", False):
                _set_sim_heading(_sim_gizmo.wxyz)
        _draw_sim_dir(pos_s)
    elif tracking:
        pin = _track.get("pin_pos")
        if pin is None and _cursor_sim is not None:
            pin = np.asarray(_cursor_sim.position, dtype=float).reshape(3)
            _track["pin_pos"] = pin.copy()
        if pin is not None:
            _draw_sim_dir(np.asarray(pin, dtype=float).reshape(3))
    elif robot_sim is not None:
        pos_s, wxyz_s = _cursor_pose_from_ee(robot_sim, "sim")
        _set_sim_heading(wxyz_s)
        _draw_sim_dir(pos_s)

    if _hold_sim > 0:
        _hold_sim -= 1
        return
    if sim_mode == SimMode.PLAY_PATH:
        return
    try:
        if robot_sim is not None:
            pos_m, _ = get_sim_cursor_pose()
            _note_motion("sim", robot_sim.angle_vector(), pos_m)
    except Exception:
        pass
    if _sim_ik_pending:
        _ik_sim_cursor()
    _update_grid_for_ee()


def _ee_z(which):
    try:
        pos, _ = get_ee_pose(which)
        return float(np.asarray(pos, dtype=float).reshape(-1)[2])
    except Exception:
        return None


def _iter_grid_handles(scene):
    out = []
    if scene is None:
        return out
    for attr in ("world_grid", "_world_grid", "grid"):
        h = getattr(scene, attr, None)
        if h is not None:
            out.append(h)
    named = getattr(scene, "_handle_from_path", None) or getattr(scene, "_handle_from_name", None)
    if callable(named):
        for name in ("/grid", "grid", "/world_grid", "world_grid"):
            try:
                h = named(name)
                if h is not None:
                    out.append(h)
            except Exception:
                pass
    nodes = getattr(scene, "_handles", None) or getattr(scene, "_nodes", None)
    if isinstance(nodes, dict):
        for name, h in nodes.items():
            if "grid" in str(name).lower():
                out.append(h)
    return out


def _set_handle_opacity(h, alpha):
    if h is None:
        return False
    ok = False
    for attr in ("opacity", "alpha"):
        if hasattr(h, attr):
            try:
                setattr(h, attr, alpha)
                ok = True
            except Exception:
                pass
    try:
        if hasattr(h, "colors") and h.colors is not None:
            c = np.asarray(h.colors)
            # leave geometry; many grids use a separate opacity field
    except Exception:
        pass
    return ok


def _update_grid_for_ee():
    global _grid_dimmed
    zs = [_ee_z("sim"), _ee_z("real")]
    below = any(z is not None and z < 0.0 for z in zs)
    if _grid_dimmed is below:
        return
    scene = None
    try:
        srv = getattr(viewer, "_server", None) if viewer is not None else None
        scene = getattr(srv, "scene", None) if srv is not None else None
        if scene is None and wp.server is not None:
            scene = getattr(wp.server, "scene", None)
    except Exception:
        scene = None
    alpha = _GRID_OPACITY_DIM if below else _GRID_OPACITY_FULL
    applied = False
    for h in _iter_grid_handles(scene):
        if _set_handle_opacity(h, alpha):
            applied = True
    if applied:
        _grid_dimmed = below
        print("grid opacity", alpha, "ee_below", below)


def _ik_to_xyzrpy(pose, seed=None):
    pose = [float(v) for v in list(pose)[:6]]
    if len(pose) < 6:
        return None
    q_sim = _flat(seed, 6) if seed is not None else _flat(robot_sim.angle_vector(), 6)
    try:
        raw = robot.GetIK(pose, seed=q_sim)
    except Exception as e:
        if getattr(_ik_to_xyzrpy, "_err", None) != str(e):
            _ik_to_xyzrpy._err = str(e)
            print("GetIK failed:", e)
        return None
    if raw is None:
        return None
    raw = [float(v) for v in raw][:6]
    if len(raw) != 6 or np.allclose(raw[:3], pose[:3], atol=1e-4):
        return None
    return np.asarray(raw, dtype=float).reshape(-1)[:6]


def _ik_to_pose(pos, wxyz, seed=None):
    if robot_connected():
        pose = aubo_pose_from_viewer(pos, wxyz)
        return _ik_to_xyzrpy(pose, seed=seed)
    if use_local_ik():
        q_now = seed
        if q_now is None and robot_sim is not None:
            q_now = _flat(robot_sim.angle_vector(), 6)
        return solve_backup_ik(pos, wxyz, q_now)
    return None


def ik_best(pos, wxyz, seeds=None, use_robot=False):
    """Local IK only by default. GetIK is RPC and too slow to call per sample."""
    cands = []
    seen = []
    seed_list = []
    for s in seeds or []:
        if s is None:
            continue
        q = np.asarray(s, dtype=float).reshape(-1)[:6]
        if q.size >= 6:
            seed_list.append(q)
    # unique seeds, cap at 2
    uniq = []
    for s in seed_list:
        key = tuple(np.round(s, 4))
        if key in seen:
            continue
        seen.append(key)
        uniq.append(s)
        if len(uniq) >= 2:
            break
    seen = []

    def _add(q):
        if q is None:
            return
        q = np.asarray(q, dtype=float).reshape(-1)[:6]
        if q.size < 6:
            return
        wrapped = ((q + np.pi) % (2.0 * np.pi)) - np.pi
        key = tuple(np.round(wrapped, 5))
        if key in seen:
            return
        seen.append(key)
        cands.append(wrapped)

    for s in uniq:
        _add(sim_ik(pos, wxyz, seed=s))
    if not cands and use_robot and robot is not None:
        pose = aubo_pose_from_viewer(pos, wxyz)
        try:
            _add(robot.GetIK(pose, seed=(uniq[0] if uniq else None)))
        except Exception:
            pass
    if not cands:
        return None
    def score(q):
        return (int(np.sum(np.abs(q) <= np.pi + 1e-6)), -float(np.sum(np.abs(q))))
    cands.sort(key=score, reverse=True)
    return cands[0]


def _q_vec(model):
    try:
        q = np.asarray(model.angle_vector(), dtype=float).reshape(-1)
    except Exception:
        return None
    if q.size < 6 or not np.all(np.isfinite(q)):
        return None
    return q


def _set_q(model, q):
    cur = _q_vec(model)
    if cur is None:
        model.angle_vector(np.asarray(q, dtype=float).reshape(-1)[:6])
        return
    out = cur.copy()
    qq = np.asarray(q, dtype=float).reshape(-1)
    n = min(out.size, qq.size)
    out[:n] = qq[:n]
    model.angle_vector(out)


def _ik_sim_cursor():
    global _sim_ik_pending, _last_good_q
    _sim_ik_pending = False
    if robot_sim is None:
        return

    q_hold = (
        np.array(_last_good_q, dtype=float)
        if _last_good_q is not None
        else np.array(robot_sim.angle_vector(), dtype=float)
    )

    cur_pos, cur_wxyz = get_sim_cursor_pose()
    ee_pos, ee_wxyz = _ee_from_cursor(cur_pos, cur_wxyz, "sim")
    q = _ik_to_pose(ee_pos, ee_wxyz)
    if q is None:
        robot_sim.angle_vector(q_hold)
        return

    robot_sim.angle_vector(q)
    pred_cursor, _ = _cursor_pose_from_ee(robot_sim, "sim")
    err = float(np.linalg.norm(np.asarray(pred_cursor) - np.asarray(cur_pos)))
    if err > max(IK_POS_TOL, 0.12):
        print("sim IK far from cursor", err)
    _last_good_q = q


def solve_backup_ik(target_pos, target_wxyz, seed_q=None):
    """Offline 6-DOF IK. Current q first, then home. No position-only fallback."""
    model = robot_ik
    if model is None:
        return None
    if seed_q is not None:
        seed = np.array(_flat(seed_q, 6), dtype=float)
    elif robot_sim is not None:
        seed = np.array(_flat(robot_sim.angle_vector(), 6), dtype=float)
    else:
        return None
    if seed.size < 6 or not np.all(np.isfinite(seed)):
        return None
    p_des = np.asarray(target_pos, dtype=float).reshape(3)
    if not np.all(np.isfinite(p_des)):
        return None
    home = np.deg2rad(np.array(HOME_Q_DEG, dtype=float))
    pose = aubo_pose_from_viewer(p_des, target_wxyz)
    rot = R.from_quat(wxyz_to_xyzw(target_wxyz)).as_matrix()

    def _pose_err(p, w):
        dp = float(np.linalg.norm(np.asarray(p, dtype=float).reshape(3) - p_des))
        try:
            qerr = R.from_matrix(rot).inv() * R.from_quat(wxyz_to_xyzw(w))
            da = float(np.linalg.norm(qerr.as_rotvec()))
        except Exception:
            da = 0.0
        return dp, da

    def _try(seed_use, tag):
        _set_q(model, seed_use)
        p0, w0 = _ee_frame(model)
        dpos, dang = _pose_err(p0, w0)
        if dpos <= 0.002 and dang <= 0.05:
            print("backup IK already there", tag, "pos", round(dpos, 4), "ang", round(dang, 4))
            return np.array(seed_use, dtype=float)
        print("backup IK target", [round(float(v), 4) for v in pose], "seed", tag, [round(float(v), 4) for v in seed_use])
        qq = _jacobian_ik6(model, p_des, rot, seed_use, iters=28)
        if qq is None:
            print("backup IK fail", tag, "target", [round(float(v), 4) for v in pose])
            return None
        _set_q(model, qq)
        p_fk, w_fk = _ee_frame(model)
        err, aerr = _pose_err(p_fk, w_fk)
        print(
            "backup IK q", [round(float(v), 4) for v in qq],
            "fk", [round(float(v), 4) for v in np.asarray(p_fk).reshape(3)],
            "err", round(err, 4), "ang", round(aerr, 4),
        )
        if err > max(float(IK_POS_TOL), 0.04):
            print("backup IK far", tag, "err", round(err, 4))
            return None
        if aerr > IK_ANG_TOL:
            print("backup IK off-axis", tag, "ang", round(aerr, 4))
            return None
        return qq

    with _ik_lock:
        try:
            q = _try(seed, "seed" if seed_q is not None else "live")
            if q is not None:
                return q
            if np.allclose(seed, home, atol=1e-6):
                return None
            q = _try(home, "home")
            if q is not None:
                print("backup IK used home seed")
            return q
        except Exception as e:
            print("backup IK exception", e)
            return None


def sim_ik(pos, wxyz, seed=None, strict=True, iters=40, ori=False):
    return solve_backup_ik(pos, wxyz, seed)


IK6_ROT_WEIGHT = 0.5
# Backup IK result is rejected above this orientation error (rad).
IK_ANG_TOL = 0.1


def _jacobian_ik6(model, p_des, R_des, seed_q, iters=28):
    """6-DOF damped Jacobian on hidden robot_ik (position + orientation)."""
    if model is None or seed_q is None:
        return None
    p_des = np.asarray(p_des, dtype=float).reshape(3)
    R_des = np.asarray(R_des, dtype=float).reshape(3, 3)
    q = np.array(seed_q, dtype=float).reshape(-1)[:6].copy()
    eps = 1.2e-3

    def _err_at(qv):
        _set_q(model, qv)
        p, w = _ee_frame(model)
        e_p = np.asarray(p, dtype=float).reshape(3) - p_des
        try:
            e_r = (R.from_matrix(R_des).inv() * R.from_quat(wxyz_to_xyzw(w))).as_rotvec()
        except Exception:
            e_r = np.zeros(3)
        return np.concatenate([e_p, e_r]), float(np.linalg.norm(e_p)), float(np.linalg.norm(e_r))

    # The step and the accept test must minimise the same weighted error.
    # Solving unweighted (rotation-dominated) steps but accepting on
    # dp + 0.15*da rejected the first step of any large re-orientation
    # (e.g. Track look-at), so backup IK failed where the pose was reachable.
    W = np.array([1.0, 1.0, 1.0, IK6_ROT_WEIGHT, IK6_ROT_WEIGHT, IK6_ROT_WEIGHT])
    try:
        e, dp, da = _err_at(q)
    except Exception:
        return None
    best_q = q.copy()
    best = float(np.linalg.norm(W * e))
    best_dp, best_da = dp, da
    for _ in range(max(8, int(iters))):
        if dp <= 0.002 and da <= 0.05:
            break
        J = np.zeros((6, 6), dtype=float)
        ok = True
        for j in range(6):
            qp = q.copy()
            qp[j] += eps
            try:
                ej, _, _ = _err_at(qp)
            except Exception:
                ok = False
                break
            J[:, j] = (ej - e) / eps
        if not ok:
            break
        Jw = W[:, None] * J
        lam = 0.02 + 0.5 * min(best, 0.5)
        try:
            dq = -np.linalg.solve(Jw.T.dot(Jw) + lam * np.eye(6), Jw.T.dot(W * e))
        except Exception:
            break
        step = np.clip(dq, -0.18, 0.18)
        moved = False
        for _ls in range(5):
            qn = q + step
            try:
                en, dpn, dan = _err_at(qn)
            except Exception:
                break
            cost = float(np.linalg.norm(W * en))
            if cost < best - 1e-7:
                q, e, dp, da = qn, en, dpn, dan
                best = cost
                best_q = q.copy()
                best_dp, best_da = dp, da
                moved = True
                break
            step = step * 0.5
        if not moved:
            break
    if best_dp + 0.15 * best_da > 0.12:
        return None
    return best_q


def _jacobian_ik(model, p_des, seed_q, iters=16):
    """Position-only damped step on hidden robot_ik. Accept only shrinking EE error."""
    if model is None or seed_q is None:
        return None
    p_des = np.asarray(p_des, dtype=float).reshape(3)
    q = np.array(seed_q, dtype=float).reshape(-1)[:6].copy()
    try:
        _set_q(model, q)
        p0, _ = _ee_frame(model)
    except Exception:
        return None
    if p0 is None:
        return None
    best_q = q.copy()
    best_d = float(np.linalg.norm(np.asarray(p0, dtype=float).reshape(3) - p_des))
    eps = 1.5e-3
    for _ in range(max(6, int(iters))):
        _set_q(model, q)
        p, _ = _ee_frame(model)
        if p is None:
            break
        e = np.asarray(p, dtype=float).reshape(3) - p_des
        d = float(np.linalg.norm(e))
        if d < best_d:
            best_d = d
            best_q = q.copy()
        if d <= 0.003:
            break
        J = np.zeros((3, 6), dtype=float)
        ok = True
        for j in range(6):
            qp = q.copy()
            qp[j] += eps
            _set_q(model, qp)
            pp, _ = _ee_frame(model)
            if pp is None:
                ok = False
                break
            J[:, j] = (np.asarray(pp, dtype=float).reshape(3) - p) / eps
        if not ok:
            break
        lam = 0.15 + 2.0 * min(d, 0.4)
        try:
            dq = -np.linalg.solve(J.T.dot(J) + lam * np.eye(6), J.T.dot(e))
        except Exception:
            break
        qn = q + np.clip(dq, -0.20, 0.20)
        _set_q(model, qn)
        pn, _ = _ee_frame(model)
        if pn is None:
            break
        dn = float(np.linalg.norm(np.asarray(pn, dtype=float).reshape(3) - p_des))
        if dn < d - 1e-5:
            q = qn
        else:
            q = best_q.copy()
            break
    if best_d > float(IK_POS_TOL):
        return None
    return best_q


def import_robots(urdf_path=URDF_PATH, urdf_sim_path=URDF_SIM_PATH):
    global robot_actual, robot_sim, robot_ik
    robot_actual = RobotModel()
    print("real urdf", urdf_path)
    robot_actual.load_urdf_file(urdf_path)
    apply_aubo_i5_joint_limits(robot_actual)
    robot_sim = RobotModel()
    print("sim urdf", urdf_sim_path)
    robot_sim.load_urdf_file(urdf_sim_path)
    apply_aubo_i5_joint_limits(robot_sim)
    robot_ik = RobotModel()
    robot_ik.load_urdf_file(urdf_sim_path)
    apply_aubo_i5_joint_limits(robot_ik)
    for model in (robot_actual, robot_sim, robot_ik):
        _apply_base(model)
    set_opacity_only(robot_sim, SIM_OPACITY, rgb=SIM_RGB)
    viewer.add(robot_actual)
    viewer.add(robot_sim)
    try:
        print("ik model joints", len(np.asarray(robot_ik.angle_vector()).reshape(-1)))
    except Exception:
        pass
    apply_sim_viewer_opacity(robot_sim, SIM_OPACITY, rgb=SIM_RGB)
    print("EE link", getattr(_ee_link(robot_sim), "name", None))


def start_hold_move(ik=None, q=None, mode="ik"):
    global _hold_move, _hold_target_ik, _hold_target_q, _hold_arrived, _hold_mode
    if robot is None or not robot_connected():
        print("hold move: no robot")
        return False
    want_fk = str(mode or "fk").lower() != "ik"
    _hold_target_ik = [float(v) for v in ik[:6]] if ik is not None else None
    q_src = [float(v) for v in q[:6]] if q is not None else None
    # Move-to-sim-pose / Home / path-start: always current or supplied joints.
    if q_src is None and robot_sim is not None:
        try:
            q_src = _flat(robot_sim.angle_vector(), 6)
        except Exception:
            q_src = None
    if want_fk and q_src is None:
        print("hold move: no sim q")
        return False
    _hold_mode = "fk" if (want_fk or q_src is not None) else "ik"
    _hold_target_q = q_src
    _hold_arrived = False
    if _hold_mode == "fk" and _hold_target_q is None:
        return False
    try:
        lin = max(0.001, float(HOLD_LIN_MPS))
        # Full hold profile is ~0.375 m/s TCP at SetSpeed(1) + default jvel.
        scale = max(0.02, min(1.0, lin / 0.375))
        try:
            robot.SetSpeed(scale)
        except Exception:
            pass
        try:
            robot.StartMove()
        except Exception:
            pass
        jvel = float(HOLD_MOVE_SPEED)
        jacc = float(HOLD_MOVE_ACCEL)
        if _hold_mode == "fk":
            if hasattr(robot, "MoveJAsync"):
                robot.MoveJAsync(_hold_target_q, jvel=jvel, jacc=jacc)
            else:
                robot.mc.moveJoint(_hold_target_q, jvel, jacc, 0.0, 0)
        elif hasattr(robot, "MoveLAsync") and _hold_target_ik is not None:
            try:
                robot.MoveLAsync(_hold_target_ik, v=lin)
            except TypeError:
                robot.MoveLAsync(_hold_target_ik)
        else:
            return False
        _hold_move = True
        print("hold move start", _hold_mode, "lin", lin, "m/s", "jvel", jvel, "jacc", jacc, "q", _hold_target_q)
        return True
    except Exception as e:
        print("hold move failed:", e)
        _hold_move = False
        return False


def stop_hold_move():
    global _hold_move
    _hold_move = False
    if robot is None:
        return
    try:
        robot.StopMove()
    except Exception:
        pass


def _hold_has_arrived():
    if robot is None:
        return False
    try:
        if _hold_mode == "fk" and _hold_target_q is not None:
            q = _flat(robot.GetJoints(), 6)
            return q is not None and np.allclose(q, _hold_target_q, atol=0.012)
        if _hold_target_ik is not None:
            p = robot.GetToolPose() or robot.GetTcpPose()
            if p is None:
                return False
            p = [float(v) for v in p[:6]]
            t = _hold_target_ik
            return (
                np.allclose(p[:3], t[:3], atol=0.003)
                and np.allclose(p[3:6], t[3:6], atol=0.03)
            )
    except Exception:
        return False
    return False


def poll_hold_move():
    global _hold_arrived
    if not _hold_move:
        return
    if _hold_has_arrived():
        _hold_arrived = True


def hold_move_status():
    mismatch = False
    try:
        if robot is not None and robot_sim is not None:
            qr = _flat(robot.GetJoints(), 6)
            qs = _flat(robot_sim.angle_vector(), 6)
            if qr is not None and qs is not None:
                mismatch = not np.allclose(qr, qs, atol=0.012)
    except Exception:
        mismatch = False
    return {
        "holding": bool(_hold_move),
        "arrived": bool(_hold_arrived) or not mismatch,
        "mismatch": bool(mismatch),
        "mode": _hold_mode,
    }


# Per arm: was its controller up on the last pass, and failed passes since.
_real_link = {"last_ok": None, "miss": 0}
REAL_RECONNECT_MISSES = 40  # ~2 s of failed passes between reconnect tries


def _probe_joints(bot):
    """Joints of a live, powered controller, else None. No arm state touched,
    so it runs without arms.LOCK (an RPC can take a while)."""
    if bot is None or not _rpc_probe_ready:
        return None
    try:
        if not bot.hasConnected():
            return None
        q = _flat(bot.GetJoints(), 6)
    except Exception:
        return None
    if q is None or len(q) < 6 or all(abs(float(v)) < 1e-9 for v in q[:6]):
        return None
    return q


def _on_real_up(first):
    global _start_snap_done
    set_hide_real_robot(False)
    if not _start_snap_done:
        try:
            reset_sim_to_real()
            _start_snap_done = True
            print("startup snap sim = real", arms.name_of())
        except Exception as e:
            print("startup snap failed", e)
    if not first:
        try:
            import path_bake as bake
            bake.clear_bake()
            print("bake cleared after reconnect", arms.name_of())
        except Exception as e:
            print("bake clear failed", e)
    try:
        wp.refresh_pose_markers()
    except Exception:
        pass


def _real_step(aid):
    """One pass of the real-robot updater for one arm."""
    with arms.using(aid):
        bot = robot
        link = _real_link
    q = _probe_joints(bot)
    if q is None:
        with arms.using(aid):
            if link.get("last_ok"):
                print("robot dropped — hiding real arm", arms.name_of())
                set_hide_real_robot(True)
                try:
                    wp.refresh_pose_markers()
                except Exception:
                    pass
            link["last_ok"] = False
        link["miss"] = int(link.get("miss") or 0) + 1
        if bot is not None and link["miss"] >= REAL_RECONNECT_MISSES:
            link["miss"] = 0
            # reconnect() can wait seconds: never on this loop, never under the lock.
            threading.Thread(target=bot.reconnect, daemon=True).start()
        return
    link["miss"] = 0
    with arms.using(aid):
        if robot is not bot:
            return  # address changed meanwhile
        if not link.get("last_ok"):
            if link.get("last_ok") is False:
                print("robot reconnected", arms.name_of())
            _on_real_up(first=link.get("last_ok") is None)
            link["last_ok"] = True
        try:
            if robot_actual is not None:
                robot_actual.angle_vector(np.array(q, dtype=float))
            try:
                ee, _ = _ee_frame(robot_actual)
                _note_motion("real", q, ee)
            except Exception:
                _note_motion("real", q, None)
            poll_hold_move()
        except Exception as e:
            print("Real pose update error:", e)
            link["last_ok"] = False


def real_robot_updater():
    global _rpc_probe_ready
    time.sleep(3.0)
    _rpc_probe_ready = True
    while not stop_flag.is_set():
        for aid in arms.each():
            try:
                _real_step(aid)
            except Exception as e:
                print("Real pose update error:", e)
        time.sleep(0.05)


def sim_robot_controller():
    global sim_mode, sim_path, _sim_ik_pending, _hold_sim
    while not stop_flag.is_set():
        with mode_lock:
            mode = sim_mode
            path = sim_path
        if mode == SimMode.FOLLOW:
            time.sleep(0.05)
        elif mode == SimMode.PLAY_PATH and path is not None:
            hide_sim_gizmo()
            play_path(robot_sim, path, duration=sim_path_duration)
            _snap_sim_cursor_to_ee()
            time.sleep(return_delay)
            with mode_lock:
                sim_mode = SimMode.FOLLOW
                sim_path = None
            _sim_ik_pending = False
            _hold_sim = 8
            _snap_sim_cursor_to_ee()
        else:
            # Clock-driven PLAY_PATH (tick_simulation). Do not busy-spin.
            time.sleep(0.01)


def play_path(model, path, duration=2.0):
    qs = []
    for q in path:
        q = np.asarray(q, dtype=float).reshape(-1)[:6]
        if q.size < 6:
            continue
        qs.append(np.clip(q, -3.05, 3.05))
    if len(qs) < 2 or model is None:
        return
    nseg = len(qs) - 1
    steps_per_segment = max(4, int(max(20, 80 / nseg)))
    total_steps = steps_per_segment * nseg
    dt = max(0.01, float(duration) / total_steps)
    for i in range(nseg):
        if sim_mode != SimMode.PLAY_PATH:
            break
        start, end = qs[i], qs[i + 1]
        for s in range(1, steps_per_segment + 1):
            if sim_mode != SimMode.PLAY_PATH:
                break
            a = s / steps_per_segment
            a = a * a * (3 - 2 * a)
            model.angle_vector(start * (1.0 - a) + end * a)
            time.sleep(dt)


_pb = {
    "active": False,
    "paused": False,
    "t": 0.0,
    "t_end": 0.0,
    "samples": [],
    "saved_q": None,
    "saved_pos": None,
    "saved_wxyz": None,
    "finish_at": None,
    "lin_mps": 0.0,
    "last_ee": None,
    "last_t": None,
    "playhead_restore": None,
}


def _apply_q_at_t(t):
    samples = _pb.get("samples") or []
    if not samples or robot_sim is None:
        return
    if t <= samples[0]["t"]:
        q = samples[0]["q"]
    elif t >= samples[-1]["t"]:
        q = samples[-1]["q"]
    else:
        q = samples[-1]["q"]
        for a, b in zip(samples, samples[1:]):
            if a["t"] <= t <= b["t"]:
                qa = np.asarray(a["q"], dtype=float).reshape(-1)[:6]
                qb = np.asarray(b["q"], dtype=float).reshape(-1)[:6]
                if float(np.max(np.abs(qa - qb))) < 1e-5:
                    q = qa
                else:
                    span = max(1e-9, b["t"] - a["t"])
                    u = (t - a["t"]) / span
                    q = (1 - u) * qa + u * qb
                break
    robot_sim.angle_vector(np.asarray(q, dtype=float).reshape(-1)[:6])


def enter_simulation_mode(loops=0, delay=0.0):
    import path_bake as bake
    global sim_mode, _sim_ik_pending, _hold_sim, _halt
    _halt = False
    samples = bake.padded_timed_joints() if hasattr(bake, "padded_timed_joints") else (
        bake.timed_joint_samples() if hasattr(bake, "timed_joint_samples") else []
    )
    ik_ok = bake.is_ik_playable() if hasattr(bake, "is_ik_playable") else len(samples) >= 2
    if len(samples) < 2 or not ik_ok:
        print("simulate: need IK-baked samples")
        return False
    # Sim plays baked q(t) as-is. Joint vel limits do not block it.
    _pb["had_gizmo"] = _sim_gizmo is not None
    hide_sim_gizmo()
    pos, wxyz = get_sim_cursor_pose()
    _pb["saved_q"] = _flat(robot_sim.angle_vector(), 6) if robot_sim is not None else None
    _pb["saved_pos"] = np.array(pos, dtype=float)
    _pb["saved_wxyz"] = np.array(wxyz, dtype=float)
    _pb["samples"] = samples
    _pb["t_end"] = float(samples[-1]["t"])
    try:
        clear_speed_log()
    except Exception:
        pass
    restore_t = 0.0
    try:
        import robot_sim_run as viewer_main
        restore_t = float(getattr(viewer_main, "playhead_t", 0.0) or 0.0)
    except Exception:
        restore_t = 0.0
    _pb["playhead_restore"] = restore_t
    _pb["t"] = 0.0
    _pb["active"] = True
    _pb["paused"] = False
    _pb["finish_at"] = None
    _pb["wall"] = time.perf_counter()
    _pb["lin_mps"] = 0.0
    _pb["origin_wall"] = _pb["wall"]
    _pb["origin_t"] = 0.0
    try:
        _pb["loops_left"] = int(loops)
    except Exception:
        _pb["loops_left"] = 0
    try:
        _pb["loop_delay"] = max(0.0, float(delay or 0.0))
    except Exception:
        _pb["loop_delay"] = 0.0
    _pb["last_ee"] = None
    _pb["last_t"] = None
    _sim_ik_pending = False
    _hold_sim = 9999
    with mode_lock:
        sim_mode = SimMode.PLAY_PATH
        sim_path = None
    _apply_q_at_t(0.0)
    print("sim mode on", "t_end", _pb["t_end"], "n", len(samples))
    return True


def exit_simulation_mode():
    global sim_mode, _hold_sim, _sim_ik_pending, _last_good_q
    if not _pb["active"] and _pb["saved_q"] is None:
        return False
    _pb["active"] = False
    _pb["paused"] = False
    _pb["finish_at"] = None
    with mode_lock:
        sim_mode = SimMode.FOLLOW
        sim_path = None
    try:
        if _pb["saved_q"] is not None and robot_sim is not None:
            q = np.array(_pb["saved_q"], dtype=float)
            robot_sim.angle_vector(q)
            _last_good_q = q.copy()
        if _pb["saved_pos"] is not None:
            # Force, not keep: playback overwrote the heading with the EE's.
            _force_sim_heading(_pb["saved_wxyz"] if _pb["saved_wxyz"] is not None else _sim_wxyz)
            if _cursor_sim is not None:
                _cursor_sim.position = tuple(float(v) for v in _pb["saved_pos"])
            _draw_sim_dir(_pb["saved_pos"])
            # Bring the gizmo back if the cursor was selected before or during play.
            sim_picked = any(k == "sim" for k, _ in (getattr(wp, "_selection", None) or []))
            if (_pb.get("had_gizmo") or sim_picked) and wp.server is not None:
                _make_sim_gizmo(wp.server, _pb["saved_pos"], _sim_wxyz)
            try:
                if hasattr(wp, "_propagate_children"):
                    wp._propagate_children(arms.cid("sim_cursor"))
            except Exception:
                pass
    except Exception as e:
        # Sim mode is already off above; a restore error must not leave it on.
        print("sim exit restore failed:", e)
    _hold_sim = 8
    _sim_ik_pending = False
    _pb["saved_q"] = None
    restore_t = _pb.get("playhead_restore")
    _pb["playhead_restore"] = None
    if restore_t is not None:
        try:
            import robot_sim_run as viewer_main
            viewer_main.playhead_t = float(restore_t)
        except Exception:
            pass
    print("sim mode off")
    return True


def toggle_sim_pause():
    if not _pb["active"]:
        return False
    _pb["paused"] = not _pb["paused"]
    if _pb["paused"]:
        _pb["wall"] = None
        _pb["origin_wall"] = None
    else:
        now = time.perf_counter()
        _pb["wall"] = now
        _pb["origin_wall"] = now
        _pb["origin_t"] = float(_pb.get("t") or 0.0)
    return _pb["paused"]


def set_sim_time(t):
    if not _pb["active"]:
        return 0.0
    t = max(0.0, min(float(t), _pb["t_end"]))
    _pb["t"] = t
    _apply_q_at_t(t)
    _snap_sim_cursor_to_ee()
    try:
        wp._propagate_children(arms.cid("sim_cursor"))
    except Exception:
        pass
    return t


_last_vis = 0.0


def tick_simulation(dt=None):
    global _last_vis
    now = time.perf_counter()
    if _halt:
        _pb["active"] = False
        return
    if not _pb["active"] or _pb["paused"]:
        _pb["wall"] = None
        if _pb["active"] and _pb["finish_at"] is not None and time.time() >= _pb["finish_at"]:
            if _pb.get("_loop_pending") and not _halt:
                _pb["_loop_pending"] = False
                _pb["paused"] = False
                _pb["finish_at"] = None
                now = time.perf_counter()
                _pb["t"] = 0.0
                _pb["origin_t"] = 0.0
                _pb["origin_wall"] = now
                _pb["wall"] = now
                _apply_q_at_t(0.0)
                return
            exit_simulation_mode()
        return
    last = _pb.get("wall")
    _pb["wall"] = now
    if _pb.get("origin_wall") is None:
        _pb["origin_wall"] = now
        _pb["origin_t"] = float(_pb.get("t") or 0.0)
    t0 = _pb["samples"][0]["t"] if _pb["samples"] else 0.0
    _pb["t"] = min(_pb["t_end"], float(_pb["origin_t"]) + (now - float(_pb["origin_wall"])))
    moved = False
    if now - _last_vis >= 0.07:
        _last_vis = now
        moved = True
        if _pb["t"] < t0:
            _apply_q_at_t(t0)
        else:
            _apply_q_at_t(_pb["t"])
    if _cursor_sim is not None and robot_sim is not None:
        try:
            pos, ee_wxyz = _cursor_pose_from_ee(robot_sim, "sim")
            _cursor_sim.position = tuple(float(v) for v in pos)
            # Children of the cursor follow the EE rotation too; exit restores the heading.
            _force_sim_heading(ee_wxyz)
            try:
                if hasattr(wp, "_propagate_children"):
                    wp._propagate_children(arms.cid("sim_cursor"))
            except Exception:
                pass
            # Only score speed on ticks that moved the arm; the ticks between
            # see no motion and read 0, which flattened the speed graph.
            if moved:
                q_now = _flat(robot_sim.angle_vector(), 6)
                _note_motion("sim", q_now, pos, t=_pb.get("t"))
                _pb["lin_mps"] = float(_motion["sim"]["lin_mps"])
        except Exception:
            pass
    try:
        import robot_sim_run as viewer_main
        viewer_main.playhead_t = float(_pb["t"])
    except Exception:
        pass
    if _pb["t"] >= _pb["t_end"] - 1e-6 and _pb["finish_at"] is None:
        left = int(_pb.get("loops_left") or 0)
        if left < 0 or left > 0:
            if left > 0:
                _pb["loops_left"] = left - 1
            delay = float(_pb.get("loop_delay") or 0.0)
            if delay > 0:
                _pb["paused"] = True
                _pb["finish_at"] = time.time() + delay
                _pb["wall"] = None
                _pb["origin_wall"] = None
                _pb["_loop_pending"] = True
            else:
                now = time.perf_counter()
                _pb["t"] = 0.0
                _pb["origin_t"] = 0.0
                _pb["origin_wall"] = now
                _pb["wall"] = now
                _apply_q_at_t(0.0)
        else:
            try:
                import waypoints as _wp
                if hasattr(_wp, "stop_all_camera_records"):
                    _wp.stop_all_camera_records()
            except Exception:
                pass
            exit_simulation_mode()
            print("sim path finished")


def seek_playhead(t):
    """Idle timeline scrub. The viewer loop centres the path window on its
    playhead_t; without this it kept the last play time and fought the UI."""
    if _pb["active"] or _rp.get("active"):
        return False
    try:
        t = max(0.0, float(t))
    except (TypeError, ValueError):
        return False
    try:
        import robot_sim_run as viewer_main
        viewer_main.playhead_t = t
    except Exception:
        return False
    return True


def sim_status():
    return {
        "active": bool(_pb["active"]),
        "paused": bool(_pb["paused"]),
        "t": float(_pb["t"]),
        "t_end": float(_pb["t_end"]),
        "lin_mps": float(_pb.get("lin_mps") or 0.0),
        "playhead_restore": _pb.get("playhead_restore"),
    }


_rp = {
    "active": False,
    "paused": False,
    "t": 0.0,
    "t_end": 0.0,
    "samples": [],
    "wall": None,
    "origin_wall": None,
    "origin_t": 0.0,
    "finish_at": None,
    "at_start": False,
    "name": "rec",
}
_halt = False
_rp_gen = 0


def _abort_real_motion(name=None):
    """Talk to the controller on this thread. stopMove alone does not abort movePathBuffer."""
    if robot is None:
        return
    buf = str(name or _rp.get("name") or "rec")
    try:
        if hasattr(robot, "AbortPathBuffer"):
            robot.AbortPathBuffer(buf)
            return
    except Exception as e:
        print("AbortPathBuffer failed:", e)
    try:
        robot.StopMove()
    except Exception:
        pass
    try:
        robot.mc.stopMove(True, True)
    except Exception:
        pass
    try:
        robot.mc.stopJoint(8.0)
    except Exception:
        pass
    try:
        robot.mc.stopLine(4.0, 8.0)
    except Exception:
        pass
    try:
        robot.mc.clearPath()
    except Exception:
        pass
    try:
        robot.mc.pathBufferFree(buf)
    except Exception:
        pass


def request_halt():
    global _halt, _rp_gen
    _halt = True
    _rp_gen += 1
    _rp["active"] = False
    _rp["paused"] = False
    _rp["armed"] = False
    _rp["finish_at"] = None
    _rp["wall"] = None
    _pb["active"] = False
    _pb["paused"] = False
    _pb["loops_left"] = 0
    _pb["_loop_pending"] = False
    _rp["loops_left"] = 0
    _rp["_loop_pending"] = False
    _rp["uploading"] = False
    _rp["uploaded"] = False
    _rp["phase"] = ""
    _abort_real_motion(_rp.get("name") or "rec")
    try:
        exit_simulation_mode()
    except Exception:
        pass
    # One-shot: upload threads see _halt, then we clear so clock/sim can run again.
    _halt = False
    print("halt requested")
    try:
        import waypoints as _wp
        if hasattr(_wp, "stop_all_camera_records"):
            _wp.stop_all_camera_records()
    except Exception:
        pass
    return True


def _free_buffer_bg():
    _abort_real_motion(_rp.get("name") or "rec")


def edits_locked():
    return bool(_pb.get("active") or _rp.get("active"))


def _first_path_q():
    import path_bake as bake
    samples = bake.padded_timed_joints() if hasattr(bake, "padded_timed_joints") else []
    if not samples:
        return None
    return [float(v) for v in samples[0]["q"][:6]]


def real_at_start():
    q0 = _first_path_q()
    if q0 is None or robot is None:
        return False
    try:
        qr = _flat(robot.GetJoints(), 6)
        return qr is not None and np.allclose(qr, q0, atol=0.012)
    except Exception:
        return False


def start_goto_path_start():
    q0 = _first_path_q()
    if q0 is None:
        return False
    return start_hold_move(q=q0, mode="fk")


def _path_samples():
    import path_bake as bake
    samples = []
    if hasattr(bake, "padded_timed_joints"):
        samples = bake.padded_timed_joints() or []
    if len(samples) < 2 and hasattr(bake, "execution_joint_samples"):
        samples = bake.execution_joint_samples() or []
    if len(samples) < 2:
        extra = []
        for seg in getattr(bake, "segments", None) or []:
            for sm in seg.get("samples") or []:
                if sm.get("q") is not None and sm.get("t") is not None:
                    extra.append({
                        "t": float(sm["t"]),
                        "q": list(np.asarray(sm["q"], dtype=float).reshape(-1)[:6]),
                    })
        extra.sort(key=lambda r: r["t"])
        samples = extra
    return samples


def drop_uploaded_buffer():
    _rp["uploaded"] = False
    if not _rp.get("active"):
        _rp["uploading"] = False
        if _rp.get("phase") in ("uploaded", "uploading"):
            _rp["phase"] = ""


def upload_real_path(loops=0, delay=0.0):
    """Plan + write type-2 buffer only. Does not arm or movePathBuffer."""
    global _halt, _rp_gen
    _halt = False
    _rp_gen += 1
    gen = _rp_gen
    import path_bake as bake
    if not bake.is_runnable():
        print("real path: not runnable")
        return False
    samples = _path_samples()
    if len(samples) < 2:
        print("real path: not enough samples")
        return False
    interval = bake.sample_interval(samples)
    qs = [s["q"] for s in samples]
    times = [float(s["t"]) for s in samples]
    _rp["samples"] = samples
    path_end = float(samples[-1]["t"])
    hold = 0.0
    try:
        import waypoints as _wp
        if hasattr(_wp, "last_camera_rec_t"):
            hold = float(_wp.last_camera_rec_t() or 0.0)
    except Exception:
        hold = 0.0
    preroll = 0.0
    try:
        import waypoints as _wp
        if hasattr(_wp, "rec_preroll_s"):
            preroll = float(_wp.rec_preroll_s() or 0.0)
    except Exception:
        preroll = 0.0
    _rp["path_end"] = path_end
    _rp["preroll"] = preroll
    _rp["motion_started"] = False
    _rp["t_end"] = max(path_end + preroll, hold)
    _rp["t"] = 0.0
    _rp["active"] = False
    _rp["paused"] = False
    _rp["armed"] = False
    _rp["uploading"] = True
    _rp["uploaded"] = False
    _rp["phase"] = "uploading"
    _rp["finish_at"] = None
    try:
        _rp["loops_left"] = int(loops)
    except Exception:
        _rp["loops_left"] = 0
    try:
        _rp["loop_delay"] = max(0.0, float(delay or 0.0))
    except Exception:
        _rp["loop_delay"] = 0.0
    _rp["_loop_pending"] = False

    def _upload_only():
        if _halt or gen != _rp_gen:
            _rp["uploading"] = False
            _rp["phase"] = "aborted"
            return
        try:
            plan = getattr(robot, "plan_joint_path", None) or getattr(robot, "prepare_path_buffer", None)
            if plan is None:
                raise AttributeError("no plan_joint_path")
            kw = {"interval": interval, "name": _rp["name"], "times": times}
            try:
                plan(qs, **kw)
            except TypeError:
                plan(qs, interval=interval, name=_rp["name"])
            if _halt or gen != _rp_gen:
                try:
                    _abort_real_motion(_rp["name"])
                except Exception:
                    pass
                _rp["uploading"] = False
                _rp["uploaded"] = False
                _rp["phase"] = "aborted"
                return
            _rp["uploaded"] = True
            _rp["uploading"] = False
            _rp["phase"] = "uploaded"
            print("real path uploaded n", len(qs), "interval", interval)
        except Exception as e:
            print("real path upload failed:", e)
            _rp["uploading"] = False
            _rp["uploaded"] = False
            _rp["phase"] = "failed: %s" % e

    threading.Thread(target=_upload_only, daemon=True).start()
    return True


def plan_joint_path(loops=0, delay=0.0):
    """UI/API name: plan + upload type-2 buffer only."""
    return upload_real_path(loops=loops, delay=delay)


def _start_path_buffer():
    if robot is None:
        return None
    try:
        robot.mc.setSpeedFraction(1.0)
    except Exception:
        pass
    run = getattr(robot, "run_path_buffer", None)
    if run is not None:
        run(_rp["name"], wait=False)
    else:
        robot.mc.movePathBuffer(_rp["name"])
    exec_id = None
    try:
        exec_id = robot.mc.getExecId()
    except Exception:
        exec_id = None
    if exec_id == -1:
        try:
            robot.mc.movePathBuffer(_rp["name"])
            exec_id = robot.mc.getExecId()
        except Exception as e:
            print("movePathBuffer retry failed:", e)
    return exec_id


def fire_real_path():
    """movePathBuffer on an already-uploaded buffer. No plan, no sleep."""
    if not _rp.get("uploaded"):
        print("real path: not uploaded")
        return False
    if _rp.get("uploading"):
        print("real path: still uploading")
        return False
    if not real_at_start():
        print("real path: not at start")
        return False
    if robot is None:
        return False
    preroll = float(_rp.get("preroll") or 0.0)
    try:
        import waypoints as _wp
        if hasattr(_wp, "rec_preroll_s"):
            preroll = float(_wp.rec_preroll_s() or 0.0)
            _rp["preroll"] = preroll
        hold = float(_wp.last_camera_rec_t() or 0.0) if hasattr(_wp, "last_camera_rec_t") else 0.0
        pe = float(_rp.get("path_end") or 0.0)
        _rp["t_end"] = max(pe + preroll, hold)
    except Exception:
        pass
    _rp["active"] = True
    _rp["paused"] = False
    _rp["armed"] = True
    _rp["motion_started"] = False
    _rp["phase"] = "running"
    _rp["t"] = 0.0
    _rp["finish_at"] = None
    try:
        clear_speed_log()
    except Exception:
        pass
    now0 = time.perf_counter()
    _rp["wall"] = now0
    _rp["origin_wall"] = now0
    _rp["origin_t"] = 0.0
    _rp["_last_tick"] = None
    if preroll <= 1e-6:
        _start_path_buffer()
        _rp["motion_started"] = True
    return True


def start_real_path(loops=0, delay=0.0):
    if _rp.get("uploaded") and not _rp.get("uploading"):
        return fire_real_path()
    print("real path: upload first")
    return False


def pause_real_path():
    if not _rp["active"]:
        return False
    _rp["paused"] = not _rp["paused"]
    _rp["wall"] = None
    if _rp["paused"]:
        _rp["origin_wall"] = None
    else:
        now = time.perf_counter()
        _rp["wall"] = now
        _rp["origin_wall"] = now
        _rp["origin_t"] = float(_rp.get("t") or 0.0)
    if robot is None:
        return _rp["paused"]
    try:
        if _rp["paused"]:
            try:
                robot.mc.setSpeedFraction(0.0)
            except Exception:
                pass
            try:
                robot.mc.stopMove(True, True)
            except Exception:
                pass
            print("real path paused")
        else:
            try:
                robot.mc.setSpeedFraction(1.0)
            except Exception:
                pass
            try:
                robot.mc.startMove()
            except Exception:
                pass
            # If the planner dropped the buffer during pause, re-feed the tail.
            exec_id = -1
            try:
                exec_id = robot.mc.getExecId()
            except Exception:
                exec_id = -1
            if exec_id == -1:
                import path_bake as bake
                # _rp["t"] runs from the camera preroll; sample times start at
                # 0 when motion starts, so cut the tail on path time.
                t = max(0.0, float(_rp["t"]) - float(_rp.get("preroll") or 0.0))
                rest = [s for s in _rp["samples"] if s["t"] >= t - 1e-6]
                if len(rest) < 2:
                    rest = (_rp["samples"] or [])[-2:]
                if len(rest) >= 2:
                    interval = bake.sample_interval(rest)
                    robot.plan_joint_path([s["q"] for s in rest], interval=interval, name=_rp["name"])
                    robot.run_path_buffer(_rp["name"], wait=False)
            print("real path resumed")
    except Exception as e:
        print("real path pause/resume failed:", e)
    return _rp["paused"]


def stop_real_path():
    global _halt, _rp_gen
    _halt = True
    _rp_gen += 1
    _rp["active"] = False
    _rp["paused"] = False
    _rp["armed"] = False
    _rp["uploading"] = False
    _rp["uploaded"] = False
    _rp["phase"] = ""
    _rp["finish_at"] = None
    _rp["wall"] = None
    _rp["loops_left"] = 0
    _rp["_loop_pending"] = False
    _abort_real_motion(_rp.get("name") or "rec")
    _halt = False
    print("real path off")
    try:
        import waypoints as _wp
        if hasattr(_wp, "stop_all_camera_records"):
            _wp.stop_all_camera_records()
    except Exception:
        pass
    return True


def tick_real_path():
    if not _rp["active"]:
        return
    if not _rp.get("armed"):
        return
    if _rp["paused"]:
        _rp["wall"] = None
        if _rp["finish_at"] is not None and time.time() >= _rp["finish_at"]:
            if _rp.get("_loop_pending") and not _halt:
                _rp["_loop_pending"] = False
                _rp["paused"] = False
                _rp["finish_at"] = None
                now2 = time.perf_counter()
                _rp["t"] = 0.0
                _rp["origin_t"] = 0.0
                _rp["origin_wall"] = now2
                _rp["wall"] = now2
                _rp["motion_started"] = True
                _start_path_buffer()
                return
            stop_real_path()
        return
    now = time.perf_counter()
    _rp["wall"] = now
    last = _rp.get("_last_tick")
    if last is None:
        _rp["origin_wall"] = now
        _rp["origin_t"] = float(_rp.get("t") or 0.0)
    _rp["_last_tick"] = now
    if _rp.get("origin_wall") is None:
        _rp["origin_wall"] = now
        _rp["origin_t"] = float(_rp.get("t") or 0.0)
    raw_t = float(_rp["origin_t"]) + (now - float(_rp["origin_wall"]))
    _rp["t"] = raw_t
    preroll = float(_rp.get("preroll") or 0.0)
    pe = float(_rp.get("path_end") or 0.0)
    if preroll > 1e-6 and not _rp.get("motion_started") and raw_t >= preroll - 1e-6:
        _start_path_buffer()
        _rp["motion_started"] = True
    path_t = max(0.0, raw_t - preroll)
    try:
        _apply_real_q_at_t(min(path_t, pe))
    except Exception:
        pass
    left = int(_rp.get("loops_left") or 0)
    looping = left < 0 or left > 0
    if path_t >= pe - 1e-6 and _rp["finish_at"] is None:
        if looping:
            if left > 0:
                _rp["loops_left"] = left - 1
            delay = float(_rp.get("loop_delay") or 0.0)
            if delay > 0:
                _rp["paused"] = True
                _rp["finish_at"] = time.time() + delay
                _rp["wall"] = None
                _rp["_loop_pending"] = True
            else:
                now2 = time.perf_counter()
                _rp["t"] = 0.0
                _rp["origin_t"] = 0.0
                _rp["origin_wall"] = now2
                _rp["motion_started"] = True
                _start_path_buffer()
        else:
            hold = pe + preroll
            rec_hold = 0.0
            try:
                import waypoints as _wp
                if hasattr(_wp, "last_camera_rec_t"):
                    rec_hold = float(_wp.last_camera_rec_t() or 0.0)
                    if rec_hold > 1e-6:
                        hold = max(hold, rec_hold)
            except Exception:
                pass
            _rp["t_end"] = hold
            exec_done = False
            if _rp.get("motion_started") and robot is not None:
                try:
                    exec_done = int(robot.mc.getExecId()) == -1
                except Exception:
                    exec_done = False
            if raw_t >= hold - 1e-6 or (exec_done and rec_hold <= 1e-6) or (pe > 1e-6 and path_t >= pe - 1e-6 and rec_hold <= 1e-6):
                print("real path finished", "t", round(raw_t, 3), "pe", round(pe, 3), "exec_done", exec_done)
                stop_real_path()


def _apply_real_q_at_t(t):
    samples = _rp.get("samples") or []
    if not samples:
        return
    if t <= samples[0]["t"]:
        q = samples[0]["q"]
    elif t >= samples[-1]["t"]:
        q = samples[-1]["q"]
    else:
        q = samples[-1]["q"]
        for a, b in zip(samples, samples[1:]):
            if a["t"] <= t <= b["t"]:
                qa = np.asarray(a["q"], dtype=float).reshape(-1)[:6]
                qb = np.asarray(b["q"], dtype=float).reshape(-1)[:6]
                if float(np.max(np.abs(qa - qb))) < 1e-5:
                    q = qa
                else:
                    span = max(1e-9, b["t"] - a["t"])
                    u = (t - a["t"]) / span
                    q = (1 - u) * qa + u * qb
                break
    if robot_sim is not None:
        robot_sim.angle_vector(np.asarray(q, dtype=float).reshape(-1)[:6])
        pos = None
        try:
            pos, _ = _cursor_pose_from_ee(robot_sim, "sim")
        except Exception:
            pos = None
        _note_motion("sim", q, pos, t=float(t))
        _snap_sim_cursor_to_ee()


def _ee_lerp_samples(recs, t):
    if not recs:
        return None
    if t <= float(recs[0].get("t") or 0):
        p = recs[0].get("pos")
        return None if p is None else np.asarray(p, dtype=float).reshape(3)
    if t >= float(recs[-1].get("t") or 0):
        p = recs[-1].get("pos")
        return None if p is None else np.asarray(p, dtype=float).reshape(3)
    for a, b in zip(recs, recs[1:]):
        t0 = float(a.get("t") or 0)
        t1 = float(b.get("t") or 0)
        if t0 <= t <= t1:
            span = max(1e-9, t1 - t0)
            u = (t - t0) / span
            pa = np.asarray(a.get("pos") or [0, 0, 0], dtype=float).reshape(3)
            pb = np.asarray(b.get("pos") or [0, 0, 0], dtype=float).reshape(3)
            return (1.0 - u) * pa + u * pb
    return None


def real_path_status():
    # Polled by /api/clock during play; a failure here would leave the page
    # stuck in run mode, so never raise.
    try:
        at = real_at_start()
    except Exception:
        at = False
    try:
        start_q = _first_path_q()
    except Exception:
        start_q = None
    return {
        "active": bool(_rp["active"]),
        "paused": bool(_rp["paused"]),
        "armed": bool(_rp.get("armed")),
        "uploading": bool(_rp.get("uploading")),
        "uploaded": bool(_rp.get("uploaded")),
        "phase": _rp.get("phase") or "",
        "t": float(_rp["t"]),
        "t_end": float(_rp["t_end"]),
        "at_start": bool(at),
        "locked": edits_locked(),
        "start_q": start_q,
        "speed": HOLD_MOVE_SPEED,
        "connected": robot_connected(),
        "ik_pref": IK_PREF,
        "ik_ok": ik_allowed(),
    }


BURNER_DURATION = 2.0
_burn_pts = []
_burn_line = None


def _afterburner_push(ee):
    now = time.time()
    _burn_pts.append((now, float(ee[0]), float(ee[1]), float(ee[2])))
    cut = now - max(0.05, float(BURNER_DURATION))
    while _burn_pts and _burn_pts[0][0] < cut:
        _burn_pts.pop(0)


def tick_afterburner():
    global _burn_line
    if wp.server is None:
        return
    pts = [p[1:4] for p in _burn_pts]
    if len(pts) < 2:
        if _burn_line is not None:
            try:
                _burn_line.visible = False
            except Exception:
                pass
        return
    arr = np.asarray(pts, dtype=float)
    try:
        if _burn_line is None:
            _burn_line = wp.server.scene.add_line_segments(
                name="/afterburner%s" % arms.ns(),
                points=arr.reshape(-1, 2, 3) if False else np.stack([arr[:-1], arr[1:]], axis=1),
                colors=(180, 80, 180),
            )
        else:
            segs = np.stack([arr[:-1], arr[1:]], axis=1)
            _burn_line.points = segs
            _burn_line.visible = True
    except Exception:
        try:
            _burn_line = wp.server.scene.add_spline_catmull_rom(
                name="/afterburner%s" % arms.ns(), positions=arr, color=(180, 80, 180), line_width=3.0,
            )
        except Exception:
            pass


_motion = {
    "sim": {"q": None, "t": None, "ee": None, "qdot": [0.0] * 6, "lin_mps": 0.0},
    "real": {"q": None, "t": None, "ee": None, "qdot": [0.0] * 6, "lin_mps": 0.0},
}
_speed_log = {"sim": [], "real": []}
_SPEED_LOG_MAX = 16000
_speed_lock = __import__("threading").Lock()


def clear_speed_log():
    with _speed_lock:
        _speed_log["sim"] = []
        _speed_log["real"] = []


def speed_series():
    with _speed_lock:
        return {
            "sim": list(_speed_log["sim"]),
            "real": list(_speed_log["real"]),
        }


def _play_t():
    if _rp.get("active"):
        try:
            return float(_rp.get("t") or 0.0)
        except Exception:
            pass
    if _pb.get("active"):
        try:
            return float(_pb.get("t") or 0.0)
        except Exception:
            pass
    return None


def _note_motion(which, q, ee, t=None):
    st = _motion[which]
    now = time.perf_counter() if t is None else float(t)
    q = _flat(q, 6)
    try:
        ee = np.asarray(ee, dtype=float).reshape(3)
    except Exception:
        ee = None
    prev_q, prev_t, prev_ee = st["q"], st["t"], st["ee"]
    if q is not None and prev_q is not None and prev_t is not None:
        dt = now - prev_t
        if 1e-4 < dt < 0.35:
            st["qdot"] = [(q[i] - prev_q[i]) / dt for i in range(6)]
            if ee is not None and prev_ee is not None:
                st["lin_mps"] = float(np.linalg.norm(ee - prev_ee) / dt)
    if q is not None:
        st["q"] = list(q)
    st["t"] = now
    if ee is not None:
        st["ee"] = ee
    if which == "real" and ee is not None:
        _afterburner_push(ee)
    pt = _play_t()
    if pt is not None:
        with _speed_lock:
            arr = _speed_log[which]
            v = float(st.get("lin_mps") or 0.0)
            if arr and abs(float(arr[-1]["t"]) - float(pt)) < 0.03:
                arr[-1] = {"t": float(pt), "v": v}
            else:
                arr.append({"t": float(pt), "v": v})
            if len(arr) > _SPEED_LOG_MAX:
                del arr[: len(arr) - _SPEED_LOG_MAX]
    return st


def get_selected_props():
    import path_bake as bake
    out = {
        "kind": None,
        "name": "",
        "index": None,
        "ik": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "q": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    }
    kind = getattr(wp, "selection_kind", None)
    sel = list(getattr(wp, "_selection", []) or [])
    if sel and sel[-1][0] == "sim":
        kind = "sim"
    if sel and sel[-1][0] == "bot" and hasattr(wp, "sim_bot_props"):
        return wp.sim_bot_props()
    real_pick = bool(sel and sel[-1][0] == "real")
    if kind == "object" and hasattr(wp, "selected_object_props") and not real_pick:
        prim = wp.selected_object_props()
        if prim:
            return prim
    idx = None if real_pick else getattr(wp, "selected_idx", None)
    if kind != "sim" and idx is not None and 0 <= idx < len(wp.waypoints):
        w = wp.waypoints[idx]
        ik = bake._ik_from_pos_wxyz(w.get("pos"), w.get("wxyz"))
        q = w.get("q")
        if q is None and robot_sim is not None:
            q = _flat(robot_sim.angle_vector(), 6)
        out.update({
            "kind": "waypoint",
            "name": "Waypoint #%d" % idx,
            "editable": True,
            "index": int(idx),
            "ik": ik or [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "q": [float(v) for v in np.asarray(q if q is not None else [0] * 6).reshape(-1)[:6]],
            "track": dict(w.get("track_object") or {}),
        })
        tr = out["track"]
        if tr.get("on") and tr.get("name"):
            tgt = _track_target_pos(tr.get("name"))
            tr["valid"] = tgt is not None
            tr["pos"] = None if tgt is None else [float(tgt[0]), float(tgt[1]), float(tgt[2])]
            tr["added"] = True
        # A waypoint keeps the sim cursor's Offset as it was when the point was added.
        tr["offset_state"] = _offset_state(getattr(wp, "sim_offset", None))
        return out
    if not real_pick and (kind == "sim" or ((kind is None) and (_sim_gizmo is not None or _sim_selected) and not getattr(wp, "selected_obj_id", None))):
        pos, wxyz = get_sim_cursor_pose()
        ee, wxyz = _ee_from_cursor(pos, wxyz, "sim")
        ik = aubo_pose_from_viewer(ee, wxyz)
        q = _flat(robot_sim.angle_vector(), 6) if robot_sim is not None else [0.0] * 6
        out.update({
            "kind": "sim",
            "id": "sim",
            "name": getattr(wp, "SIM_CURSOR_NAME", "Sim cursor"),
            "editable": True,
            "ik": [float(v) for v in ik],
            "q": [float(v) for v in q],
            "lin_mps": float(_motion["sim"]["lin_mps"]),
            "qdot": [float(v) for v in _motion["sim"]["qdot"]],
            "track": track_status(),
            "camera": dict(getattr(wp, "sim_camera", None) or {}) if getattr(wp, "sim_camera", None) else None,
            "offset": wp.offsets.offset_status("sim") if hasattr(wp, "offsets") else {"added": False},
            "collider": wp.colliders.collider_status("sim") if hasattr(wp, "colliders") else {"added": False},
            "comp_order": list(getattr(wp, "sim_comp_order", None) or []),
        })
        return out
    if hasattr(wp, "selected_object_props") and not real_pick:
        prim = wp.selected_object_props()
        if prim:
            return prim
    sel_sm = None if real_pick else bake.get_selected_sample() if hasattr(bake, "get_selected_sample") else None
    if sel_sm is not None:
        seg_i, k, sm = sel_sm
        ik = bake._ik_from_pos_wxyz(sm.get("pos"), sm.get("wxyz"))
        q = sm.get("q")
        out.update({
            "kind": "sample",
            "name": "Sample s%d.%d" % (seg_i, k),
            "editable": False,
            "index": int(k),
            "seg": int(seg_i),
            "ik": ik or [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "q": [float(v) for v in np.asarray(q if q is not None else [0] * 6).reshape(-1)[:6]],
        })
        return out
    model = robot_actual if robot_actual is not None else robot_sim
    if model is not None:
        try:
            q = _flat(model.angle_vector(), 6)
            which = "real" if model is robot_actual else "sim"
            pos, wxyz = _ee_frame(model)
            ik = aubo_pose_from_viewer(pos, wxyz)
        except Exception:
            q = [0.0] * 6
            ik = [0.0] * 6
        out.update({
            "kind": "real",
            "name": "Real cursor" if real_pick else "Actual robot",
            "editable": False,
            "ik": [float(v) for v in ik],
            "q": [float(v) for v in q],
            "lin_mps": float(_motion["real"]["lin_mps"]),
            "qdot": [float(v) for v in _motion["real"]["qdot"]],
        })
        return out
    return out


def apply_selected_ik(ik):
    import path_bake as bake
    if getattr(bake, "selected_sample", None) is not None:
        return False
    ik = [float(v) for v in list(ik)[:6]]
    pos, wxyz = bake._pos_wxyz_from_ik(ik)
    if pos is None:
        return False
    q = None
    try:
        ee_pos, ee_wxyz = _ee_from_cursor(pos, wxyz, "sim")
        q = _ik_to_pose(ee_pos, ee_wxyz)
    except Exception:
        q = None
    idx = getattr(wp, "selected_idx", None)
    if idx is not None and 0 <= idx < len(wp.waypoints):
        wp.move_waypoint(idx, pos, wxyz=wxyz, q=q)
        if hasattr(wp, "draw_path"):
            wp.draw_path()
        return True
    if robot_sim is None:
        return False
    _set_sim_heading(wxyz)
    if _cursor_sim is not None:
        _cursor_sim.position = tuple(float(v) for v in pos)
    if _sim_gizmo is not None:
        try:
            _sim_gizmo.position = tuple(float(v) for v in pos)
        except Exception:
            pass
    _ik_sim_cursor()
    _draw_sim_dir(pos)
    return True


def apply_selected_q(q):
    q = np.asarray(q, dtype=float).reshape(-1)[:6]
    if robot_sim is None:
        return False
    robot_sim.angle_vector(q)
    pos, wxyz = _cursor_pose_from_ee(robot_sim, "sim")
    idx = getattr(wp, "selected_idx", None)
    if idx is not None and 0 <= idx < len(wp.waypoints):
        wp.move_waypoint(idx, pos, wxyz=wxyz, q=q)
        if hasattr(wp, "draw_path"):
            wp.draw_path()
        return True
    _set_sim_heading(wxyz)
    if _cursor_sim is not None:
        _cursor_sim.position = tuple(float(v) for v in pos)
    if _sim_gizmo is not None:
        try:
            _sim_gizmo.position = tuple(float(v) for v in pos)
        except Exception:
            pass
    _draw_sim_dir(pos)
    return True


def trigger_simulation(path, duration=2.0, delay_after=1.5):
    return enter_simulation_mode()
    global sim_mode, sim_path, sim_path_duration, return_delay, _sim_ik_pending, _hold_sim
    _sim_ik_pending = False
    _hold_sim = 0
    hide_sim_gizmo()
    with mode_lock:
        sim_path = [np.asarray(q, dtype=float).reshape(-1)[:6] for q in path]
        sim_path_duration = max(0.4, float(duration))
        return_delay = float(delay_after)
        sim_mode = SimMode.PLAY_PATH


def start_threads():
    global _t_real, _t_sim
    stop_flag.clear()
    _t_real = threading.Thread(target=real_robot_updater, daemon=True)
    _t_sim = threading.Thread(target=sim_robot_controller, daemon=True)
    _t_real.start()
    _t_sim.start()


def shutdown():
    stop_flag.set()