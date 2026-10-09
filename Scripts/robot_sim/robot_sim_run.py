# version: beta.0904
import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parent
_PARENT = _ROOT.parent
if str(_PARENT) not in sys.path:
    sys.path.insert(0, str(_PARENT))
if str(_ROOT) not in sys.path:
    sys.path.insert(1, str(_ROOT))
from skrobot.viewers import ViserViewer
import time
import numpy as np
import webbrowser
import inspect
import keymap
import scene_background as sceneBG
import waypoints as wp
import robot_runtime as rr
import path_bake as bake
import file_manage as fman
print("robot_runtime", getattr(rr, "__file__", rr), "init", hasattr(rr, "init"))

viewer = None
server = None
VISER_PORT = None
add_requests = []
bake_requests = []
sim_requests = []
delete_requests = []
select_requests = []
playhead_t = 0.0
gizmo_space_requests = []
reset_requests = []
edit_requests = []
file_requests = []
prop_requests = []
move_requests = []
real_requests = []


def _no_browser(*args, **kwargs):
    return None


def _silence_browser_opens():
    webbrowser.open = _no_browser
    webbrowser.open_new = _no_browser
    webbrowser.open_new_tab = _no_browser


def _read_viser_port(v):
    s = getattr(v, "_server", None) or getattr(v, "server", None)
    if s is None:
        return None

    for obj in (s, getattr(s, "_websock_server", None), getattr(s, "_backend", None)):
        if obj is None:
            continue
        for attr in ("port", "_port"):
            if hasattr(obj, attr):
                try:
                    return int(getattr(obj, attr))
                except Exception:
                    pass
        if hasattr(obj, "get_port"):
            try:
                return int(obj.get_port())
            except Exception:
                pass

    for name in ("get_url", "request_url"):
        if not hasattr(s, name):
            continue
        try:
            val = getattr(s, name)
            url = val() if callable(val) else val
            return int(str(url).rsplit(":", 1)[-1].split("/")[0])
        except Exception:
            pass
    return None


def _make_viewer():
    kwargs = {"enable_ik": False, "enable_motion_planning": False}
    try:
        params = inspect.signature(ViserViewer.__init__).parameters
        if "port" in params:
            kwargs["port"] = 8080
        if "host" in params:
            kwargs["host"] = "127.0.0.1"
    except Exception:
        pass
    return ViserViewer(**kwargs)


def _force_collapse_viser_gui(server):
    try:
        server.gui.configure_theme(
            dark_mode=True,
            show_logo=False,
            show_share_button=False,
            control_layout="collapsible",
        )
    except TypeError:
        try:
            server.gui.configure_theme(dark_mode=True, show_logo=False)
        except Exception:
            pass

    def _collapse_one(gui):
        if gui is None:
            return
        if hasattr(gui, "visible"):
            try:
                gui.visible = False
            except Exception:
                pass
        panel = getattr(gui, "main_panel", None)
        if panel is None:
            return
        for name in ("minimize", "collapse", "hide"):
            fn = getattr(panel, name, None)
            if callable(fn):
                try:
                    fn()
                except Exception:
                    pass

    _collapse_one(getattr(server, "gui", None))

    @server.on_client_connect
    def _hide_gui(client):
        _collapse_one(getattr(server, "gui", None))
        _collapse_one(getattr(client, "gui", None))


def InitViewer():
    global viewer, server
    viewer.show()
    rr.init(viewer)
    server = sceneBG.SetScene(viewer)
    wp.init(server)
    rr.init_cursors(server)
    if hasattr(rr, "robot_connected") and not rr.robot_connected():
        rr.set_hide_real_robot(True)


def start_viewer():
    global viewer, server, VISER_PORT, playhead_t
    _silence_browser_opens()

    viewer = _make_viewer()
    InitViewer()

    VISER_PORT = _read_viser_port(viewer) or 8080
    print("Viser port:", VISER_PORT)
    _force_collapse_viser_gui(server)

    def _joints_of(which):
        try:
            model = rr.robot_actual if which == "real" else rr.robot_sim
            if model is None:
                return None
            return list(np.asarray(model.angle_vector(), dtype=float).reshape(-1)[:6])
        except Exception:
            return None

    def _add_from_real(_pose=None, mode="arc", t=None):
        pose = rr.get_real_cursor_pose()
        if t is None:
            t = playhead_t
        wp.add_waypoint(pose, mode="arc", q=_joints_of("real"), t=t)

    def _add_from_sim(_pose=None, mode="arc", t=None):
        pose = rr.get_sim_cursor_pose()
        print("add SIM pose", pose)
        if t is None:
            t = playhead_t
        wp.add_waypoint(pose, mode="arc", q=_joints_of("sim"), t=t)

    def _bake(_spacing=None):
        try:
            import path_bake as pb
            fn = getattr(pb, "run_bake", None) or getattr(pb, "bake", None)
            if not callable(fn):
                print(
                    "bake failed: path_bake.bake is",
                    type(fn),
                    "from",
                    getattr(pb, "__file__", pb),
                )
                return
            result = fn(_spacing)
            print("bake:", result)
        except Exception as e:
            print("bake failed:", e)

    keymap.setup_hotkeys(
        server,
        add_from_real=_add_from_real,
        add_from_sim=_add_from_sim,
        clear_waypoints=wp.clear_waypoints,
        delete_selected=wp.delete_selected,
        undo=wp.undo,
        redo=wp.redo,
        bake_path=_bake,
        reset_camera=lambda: rr.apply_camera_view("front"),
    )

    try:
        while viewer.is_active:
            while add_requests:
                req = add_requests.pop(0)
                try:
                    replace = True
                    if isinstance(req, str):
                        src, t = req, playhead_t
                    else:
                        src, t = req.get("source", "real"), req.get("t", playhead_t)
                        replace = bool(req.get("replace", True))
                    if src == "sim":
                        pose = rr.get_sim_cursor_pose()
                        wp.add_waypoint(pose, mode="arc", q=_joints_of("sim"), t=t if t is not None else playhead_t, replace=replace)
                    else:
                        pose = rr.get_real_cursor_pose()
                        wp.add_waypoint(pose, mode="arc", q=_joints_of("real"), t=t if t is not None else playhead_t, replace=replace)
                    item = wp.waypoints[-1] if wp.waypoints else None
                    if item is not None:
                        item["added_from"] = src
                        item["track_object"] = None
                        if src == "sim" and hasattr(rr, "track_status"):
                            st = rr.track_status() or {}
                            if st.get("on") and st.get("added"):
                                item["track_object"] = {
                                    "on": True,
                                    "added": True,
                                    "name": st.get("name") or "",
                                    "offset": list(st.get("offset") or [0.0, 0.0, 0.0]),
                                }
                except Exception as e:
                    print("add_waypoint failed:", e)
            while delete_requests:
                delete_requests.pop(0)
                try:
                    wp.delete_selected()
                except Exception as e:
                    print("delete_selected failed:", e)
            while select_requests:
                req = select_requests.pop(0)
                try:
                    if isinstance(req, dict):
                        idx, mode = req.get("index"), req.get("mode") or "replace"
                    else:
                        idx, mode = req, "replace"
                    if idx is None:
                        wp.deselect_all()
                    elif 0 <= int(idx) < len(wp.waypoints):
                        wp.select_waypoint(int(idx), mode=mode)
                except Exception as e:
                    print("select_waypoint failed:", e)
            while gizmo_space_requests:
                mode = gizmo_space_requests.pop(0)
                try:
                    wp.apply_gizmo_space(mode)
                    print("gizmo space", mode)
                except Exception as e:
                    print("gizmo space failed:", e)
            while real_requests:
                item = real_requests.pop(0)
                kind = item[0] if isinstance(item, (tuple, list)) else item
                try:
                    if kind == "goto":
                        rr.start_goto_path_start()
                    elif kind == "start":
                        if hasattr(rr, "fire_real_path"):
                            rr.fire_real_path()
                        else:
                            rr.start_real_path()
                    elif kind == "pause":
                        rr.pause_real_path()
                    elif kind == "stop":
                        rr.stop_real_path()
                except Exception as e:
                    print("real path request failed", e)
            while move_requests:
                kind, payload = move_requests.pop(0)
                try:
                    if kind == "start":
                        rr.start_hold_move(
                            ik=payload.get("ik"),
                            q=payload.get("q"),
                            mode=payload.get("mode", "ik"),
                        )
                    elif kind == "stop":
                        rr.stop_hold_move()
                except Exception as e:
                    print("move request failed", e)
            while prop_requests:
                kind, payload = prop_requests.pop(0)
                try:
                    if kind == "ik":
                        rr.apply_selected_ik(payload)
                    elif kind == "q":
                        rr.apply_selected_q(payload)
                except Exception as e:
                    print("prop apply failed", e)
            while file_requests:
                kind, payload = file_requests.pop(0)
                try:
                    if kind == "new":
                        if hasattr(wp, "reset_document"):
                            wp.reset_document()
                        else:
                            wp.clear_waypoints()
                            wp.extra_tracks = []
                            wp.sim_poses = []
                            if hasattr(wp, "ensure_builtin_sim_poses"):
                                wp.ensure_builtin_sim_poses()
                    elif kind == "load":
                        payload = payload or {}
                        if isinstance(payload, dict) and payload.get("doc"):
                            fman.apply_loaded_doc(payload.get("doc"))
                        elif isinstance(payload, dict) and payload.get("path") and not payload.get("waypoints") and not payload.get("points"):
                            fman.load_from_path(payload.get("path"))
                        else:
                            fman.apply_loaded_doc(payload)
                    elif kind == "undo":
                        wp.undo()
                    elif kind == "redo":
                        wp.redo()
                except Exception as e:
                    print("file request failed", kind, e)
            while reset_requests:
                reset_requests.pop(0)
                try:
                    rr.reset_sim_to_real()
                except Exception as e:
                    print("reset sim failed:", e)
            while bake_requests:
                sp = bake_requests.pop(0)
                _bake(sp)
            while sim_requests:
                dur = sim_requests.pop(0)
                try:
                    if not hasattr(bake, "joint_path"):
                        raise RuntimeError(
                            "path_bake.joint_path missing — replace robot_sim/path_bake.py "
                            "with the project copy (loaded %s)" % getattr(bake, "__file__", bake)
                        )
                    qs = bake.joint_path()
                    if len(qs) < 2:
                        print("simulate: bake a path first")
                    else:
                        dt = max(0.5, float(dur) if dur else max(1.0, 0.05 * len(qs)))
                        rr.trigger_simulation(qs, duration=dt, delay_after=0.3)
                        print("simulate:", len(qs), "samples in", dt, "s")
                except Exception as e:
                    print("simulate failed:", e)
            playing = bool(getattr(rr, "_pb", {}).get("active")) and not bool(
                getattr(rr, "_pb", {}).get("paused")
            )
            real_playing = bool(getattr(rr, "_rp", {}).get("active")) and not bool(
                getattr(rr, "_rp", {}).get("paused")
            )
            if not playing and not real_playing and hasattr(wp, "snap_global_gizmos_idle"):
                wp.snap_global_gizmos_idle()
            if not playing and not real_playing and hasattr(wp, "flush_idle_path_rebuild"):
                wp.flush_idle_path_rebuild()
            if hasattr(rr, "tick_simulation"):
                rr.tick_simulation()
            if hasattr(rr, "tick_real_path"):
                rr.tick_real_path()
            if hasattr(rr, "tick_track_object"):
                rr.tick_track_object()
            try:
                t_win = playhead_t
                st = rr.sim_status() if hasattr(rr, "sim_status") else {}
                rt = rr.real_path_status() if hasattr(rr, "real_path_status") else {}
                if st.get("active"):
                    t_win = float(st.get("t") or 0.0)
                elif rt.get("active"):
                    t_win = float(rt.get("t") or 0.0)
                playhead_t = t_win
                if hasattr(wp, "follow_path_window"):
                    wp.follow_path_window(t_win)
            except Exception:
                pass
            try:
                if hasattr(wp, "tick_cameras"):
                    wp.tick_cameras()
            except Exception:
                pass
            if playing:
                try:
                    viewer.redraw()
                except Exception:
                    pass
                time.sleep(0.008)
            else:
                rr.update_cursors()
                viewer.redraw()
                time.sleep(0.03)
    except KeyboardInterrupt:
        print("Shutting down...")
        rr.shutdown()


if __name__ == "__main__":
    start_viewer()
