from skrobot.viewers import ViserViewer
from skrobot.model import RobotModel
import time
import numpy as np


import numpy as np

def SetScene(viewer):
    server = viewer._server

    bg = np.full((8, 8, 3), (33, 34, 36), dtype=np.uint8)
    server.scene.set_background_image(bg)
    server.scene.set_up_direction("+z")

    server.scene.add_box(
        "/floor",
        dimensions=(4.0, 4.0, 0.004),
        position=(0.0, 0.0, -0.002),
        color=(90, 90, 90),
    )
    server.scene.add_grid(
        "/grid",
        width=4.0,
        height=4.0,
        plane="xy",
        cell_size=0.1,
        section_size=1.0,
        position=(0.0, 0.0, 0.001),
        section_color=(60, 60, 60),
    )

    try:
        server.gui.configure_theme(
            dark_mode=True,
            show_logo=False,
            show_share_button=False,
            control_layout="collapsible",
            control_width="small",
        )
    except TypeError:
        server.gui.configure_theme(dark_mode=True, show_logo=False)

    def _strip_gui(gui):
        for attr in ("_children", "_gui_handles", "_handle_from_uuid"):
            store = getattr(gui, attr, None)
            if not store:
                continue
            items = store.values() if hasattr(store, "values") else store
            for h in list(items):
                for op in ("remove",):
                    if hasattr(h, op):
                        try:
                            getattr(h, op)()
                        except Exception:
                            pass
                if hasattr(h, "visible"):
                    try:
                        h.visible = False
                    except Exception:
                        pass

    _strip_gui(server.gui)

    @server.on_client_connect
    def _(client):
        client.camera.near = 0.02
        client.camera.far = 50.0
        try:
            client.gui.visible = False
        except Exception:
            pass
        _strip_gui(getattr(client, "gui", server.gui))

    return server


if __name__ == "__main__":
    robot = RobotModel()
    robot.load_urdf_file(r"C:\Users\User\OneDrive - Roborn Technology Limited\01 Roborn\23 Aubo Arcs Simulator\aubo-i5-full-master\aubo_description\urdf\aubo_i5.urdf")

    viewer = ViserViewer(enable_ik=True, enable_motion_planning=True)
    viewer.add(robot)
    viewer.show()
    setScene(viewer)
    print("Viser running at http://localhost:8080")
    print("Press Ctrl+C to stop.")

    try:
        while viewer.is_active:
            viewer.redraw()
            time.sleep(0.05)
    except KeyboardInterrupt:
        print("Shutting down...")