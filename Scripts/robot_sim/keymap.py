# version: 0.11.0
def setup_hotkeys(
    server,
    *,
    add_from_real,
    add_from_sim,
    clear_waypoints,
    delete_selected,
    undo,
    redo,
    bake_path=None,
    set_mode=None,
    reset_camera=None,
):
    add_real = server.gui.add_command(
        "Waypoint from real cursor",
        description="Add waypoint at the real-robot cursor",
        hotkey="E",
    )

    @add_real.on_trigger
    def _(_):
        add_from_real()

    add_sim = server.gui.add_command(
        "Waypoint from sim cursor",
        description="Add waypoint at the sim cursor",
        hotkey="R",
    )

    @add_sim.on_trigger
    def _(_):
        add_from_sim()

    del_cmd = server.gui.add_command("Delete Selected Waypoint", hotkey="delete")

    @del_cmd.on_trigger
    def _(_):
        delete_selected()

    clear_cmd = server.gui.add_command("Clear All Waypoints", hotkey="C")

    @clear_cmd.on_trigger
    def _(_):
        clear_waypoints()

    undo_cmd = server.gui.add_command("Undo", hotkey="Z", modifier="cmd/ctrl")

    @undo_cmd.on_trigger
    def _(_):
        undo()

    redo_cmd = server.gui.add_command("Redo", hotkey="Z", modifier="cmd/ctrl+shift")

    @redo_cmd.on_trigger
    def _(_):
        redo()

    if bake_path is not None:
        bake_cmd = server.gui.add_command("Bake path", hotkey="B", modifier="cmd/ctrl")

        @bake_cmd.on_trigger
        def _(_):
            bake_path()

    if reset_camera is not None:
        cam_cmd = server.gui.add_command(
            "Reset camera (front)",
            description="Look at the arm from the front",
            hotkey="F",
        )

        @cam_cmd.on_trigger
        def _(_):
            reset_camera()
