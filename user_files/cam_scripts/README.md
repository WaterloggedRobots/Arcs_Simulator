# Camera scripts

OpenCV / RealSense scripts you attach to a Camera object from the GUI are copied here.
Examples to start from are in `docs/examples/cam_scripts/`.

Scripts can set `primary_target = [x, y, z]` and `targets = [[x, y, z], ...]` (pixel x, pixel y, depth in metres by default; see "Script targets" in the main README). They show in the 3D view and in the Tracked Points list.

Clicking the camera pane calls the script's `set_target(u, v)`; dragging a rectangle calls `set_box(x0, y0, x1, y1)` (image pixels). `docs/examples/cam_scripts/orbbec_box_track.py` tracks the boxed object with an Orbbec depth camera and publishes it as the primary target.
