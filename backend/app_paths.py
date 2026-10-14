"""Folder layout. Every file the app reads or writes resolves through here,
relative to this file, so the app works from any cwd.

    backend/               everything needed to run the app
      webui.py, robot_sim/, templates/, cameras/ (sony_liveview.py, libusb)
      state/temp.json      autosave (internal, not meant for hand editing)
    user_files/            ships with the app, editable by the user
      settings.json
      cam_scripts/         OpenCV / RealSense scripts added from the GUI
      paths/               saved .arcspath projects
      exports/             export.json, bake_overspeed.csv / .xlsx
      recordings/          camera recordings (.mp4)
      imported_meshes/     meshes copied in by Import mesh
    docs/                  not needed at runtime (docs, examples, archive)

A packaged (e.g. Electron) build can point the two writable folders
elsewhere with ARCS_USER_DIR and ARCS_STATE_DIR.
"""
import os
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent
ROOT_DIR = BACKEND_DIR.parent
SIM_DIR = BACKEND_DIR / "robot_sim"
CAMERAS_DIR = BACKEND_DIR / "cameras"

STATE_DIR = Path(os.environ.get("ARCS_STATE_DIR") or BACKEND_DIR / "state")

USER_DIR = Path(os.environ.get("ARCS_USER_DIR") or ROOT_DIR / "user_files")
SETTINGS_PATH = USER_DIR / "settings.json"
CAM_SCRIPTS_DIR = USER_DIR / "cam_scripts"
PATHS_DIR = USER_DIR / "paths"
EXPORTS_DIR = USER_DIR / "exports"
RECORDINGS_DIR = USER_DIR / "recordings"
MESHES_DIR = USER_DIR / "imported_meshes"

EXAMPLE_CAM_SCRIPTS_DIR = ROOT_DIR / "docs" / "examples" / "cam_scripts"


def ensure_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def relocate(path, *folders):
    """Return `path` if it exists, else the same file name in the first of
    `folders` that has it.

    Saved projects may hold absolute paths from an older layout
    (e.g. ...\\Scripts\\cam_scripts\\x.py); this finds the moved copy.
    """
    if not path:
        return path
    p = Path(str(path))
    if p.is_file():
        return str(p)
    name = str(path).replace("\\", "/").rsplit("/", 1)[-1]
    for folder in folders:
        cand = Path(folder) / name
        if cand.is_file():
            return str(cand)
    return str(path)


def add_cameras_to_sys_path():
    """Make cameras/sony_liveview.py importable as `import sony_liveview`."""
    s = str(CAMERAS_DIR)
    if CAMERAS_DIR.is_dir() and s not in sys.path:
        sys.path.append(s)
