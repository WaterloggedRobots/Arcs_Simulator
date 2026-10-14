"""Folder layout for the Scripts tree. Every file the app reads or writes
resolves through here, relative to this file, so the app works from any cwd.

    Scripts/
      config/            settings.json, temp.json (autosave)
      user_data/
        paths/           saved .arcspath projects
        exports/         export.json, bake_overspeed.csv / .xlsx
        recordings/      camera recordings (.mp4)
        imported_meshes/ meshes copied in by File > Import mesh
      cameras/           sony_liveview.py, libusb, cam_scripts/, models/
      robot_sim/         simulator / editor code
      templates/         index.html
"""
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
SIM_DIR = SCRIPTS_DIR / "robot_sim"

CONFIG_DIR = SCRIPTS_DIR / "config"
SETTINGS_PATH = CONFIG_DIR / "settings.json"

USER_DATA_DIR = SCRIPTS_DIR / "user_data"
PATHS_DIR = USER_DATA_DIR / "paths"
EXPORTS_DIR = USER_DATA_DIR / "exports"
RECORDINGS_DIR = USER_DATA_DIR / "recordings"
MESHES_DIR = USER_DATA_DIR / "imported_meshes"

CAMERAS_DIR = SCRIPTS_DIR / "cameras"
CAM_SCRIPTS_DIR = CAMERAS_DIR / "cam_scripts"
MODELS_DIR = CAMERAS_DIR / "models"


def ensure_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def relocate(path, folder):
    """Return `path` if it exists, else the same file name inside `folder`.

    Saved projects may hold absolute paths from before the folder reorganize
    (e.g. ...\\Scripts\\cam_scripts\\x.py); this finds the moved copy.
    """
    if not path:
        return path
    p = Path(str(path))
    if p.is_file():
        return str(p)
    name = str(path).replace("\\", "/").rsplit("/", 1)[-1]
    cand = Path(folder) / name
    return str(cand) if cand.is_file() else str(path)


def add_cameras_to_sys_path():
    """Make cameras/sony_liveview.py importable as `import sony_liveview`."""
    s = str(CAMERAS_DIR)
    if CAMERAS_DIR.is_dir() and s not in sys.path:
        sys.path.append(s)
