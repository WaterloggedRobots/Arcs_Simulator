# Arcs_Simulator

Flask + Viser web UI for planning, baking, and running robot paths on Aubo Arcs (sim + live).

## Run

```bat
cd Scripts
python webui.py
```

Then open **http://127.0.0.1:5000** in a browser.

Install deps once: `pip install -r Scripts/requirements.txt`

## Folder map

| Path | What it is |
|------|------------|
| `Scripts/` | App entrypoint (`webui.py`), settings, templates, camera helpers |
| `Scripts/robot_sim/` | Core Python (bake, waypoints, robot runtime) + working `path.arcspath` / `temp.json` |
| `Scripts/paths/` | Sample / saved `.arcspath` projects (use Open in the UI) |
| `Scripts/cam_scripts/` | RealSense / YOLO helpers |
| `Demo/` | Short demo videos |
| `docs/` | Bug list, export notes, function list |
| `archive/` | Large vendor / backup blobs (Sony CRC kit, old 7z, PPT template) |

## Runtime outputs (under `Scripts/`)

- `session_log.txt` — high-level session log  
- `debug_run.log` — detailed `[DBG]` trace  
- `bake_overspeed.csv` — bake speed report (also downloadable from the UI)  
- `settings.json` — UI / connection settings  
- `yolov8n.pt` — YOLO weights for cam scripts  

## Notes

- Save still defaults to `Scripts/robot_sim/path.arcspath`.  
- Open prefers `Scripts/paths/` when that folder exists.  
- Do not move `Scripts/robot_sim/*.py` or rename `Scripts/` without updating imports.
