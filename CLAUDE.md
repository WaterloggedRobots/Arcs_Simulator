# CLAUDE.md

Notes for working on this repo. The README is the user manual (features, controls, limitations, project layout); read the relevant section before changing a feature, and update it in the same change.

## What this is

"Sandbox" / AUBO ARCS Simulator: a Flask web UI (`backend/webui.py`, page `backend/templates/index.html`) around a Viser 3D viewer (`backend/robot_sim/`) for planning, baking and running paths on AUBO i-series arms over the ARCS RPC SDK (`pyaubo_sdk`, wrapped by `backend/robot_sim/arcscontrol.py`). Target platform is **Windows** (the SDK, Steam Deck Tools, XInput, Edge); Python 3.12 (pyaubo-sdk has no wheels past 3.12).

- `python backend/webui.py`: full sandbox on http://127.0.0.1:5000, Viser on 8080 (iframe).
- `python backend/deck_webui.py`: Steam Deck edition, same backend, `templates/deck.html`. Never run both at once (same ports).
- `python backend/deck_app.py`: Deck edition as a desktop app (preloader + pywebview window); the entry point of ArcsDeck.exe.

## Layout and paths

- Every file the app reads or writes resolves through `backend/app_paths.py`. `ARCS_USER_DIR` / `ARCS_STATE_DIR` relocate `user_files/` and `backend/state/` (the .exe points them into its unpack folder).
- Modules in `backend/robot_sim/` and `backend/cameras/` are imported by bare name (`import waypoints as wp`, `import robot_runtime as rr`); `webui.py` puts those folders on `sys.path`. `webui.py` registers itself as `webui` when run as `__main__`, so `import webui` reuses the running module.
- `robot_runtime.py` finds URDFs under `<its folder>/aubo-i5-full-master/aubo_description/urdf` (`aubo_i*.urdf`, with an optional `<model>_sim.urdf`).
- `user_files/settings.json` holds settings plus the global saved points and paths. `user_files/paths/` and `user_files/exports/` are git-ignored.

## Steam Deck edition

- `backend/deck_webui.py`: routes under `/api/deck/...`, jog logic, `boot()` (settings, viewer, Steam profile, pad, hold watchdog) and `serve()` (Flask, blocking).
- `backend/deck/pad.py`: Deck controls as an XInput pad. `backend/deck/steam_profile.py`: switches Steam Deck Tools' `SteamController.exe` to the hidden "Arcs Deck" profile and back on exit.
- **Version**: `DECK_VERSION` in `backend/deck/version.py` is the one source: `deck_webui.DECK_VERSION`, `deck/pad.VERSION`, Settings, the start-up print and the preloader all use it (also update the `# version:` first line of `deck_webui.py`). Bump it for a Deck feature change; commits are titled `Deck 1.x: ...`. `deck/preloader.png` has no version drawn in: `deck_app._draw_version` writes `version.label()` ("v deck.1.10") bottom right of the white strip, at the log font size on its baseline, for both splash stages.
- Nothing is autosaved in the Deck edition.
- 3D view controls (deck-1.10): `#touchView` in `deck.html` is a transparent layer over the visible part of the Viser iframe (the iframe is cross-origin, port 8080, so its events can't be read). Touch pointers on it orbit the camera (`/api/deck/view {orbit}` -> `deck_webui.orbit_view`, around the active arm's base); a mouse pointer (Deck trackpad) makes it `pointer-events: none` until the pointer shows up elsewhere on the page. Zoom slider and D-pad up / down (`pad.BUTTONS` DPAD_UP / DPAD_DOWN, 10 % steps) -> `zoom_view` (camera distance; 100 % = `rr.CAM_DIST`). Camera writes keep their own pose for `VIEW_KEEP_S` because the browser reports the camera back late.

## ArcsDeck.exe (packaging/deck)

- PyInstaller one-file build: `packaging/deck/ArcsDeck.spec`, deps in `packaging/deck/requirements-build.txt`, Windows script `packaging/deck/build_deck.bat` (venv `.venv-deck-build`, output `dist/ArcsDeck.exe`). CI: `.github/workflows/build-deck.yml` (windows-latest, Python 3.12, on push to `steamdeck/**` or by hand; the exe is the `ArcsDeck` artifact).
- Startup has two splash stages that must line up: PyInstaller's `Splash` (shown while the exe unpacks; one line, the file being unpacked) and then `deck_app.Preloader` (Tk on its own thread, last `LOG_LINES` (3) log lines). The splash is `SPLASH_WIDTH` = 504 px wide (60 % of the earlier 840); the log font is given in pixels (negative Tk size) so it fits the white strip. Both use `deck_app.SPLASH_WIDTH`, `splash_image()` and `splash_text_pos()`; the spec imports `deck_app.py` for them, so keep its top level free of heavy imports. The preloader stays up at least `MIN_SPLASH_S` (4 s) and until the page loads.
- The transparent top of `preloader.png` is made see-through on Windows with a magenta key colour (alpha is thresholded, no soft edges). PyInstaller writes the splash font family into Tcl unquoted, so the spec passes it in braces (`{Segoe UI}`).
- The exe is windowed: stdout/stderr go to the preloader (and to a console with `--console`). The app window is pywebview with `gui="edgechromium"` (WebView2); if that fails it falls back to `deck_webui._open_deck_ui` (Edge/Chrome `--app` or the browser), and the process then runs until ended. Closing the pywebview window restores the Steam profile and exits (`os._exit`).
- Bundled data: `templates/`, `deck/preloader.png`, `user_files/settings.json`, `aubo-i5-full-master/{aubo_description,robotiq_85_gripper}`, plus `collect_data_files` for viser (web client) and collada (schema for .DAE meshes) and `collect_all` for skrobot, trimesh, pyaubo_sdk, webview. `cv2`, `pandas`, `usb`, `pygrabber` are excluded (camera / bake-report features the Deck page lacks); code that lazily imports them must cope with the ImportError.
- Testing the build without Windows: on Linux the spec builds a Linux binary (install `python3.12` + `python3-tk`, the build requirements minus pyaubo-sdk, and put a stub `pyaubo_sdk.py` with an `RpcClient` class on `PYTHONPATH`). Run it under `Xvfb` and screenshot with `PIL.ImageGrab.grab(xdisplay=":99")`. pywebview has no GUI backend there, so it uses the browser fallback; ports 5000 and 8080 should answer.

## Conventions

- Commit titles: short, sentence-style, describing the behaviour change (e.g. `Deck 1.7: real jog accelerates and decelerates (Real ramp time setting)`). British spelling in UI text and docs ("colour", "centre").
- There is no automated test suite; check changes by running the app (or the steps above for the exe).
- Do not commit `user_files/paths/`, `user_files/exports/`, `build/`, `dist/` or `.venv-deck-build/`.
