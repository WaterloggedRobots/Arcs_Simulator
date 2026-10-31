# PyInstaller spec for ArcsDeck.exe: the Steam Deck edition as one file.
#
#   pyinstaller --noconfirm packaging/deck/ArcsDeck.spec     (from the repo root)
#
# or just run packaging/deck/build_deck.bat. Output: dist/ArcsDeck.exe.
#
# Everything the Deck edition needs (Python, packages, URDFs and meshes, the
# page template, the preloader, default settings) is inside the .exe. While it
# runs it unpacks to a temporary _MEI folder that is removed on exit.
import importlib.util
import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_all, collect_data_files

ROOT = Path(SPECPATH).resolve().parents[1]
BACKEND = ROOT / "backend"
WORK = Path(workpath)
WORK.mkdir(parents=True, exist_ok=True)

# The launcher's preloader layout, so the unpack splash matches it exactly.
_spec = importlib.util.spec_from_file_location("deck_app_layout", BACKEND / "deck_app.py")
deck_app = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(deck_app)

splash_img = deck_app.splash_image(deck_app.SPLASH_WIDTH)
w, h = splash_img.size
splash_png = WORK / "preloader_splash.png"
splash_img.save(splash_png)

datas = [
    (str(BACKEND / "templates"), "templates"),
    (str(BACKEND / "deck" / "preloader.png"), "deck"),
    (str(ROOT / "user_files" / "settings.json"), "user_files"),
    # robot_runtime looks for <its folder>/aubo-i5-full-master/aubo_description/urdf
    (str(BACKEND / "robot_sim" / "aubo-i5-full-master" / "aubo_description"),
     "aubo-i5-full-master/aubo_description"),
    (str(BACKEND / "robot_sim" / "aubo-i5-full-master" / "robotiq_85_gripper"),
     "aubo-i5-full-master/robotiq_85_gripper"),
]
binaries = []
hiddenimports = ["deck", "deck.pad", "deck.steam_profile", "arcscontrol",
                 "PIL.ImageTk", "PIL._tkinter_finder"]  # preloader

datas += collect_data_files("viser")  # web client build, icons
datas += collect_data_files("collada")  # schema used to load .DAE meshes
for pkg in ("skrobot", "trimesh", "pyaubo_sdk", "webview"):
    try:
        d, b, h_ = collect_all(pkg)
    except Exception as e:
        print("collect_all(%s) skipped: %s" % (pkg, e))
        continue
    datas += d
    binaries += b
    hiddenimports += h_

# Features the Deck page does not have (cameras, bake reports, file dialogs
# are lazy imports): keep them out of the .exe.
excludes = ["cv2", "pandas", "pygrabber", "usb", "matplotlib", "IPython",
            "jupyter", "notebook", "pytest"]

a = Analysis(
    [str(BACKEND / "deck_app.py")],
    pathex=[str(BACKEND), str(BACKEND / "robot_sim"), str(BACKEND / "cameras")],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    excludes=excludes,
    noarchive=False,
)
pyz = PYZ(a.pure)

# Shown by the .exe while it unpacks, with the unpack log in the same spot as
# the launcher's log; the launcher's own preloader then takes over.
splash = Splash(
    str(splash_png),
    binaries=a.binaries,
    datas=a.datas,
    text_pos=deck_app.splash_text_pos(w, h),
    text_size=deck_app.LOG_FONT[1],
    # Braced: the splash script puts the family into Tcl unquoted, and "Segoe UI"
    # has a space.
    text_font="{%s}" % deck_app.LOG_FONT[0],
    text_color="black",
    max_img_size=(w, h),
    always_on_top=True,
)

exe = EXE(
    pyz,
    a.scripts,
    splash,
    splash.binaries,
    a.binaries,
    a.datas,
    [],
    name="ArcsDeck",
    debug=False,
    strip=False,
    upx=False,
    runtime_tmpdir=None,
    console=False,
    icon=None,
)
