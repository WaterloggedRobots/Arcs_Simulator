"""Steam Deck edition as a desktop app: the entry point of ArcsDeck.exe.

    python backend/deck_app.py            # same as the .exe, from source
    ArcsDeck.exe [--console]              # --console opens a log window

It shows the preloader (deck/preloader.png) with the startup log in black on
the left of its white strip, boots the same backend as deck_webui.py, and
opens the Deck page in its own window (pywebview / Edge WebView2). The
preloader stays up for at least MIN_SPLASH_S seconds. Closing the window
restores the Steam Deck profile and quits.

Packaged by packaging/deck/ArcsDeck.spec (PyInstaller, one file). In the .exe
everything, settings included, lives in PyInstaller's temporary unpack folder,
which is deleted on exit: nothing is saved between runs.
"""
from __future__ import annotations

import os
import queue
import socket
import sys
import threading
import time
from pathlib import Path

T0 = time.time()
MIN_SPLASH_S = 4.0
URL = "http://127.0.0.1:5000"
TITLE = "AUBO ARCS Simulator - Deck"

FROZEN = bool(getattr(sys, "frozen", False))
HERE = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
PRELOADER = HERE / "deck" / "preloader.png"

# Preloader layout as fractions of preloader.png (1488 x 1025): the white
# strip starts at y = 885; the log sits left in it (x = 33 lines up with the
# "AUBO" title), clear of "v deck.x" at the bottom right.
SPLASH_WIDTH = 840
LOG_X = 33 / 1488
LOG_BOTTOM = 1012 / 1025
LOG_RIGHT = 1300 / 1488
LOG_LINES = 4
LOG_FONT = ("Segoe UI", 9) if os.name == "nt" else ("DejaVu Sans", 9)


def splash_text_pos(width, height):
    """Bottom-left of the log's last line, in splash pixels (also used by the
    PyInstaller splash shown while the .exe unpacks)."""
    return int(round(LOG_X * width)), int(round(LOG_BOTTOM * height))


def splash_image(width):
    """preloader.png at `width` px, with its alpha cut to fully opaque or fully
    transparent: Windows shows the transparent part (above the angled edge)
    as see-through via a key colour, and a soft edge would fringe."""
    from PIL import Image
    img = Image.open(PRELOADER).convert("RGBA")
    height = int(round(img.height * width / img.width))
    img = img.resize((width, height), Image.LANCZOS)
    img.putalpha(img.getchannel("A").point(lambda a: 255 if a >= 128 else 0))
    return img


if FROZEN:
    # Settings and state go to the unpack folder (deleted on exit), never
    # next to the .exe or in %TEMP%'s parent.
    os.environ.setdefault("ARCS_USER_DIR", str(HERE / "user_files"))
    os.environ.setdefault("ARCS_STATE_DIR", str(HERE / "state"))


# ---------------------------------------------------------------- logging

class _Log:
    """stdout / stderr replacement: every line goes to the preloader and,
    when there is one, the console."""

    def __init__(self, console):
        self.console = console
        self.lines = queue.Queue()
        self._buf = ""
        self._lock = threading.Lock()

    def write(self, text):
        if not text:
            return 0
        if self.console is not None:
            try:
                self.console.write(text)
                self.console.flush()
            except Exception:
                pass
        with self._lock:
            self._buf += text
            *done, self._buf = self._buf.split("\n")
        for line in done:
            line = line.strip("\r").rstrip()
            if line:
                self.lines.put(line)
        return len(text)

    def flush(self):
        pass

    def isatty(self):
        return False


def _console_stream():
    """The console to echo the log to: the terminal when run from source,
    a new console window for `ArcsDeck.exe --console`, else none."""
    if "--console" in sys.argv and os.name == "nt" and FROZEN:
        try:
            import ctypes
            if ctypes.windll.kernel32.AllocConsole():
                return open("CONOUT$", "w", encoding="utf-8", errors="replace")
        except Exception:
            return None
    return sys.__stdout__


# ---------------------------------------------------------------- preloader

class Preloader:
    """The preloader window (Tk, on its own thread so the main thread is free
    for the app window). Shows the last LOG_LINES lines of the log."""

    def __init__(self, log):
        self.log = log
        self.shown = threading.Event()
        self._close = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True, name="preloader")

    def start(self):
        self._thread.start()
        self.shown.wait(5.0)

    def close(self):
        self._close.set()
        self._thread.join(2.0)

    def _run(self):
        try:
            import tkinter as tk
            import tkinter.font as tkfont
            from PIL import ImageTk
        except Exception as e:
            print("preloader unavailable:", e)
            self.shown.set()
            return
        try:
            root = tk.Tk()
            root.withdraw()
            root.overrideredirect(True)
            root.attributes("-topmost", True)
            img = splash_image(min(SPLASH_WIDTH, int(root.winfo_screenwidth() * 0.9)))
            w, h = img.size
            bg = "black"
            if os.name == "nt":
                # Same as PyInstaller's splash: magenta is see-through.
                bg = "magenta"
                root.attributes("-transparentcolor", bg)
            photo = ImageTk.PhotoImage(img)
            canvas = tk.Canvas(root, width=w, height=h, highlightthickness=0, bd=0, bg=bg)
            canvas.pack()
            canvas.create_image(0, 0, image=photo, anchor="nw")
            font = tkfont.Font(family=LOG_FONT[0], size=LOG_FONT[1])
            x, y = splash_text_pos(w, h)
            step = font.metrics("linespace")
            max_px = int(LOG_RIGHT * w) - x
            items = [canvas.create_text(x, y - (LOG_LINES - 1 - i) * step, text="", anchor="sw",
                                        fill="black", font=font) for i in range(LOG_LINES)]
            shown = []

            def fit(line):
                if font.measure(line) <= max_px:
                    return line
                while line and font.measure(line + "...") > max_px:
                    line = line[:-1]
                return line + "..."

            def tick():
                if self._close.is_set():
                    root.destroy()
                    return
                changed = False
                while True:
                    try:
                        shown.append(fit(self.log.lines.get_nowait()))
                        changed = True
                    except queue.Empty:
                        break
                if changed:
                    del shown[:-LOG_LINES]
                    pad = [""] * (LOG_LINES - len(shown)) + shown
                    for item, line in zip(items, pad):
                        canvas.itemconfigure(item, text=line)
                root.after(50, tick)

            sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
            root.geometry("%dx%d+%d+%d" % (w, h, (sw - w) // 2, (sh - h) // 2))
            root.deiconify()
            root.update()
            self.shown.set()
            root.after(50, tick)
            root.mainloop()
        except Exception as e:
            print("preloader failed:", e)
            self.shown.set()


def _close_pyi_splash():
    """Close the splash PyInstaller shows while the .exe unpacks."""
    try:
        import pyi_splash
        pyi_splash.close()
    except Exception:
        pass


# ---------------------------------------------------------------- app

def _wait_port(port, timeout=30.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.5):
                return True
        except OSError:
            time.sleep(0.1)
    return False


def _quit(code=0):
    try:
        from deck import steam_profile
        steam_profile.restore()
    except Exception:
        pass
    os._exit(code)


def main():
    log = _Log(_console_stream())
    sys.stdout = sys.stderr = log
    pre = Preloader(log)
    pre.start()
    _close_pyi_splash()

    print("Arcs Deck starting%s" % (" (packaged)" if FROZEN else ""))
    print("Loading modules...")
    try:
        import deck_webui
        deck_webui.boot()
        threading.Thread(target=deck_webui.serve, daemon=True, name="flask").start()
        if not _wait_port(5000):
            print("Deck UI did not start on port 5000")
    except Exception:
        import traceback
        traceback.print_exc()
        time.sleep(8)  # leave the error on the preloader
        pre.close()
        _quit(1)

    print("Opening window...")
    wait = MIN_SPLASH_S - (time.time() - T0)
    if wait > 0:
        time.sleep(wait)

    try:
        _app_window(pre)
    except Exception as e:
        print("app window unavailable (%s); opening Edge / browser" % e)
    # No app window: Edge / Chrome --app window or the default browser; the
    # app then runs until the process is ended.
    pre.close()
    deck_webui._open_deck_ui(URL)
    threading.Event().wait()


def _app_window(pre):
    """Show the Deck page in its own window; returns only on failure. Closing
    the window quits the app."""
    import webview
    win = webview.create_window(TITLE, URL, maximized=True, background_color="#FFFFFF")
    done = threading.Event()

    def ready():
        if not done.is_set():
            done.set()
            pre.close()

    win.events.loaded += ready
    # In case the page never reports loaded, do not cover the window forever.
    threading.Timer(15.0, ready).start()
    # Windows: Edge WebView2 only (the old IE engine cannot run the 3D viewer).
    webview.start(gui="edgechromium" if os.name == "nt" else None)
    _quit(0)

if __name__ == "__main__":
    main()
