"""Second page UI components for PLAiCE.

This module provides a small slideshow player that displays all images
found in a hard-coded folder (`assets/images` under the repo root) and
plays them sequentially.
"""
from pathlib import Path
from typing import Callable
import tkinter as tk
import itertools
import os


# Hard-coded images folder (project root / assets / images)
IMAGE_DIR = Path(__file__).resolve().parents[2] / "frames"
# milliseconds between frames (default)
FRAME_DELAY_MS = 500
# bounds for delay when speeding up / slowing down
MIN_DELAY_MS = 10
MAX_DELAY_MS = 1000


def _load_image(path: Path, max_w: int, max_h: int):
    """Try to load an image and return a Tk-compatible PhotoImage.

    Prefer Pillow (ImageTk) for broad format support, but fall back to
    tkinter.PhotoImage where possible (PNG/GIF).
    Returns None on failure.
    """
    try:
        from PIL import Image, ImageTk
    except Exception:
        Image = None
        ImageTk = None

    try:
        if Image and ImageTk:
            im = Image.open(path)
            if max_w > 0 and max_h > 0:
                im.thumbnail((max_w, max_h), Image.LANCZOS)
            photo = ImageTk.PhotoImage(im)

            # try to extract a textual description from EXIF or info
            desc = None
            try:
                info = im.info or {}
                # common info fields
                for key in ("Description", "description", "comment", "Comment"):
                    if key in info:
                        desc = info.get(key)
                        break

                if not desc:
                    # EXIF ImageDescription tag is 270
                    try:
                        exif = im.getexif()
                        if exif:
                            val = exif.get(270)
                            if val:
                                desc = val
                    except Exception:
                        pass

                if isinstance(desc, bytes):
                    try:
                        desc = desc.decode("utf-8", errors="ignore")
                    except Exception:
                        desc = str(desc)
            except Exception:
                desc = None

            return (photo, desc)
        else:
            # tkinter.PhotoImage supports PNG/GIF on most builds
            return (tk.PhotoImage(file=str(path)), None)
    except Exception:
        return None


def create_second_page(master: tk.Misc, on_back: Callable[[], None]) -> tk.Frame:
    """Create and return the second page frame with a simple slideshow.

    Parameters
    - master: parent widget
    - on_back: callback when Back button is pressed
    """
    frame = tk.Frame(master, padx=10, pady=10)

    title = tk.Label(frame, text="Second page", font=(None, 14))
    title.pack(pady=(0, 10))

    # area where images will be shown
    img_holder = tk.Label(frame)
    img_holder.pack(expand=True)

    ctrl_frame = tk.Frame(frame)
    ctrl_frame.pack(pady=(8, 0))

    back_btn = tk.Button(ctrl_frame, text="Back", command=on_back)
    back_btn.pack(side=tk.LEFT)

    play_pause_var = tk.StringVar(value="Pause")

    def toggle_play():
        if getattr(frame, "_playing", True):
            frame._playing = False
            play_pause_var.set("Play")
            if getattr(frame, "_after_id", None):
                frame.after_cancel(frame._after_id)
                frame._after_id = None
        else:
            frame._playing = True
            play_pause_var.set("Pause")
            _schedule_next()

    play_pause_btn = tk.Button(ctrl_frame, textvariable=play_pause_var, command=toggle_play)
    play_pause_btn.pack(side=tk.LEFT, padx=(8, 0))

    # speed controls
    def _apply_new_delay():
        # if playing, reschedule next with new delay
        if getattr(frame, "_after_id", None):
            try:
                frame.after_cancel(frame._after_id)
            except Exception:
                pass
            frame._after_id = frame.after(frame._delay_ms, _schedule_next)

    def speed_up():
        # make it faster -> decrease delay
        frame._delay_ms = max(MIN_DELAY_MS, int(frame._delay_ms * 0.7))
        _apply_new_delay()

    def slow_down():
        # make it slower -> increase delay
        frame._delay_ms = min(MAX_DELAY_MS, int(frame._delay_ms * 1.5))
        _apply_new_delay()

    speed_up_btn = tk.Button(ctrl_frame, text="Speed +", command=speed_up)
    speed_up_btn.pack(side=tk.LEFT, padx=(8, 0))

    speed_down_btn = tk.Button(ctrl_frame, text="Speed -", command=slow_down)
    speed_down_btn.pack(side=tk.LEFT, padx=(4, 0))

    # fit images to screen with a small margin
    screen_w = frame.winfo_screenwidth()
    screen_h = frame.winfo_screenheight()
    max_w = max(0, screen_w - 80)
    max_h = max(0, screen_h - 220)

    # load image paths
    paths = []
    if IMAGE_DIR.exists() and IMAGE_DIR.is_dir():
        for fn in sorted(os.listdir(IMAGE_DIR)):
            p = IMAGE_DIR / fn
            if p.is_file() and fn.lower().endswith((".png", ".jpg", ".jpeg", ".gif", ".bmp")):
                paths.append(p)

    images = []
    for p in paths:
        res = _load_image(p, max_w, max_h)
        if res is not None:
            images.append(res)  # (photo, description)

    if not images:
        note = tk.Label(frame, text=f"No images found in {IMAGE_DIR}")
        note.pack(pady=(10, 0))
        frame._playing = False
        frame._after_id = None
        frame._images = []
        frame._idx = 0
        return frame

    # keep references on the frame to avoid GC
    # images: list[tuple[PhotoImage, Optional[str]]]
    frame._images = images
    frame._idx = 0
    frame._playing = True
    frame._after_id = None
    # current delay (ms) between frames
    frame._delay_ms = FRAME_DELAY_MS

    # description label
    desc_label = tk.Label(frame, text="", wraplength=max(200, max_w), justify=tk.CENTER)
    desc_label.pack(pady=(6, 0))

    def _show_index(i: int):
        photo, desc = frame._images[i]
        img_holder.configure(image=photo)
        img_holder.image = photo
        desc_label.configure(text=desc or "")

    def _schedule_next():
        if not getattr(frame, "_playing", False):
            return
        frame._idx = (frame._idx + 1) % len(frame._images)
        _show_index(frame._idx)
        frame._after_id = frame.after(frame._delay_ms, _schedule_next)

    # show first image immediately
    _show_index(0)
    frame._after_id = frame.after(frame._delay_ms, _schedule_next)

    return frame
