"""Second page UI components for PLAiCE."""
from typing import Callable
import tkinter as tk


def create_second_page(master: tk.Misc, on_back: Callable[[], None]) -> tk.Frame:
    """Create and return the second page frame.

    Parameters
    - master: parent widget
    - on_back: callback to invoke when the Back button is pressed
    """
    frame = tk.Frame(master, padx=10, pady=10)
    label = tk.Label(frame, text="Second page", font=(None, 14))
    label.pack(pady=(0, 10))

    back_btn = tk.Button(frame, text="Back", command=on_back)
    back_btn.pack()

    return frame
