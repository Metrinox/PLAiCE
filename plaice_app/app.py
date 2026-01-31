"""Simple Tkinter-based desktop app skeleton."""
from typing import Optional
import tkinter as tk


class App:
    """Application container.

    Usage:
        app = App(title="My App")
        app.run()
    """

    def __init__(self, title: str = "PLAiCE"):
        self.title = title
        self.root: Optional[tk.Tk] = None

    def build_ui(self, master: tk.Tk) -> None:
        """Build the UI into the provided master widget.

        Separated from creating the root so tests can construct widgets without
        entering the mainloop.
        """
        master.title(self.title)
        master.geometry("400x200")

        frm = tk.Frame(master, padx=10, pady=10)
        frm.pack(expand=True, fill=tk.BOTH)

        label = tk.Label(frm, text="Welcome to PLAiCE", font=(None, 14))
        label.pack(pady=(0, 10))

        btn_frame = tk.Frame(frm)
        btn_frame.pack()

        quit_btn = tk.Button(btn_frame, text="Quit", command=master.quit)
        quit_btn.pack(side=tk.LEFT)

    def run(self) -> None:
        """Create the root window, build the UI and start the main loop."""
        self.root = tk.Tk()
        # When launching normally, show the window.
        self.build_ui(self.root)
        self.root.mainloop()


def main() -> None:
    app = App()
    app.run()


if __name__ == "__main__":
    main()
