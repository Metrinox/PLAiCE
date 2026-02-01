"""Simple Tkinter-based desktop app skeleton."""
from typing import Optional
import tkinter as tk

from .pages.second_page import create_second_page


class App:
    """Application container with simple page navigation.

    The app keeps a registry of frames by name and exposes show_frame(name)
    to switch between them.
    """

    def __init__(self, title: str = "PLAiCE"):
        self.title = title
        self.root: Optional[tk.Tk] = None
        # store frames by name for simple navigation
        self.frames: dict[str, tk.Frame] = {}
        self.current_page: Optional[str] = None

    def build_ui(self, master: tk.Tk) -> None:
        """Build the UI into the provided master widget.

        Separated from creating the root so tests can construct widgets without
        entering the mainloop.
        """
        master.title(self.title)
        master.geometry("400x200")

        # Main page frame
        main = tk.Frame(master, padx=10, pady=10)
        label = tk.Label(main, text="Welcome to PLAiCE", font=(None, 14))
        label.pack(pady=(0, 10))

        btn_frame = tk.Frame(main)
        btn_frame.pack()

        next_btn = tk.Button(btn_frame, text="Next", command=lambda: self.show_frame("second"))
        next_btn.pack(side=tk.LEFT)

        quit_btn = tk.Button(btn_frame, text="Quit", command=master.quit)
        quit_btn.pack(side=tk.LEFT, padx=(8, 0))

        # Second page frame (extracted)
        second = create_second_page(master, on_back=lambda: self.show_frame("main"))

        # register frames for navigation
        self.frames["main"] = main
        self.frames["second"] = second

        # initially show main page
        self.show_frame("main")

    def run(self) -> None:
        """Create the root window, build the UI and start the main loop."""
        self.root = tk.Tk()
        # When launching normally, show the window.
        self.build_ui(self.root)
        self.root.mainloop()

    def show_frame(self, name: str) -> None:
        """Show the frame registered under `name` and hide others.

        This uses pack_forget to hide frames and pack to show the requested one.
        """
        if name not in self.frames:
            raise KeyError(f"no page named {name}")

        # hide all frames
        for nm, fr in self.frames.items():
            fr.pack_forget()

        # show requested frame
        frame = self.frames[name]
        frame.pack(expand=True, fill=tk.BOTH)
        self.current_page = name


def main() -> None:
    app = App()
    app.run()


if __name__ == "__main__":
    main()
