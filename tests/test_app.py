"""Basic tests for the PLAiCE app skeleton."""
import unittest
import tkinter as tk

from plaice_app.app import App


class AppSmokeTest(unittest.TestCase):
    def test_build_ui_and_destroy(self):
        root = tk.Tk()
        # don't show the window during tests
        root.withdraw()
        try:
            app = App(title="Test")
            app.build_ui(root)
            # no assertions beyond "no exceptions" for now; ensure widgets exist
            self.assertIsNotNone(root)
        finally:
            root.destroy()


if __name__ == "__main__":
    unittest.main()
