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

    def test_navigation_between_pages(self):
        root = tk.Tk()
        root.withdraw()
        try:
            app = App(title="NavTest")
            app.build_ui(root)
            # initial page should be 'main'
            self.assertEqual(app.current_page, "main")

            # navigate to second page
            app.show_frame("second")
            self.assertEqual(app.current_page, "second")

            # go back
            app.show_frame("main")
            self.assertEqual(app.current_page, "main")
        finally:
            root.destroy()


if __name__ == "__main__":
    unittest.main()
