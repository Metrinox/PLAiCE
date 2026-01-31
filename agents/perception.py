import numpy as np

def extract_fov(canvas: np.ndarray, x: int, y: int, radius: int):
    """
    Returns a local patch around (x, y).
    """
    h, w, _ = canvas.shape
    x0 = max(0, x - radius)
    x1 = min(w, x + radius + 1)
    y0 = max(0, y - radius)
    y1 = min(h, y + radius + 1)
    return canvas[y0:y1, x0:x1]
