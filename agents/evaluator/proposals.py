import numpy as np
from typing import List, Tuple

rgb = Tuple[int, int, int]
Proposal = Tuple[int, int, rgb]

def patch_to_pixel_coords(
    patch_idx: int,
    image_width: int,
    patch_size: int = 16
):
    patches_per_row = image_width // patch_size
    py = patch_idx // patches_per_row
    px = patch_idx % patches_per_row

    x = px * patch_size + patch_size // 2
    y = py * patch_size + patch_size // 2
    return x, y


def extract_pixel_color(image: np.ndarray, x: int, y: int) -> rgb:
    h, w = image.shape[:2]
    # Clamp coordinates to image bounds
    x = max(0, min(w - 1, x))
    y = max(0, min(h - 1, y))
    r, g, b = image[y, x]
    return int(r), int(g), int(b)


def generate_proposals(
    diff_scores: np.ndarray,
    current_img: np.ndarray,
    generated_img: np.ndarray,
    top_x: int
) -> List[Proposal]:
    """
    Select top-X most different patches and create pixel proposals
    """
    indices = np.argsort(diff_scores)[-top_x:]

    proposals = []
    h, w, _ = current_img.shape

    for idx in indices:
        x, y = patch_to_pixel_coords(idx, w)
        col = extract_pixel_color(generated_img, x, y)
        proposals.append((x, y, col))

    return proposals
