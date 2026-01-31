import numpy as np

def patch_importance(similarities, threshold=0.95):
    """
    Lower similarity → higher importance
    """
    return [1.0 - s if s < threshold else 0.0 for s in similarities]


def expand_to_pixel_map(patch_scores, image_shape, patch_size=16):
    """
    Maps patch importance back to pixel space.
    """
    h, w, _ = image_shape
    heatmap = np.zeros((h, w))

    idx = 0
    for y in range(0, h, patch_size):
        for x in range(0, w, patch_size):
            heatmap[y:y+patch_size, x:x+patch_size] = patch_scores[idx]
            idx += 1

    return heatmap
