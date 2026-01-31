def clamp_rgb(rgb):
    return tuple(max(0, min(255, int(c))) for c in rgb)
