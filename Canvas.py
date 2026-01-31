from PIL import Image
from typing import Tuple

RGB = Tuple[int, int, int]
Pos = Tuple[int, int]

class Canvas:
    def __init__(self, x, y):
        # self.pixels[y, x] = pixel at y, x
        # |---------------------> + x
        # |
        # v
        # + y
        self.pixels = [[(0, 0, 0) for _ in range(x)] for _ in range(y)]


    def write(self, x, y, col: RGB):
        self.pixels[y][x] = col
        
    def export(self):
        height = len(self.pixels)
        width = len(self.pixels[0])
        img = Image.new("RGB", (width, height))
        img.putdata([pixel for row in self.pixels for pixel in row])
        img.save("output.png")
        print("image created")


canvas = Canvas(5, 5)
canvas.write(3, 3, (255, 255, 255))
canvas.export() 
