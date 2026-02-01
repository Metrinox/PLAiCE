from PIL import Image
from typing import Tuple

rgb = Tuple[int, int, int]

class Canvas:
    def __init__(self, x, y):
        # self.pixels[y, x] = pixel at y, x
        # |---------------------> + x
        # |
        # v
        # + y
        self.pixels = [[(0, 0, 0) for _ in range(x)] for _ in range(y)]
        self.age = 0

    def read(self, startX, startY, width, height):
        result = []

        for y in range(startY, startY + height):
            if y < 0 or y >= len(self.pixels):
                continue

            row = []
            for x in range(startX, startX + width):
                if x < 0 or x >= len(self.pixels[0]):
                    continue
                row.append(self.pixels[y][x])

            if row:
                result.append(row)
        self.age += 1
        return result

    def write(self, x, y, col: rgb):
        self.pixels[y][x] = col
        self.age += 1

    def export(self):
        height = len(self.pixels)
        width = len(self.pixels[0])
        img = Image.new("RGB", (width, height))
        img.putdata([pixel for row in self.pixels for pixel in row])
        img.save("output.png")
        print("image created")

    def getAge(self):
        return self.age
