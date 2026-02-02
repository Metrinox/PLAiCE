from PIL import Image
from typing import Tuple
import random

RGB = Tuple[int, int, int]
Pos = Tuple[int, int]

class Canvas:
    def __init__(self, x, y):
        # self.pixels[y, x] = pixel at y, x
        # |---------------------> + x
        # |
        # v
        # + y
        self.pixels = [
            [
                (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                for _ in range(x)
            ]
            for _ in range(y)
        ]
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
        return result

    def write(self, x, y, col: RGB):
        self.pixels[y][x] = col

    def export(self, path="output.png"):
        height = len(self.pixels)
        width = len(self.pixels[0])
        print(f"canvas age: {self.age}")
        img = Image.new("RGB", (width, height))
        img.putdata([pixel for row in self.pixels for pixel in row])
        img.save(path)
        print(f"image created: {path}")

    def getAge(self):
        return self.age

    def increment_age(self):
        self.age += 1
