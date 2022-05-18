import webcolors
from random import randint

class ColorUtils:
    def __init__(self):
        pass

    def create_n_hex_colors(self, n):
        return ['#%06X' % randint(0, 0xFFFFFF) for _ in range(n)]

    def hex_to_name(self, colors):
        return [webcolors.hex_to_name(color) for color in colors]