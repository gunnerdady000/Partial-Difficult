import matplotlib.pyplot as plt
import numpy as np
import noise
from scipy import ndimage
from PIL import Image
from numba import jit
from matplotlib import cm
from generation import CaDeer

def main():
    x, y = 250, 250
    scale = 100.0
    octaves = 6
    deer = CaDeer(scale=scale, octaves=octaves, features=5)
    deer.create_world(x, y)
    deer.output_world(deer.world, gray=True)
    deer.color_world()
    deer.output_world(deer.world_color, gray=None)
    deer.output_world(deer.ca_world, gray=True)
    deer.ca_setup()
    deer.pathing(10000, True, True)


if __name__ == "__main__":
    main()
