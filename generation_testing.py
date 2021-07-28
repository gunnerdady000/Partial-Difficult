import timeit
import numpy as np
import pandas as pd
from generation import CaDeer


def main():
    ca_test()


def csv_test():
    deer = CaDeer()
    path = [.32, .2, .4, .6]
    motility = ["nope", "nope", "yup", "yup"]
    deer.excel_write(path_taken=path, excel_output_name="testOutput", motilities_taken=motility)
    print("Yeet")


def ca_test():
    # size of the world
    x, y = 150, 150

    # scaling of the world
    scale = 100.0

    # number of octaves
    octaves = 8

    # trying w/ 15 features
    names = ['Open space', 'Low Space', 'Med Space', 'barren', 'pasture', 'crops', 'sparse veg', 'open water',
             'apline sparse', 'aspen', 'juniper', 'dry spruce', 'dry mixed', 'pine', 'mixed conifer']

    # 15 feature motility values
    motility_values = np.array(
        [0.94, 0.59, 1.14, 3.12, 0.46, 0.96, 1.11, 1.02, 1.80, 1.51, 1.90, 2.33, 1.41, 1.98, 3.09])

    # 15 feature color set
    colors = np.array([[128, 128, 0], [85, 107, 47], [107, 142, 35], [210, 180, 140], [154, 205, 50], [218, 165, 32],
                       [144, 238, 144], [65, 105, 225], [255, 248, 220], [173, 255, 47], [143, 188, 143], [0, 128, 0],
                       [34, 139, 34], [46, 139, 87], [0, 100, 0]])

    # 15 feature color range
    color_range = np.array([-0.867, -0.733, -0.6, -0.467, -0.333, -0.2, -0.067, 0, 0.067, 0.2, 0.333, 0.467, 0.733,
                            0.867, 1])

    # creating and initializing the Cellar Autonoma of deer
    deer = CaDeer(scale=scale, octaves=octaves, features=15)

    # deer.excel_read("list_of_stuff.xlsx")
    deer.gather_features("test_output", light_mode=False, input_excel_name="list_of_stuff.xlsx", color_range=color_range, colors=colors, motility_values=motility_values,
                         terrain_names=names)

    # creating the world given the sizes and feature list
    deer.create_world(length=x, width=y)

    # outputting the perlin noise world
    deer.output_world(world=deer.world, gray=True)

    # creates the color version of the world
    deer.color_world()

    # outputting the world in color after applying the color range values
    deer.output_world(world=deer.world_color, gray=False)

    # outputting the grayscale version of the color world
    deer.output_world(world=deer.ca_world, gray=True)

    # showing the CA pathing of the deer showing Live update and recording the path taken
    deer.pathing(time=50000, live_update=True, mpfour_output="output")

    print("Done")


def numpy_testing():
    world = np.random.uniform(-1, 1, size=(10, 10))
    color_range = np.array([-0.6, -0.2, 0, 0.2, 0.6, 1])
    world = np.where(world < color_range[0], color_range[0], world)
    print(world)
    for i in range(1, color_range.size - 1):
        print(i)
        previous = color_range[i - 1]
        print(previous)
        current = color_range[i]
        print(current)
        world = np.where(previous < world < current, current, world)
        print(world)


if __name__ == "__main__":
    main()
