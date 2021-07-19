import numpy as np
import pandas as pd
from generation import CaDeer


def main():
    ca_test()


def csv_test():
    xl = pd.ExcelFile("list_of_stuff.xlsx")
    df = xl.parse(0, skiprows=0)
    stuff = df.values
    names = stuff[:, 0].tolist()
    motility_values = stuff[:, 1]
    colors = stuff[:, 2]
    colors = np.asarray([colors[i].replace(' ', '') for i in range(colors.size)])
    colors = np.asarray([np.fromstring(colors[i], dtype=int, sep=',') for i in range(colors.size)])

    ones = np.ones_like(colors[:, 0]).reshape((colors.shape[0], 1))
    colors = np.hstack((colors, ones))

    color_range = stuff[:, 3]
    print("Yolo")


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
    colors = np.array(
        [[128 / 255, 128 / 255, 0 / 255, 1], [85 / 255, 107 / 255, 47 / 255, 1], [107 / 255, 142 / 255, 35 / 255, 1],
         [210 / 255, 180 / 255, 140 / 255, 1], [154 / 255, 205 / 255, 50 / 255, 1], [218 / 255, 165 / 255, 32 / 255, 1],
         [144 / 255, 238 / 255, 144 / 255, 1], [65 / 255, 105 / 255, 225 / 255, 1],
         [255 / 255, 248 / 255, 220 / 255, 1],
         [173 / 255, 255 / 255, 47 / 255, 1], [143 / 255, 188 / 255, 143 / 255, 1], [0 / 255, 128 / 255, 0 / 255, 1],
         [34 / 255, 139 / 255, 34 / 255, 1], [46 / 255, 139 / 255, 87 / 255, 1],
         [0 / 255, 100 / 255, 0 / 255, 1]])

    # 15 feature color range
    color_range = np.array([[-0.867, -0.733, -0.6, -0.467, -0.333, -0.2, -0.067, 0, 0.067, 0.2, 0.333, 0.467, 0.733,
                             0.867, 1]])

    # feature list which must be as follows: ndarray([colors, color_range], dtype=object)
    feature_list = np.array([colors, color_range], dtype=object)

    # creating and initializing the Cellar Autonoma of deer
    deer = CaDeer(scale=scale, octaves=octaves, features=15)
    # deer = CaDeer(scale=scale, octaves=octaves, features=5)

    deer.excel_read("list_of_stuff.xlsx")

    # creating the world given the sizes and feature list
    deer.create_world(length=x, width=y, feature_list=feature_list)

    # outputting the perlin noise world
    deer.output_world(world=deer.world, gray=True)

    # creates the color version of the world
    deer.color_world()

    # outputting the world in color after applying the color range values
    deer.output_world(world=deer.world_color, gray=False)

    # outputting the grayscale version of the color world
    deer.output_world(world=deer.ca_world, gray=True)

    # setup for the CA to work correctly, which requires the motility values
    deer.ca_setup(motility_values=motility_values, light_mode=True)

    # showing the CA pathing of the deer showing Live update and recording the path taken
    deer.pathing(time=100000, live_update=False, names=names)


def time_test():
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
    colors = np.array(
        [[128 / 255, 128 / 255, 0 / 255, 1], [85 / 255, 107 / 255, 47 / 255, 1], [107 / 255, 142 / 255, 35 / 255, 1],
         [210 / 255, 180 / 255, 140 / 255, 1], [154 / 255, 205 / 255, 50 / 255, 1], [218 / 255, 165 / 255, 32 / 255, 1],
         [144 / 255, 238 / 255, 144 / 255, 1], [65 / 255, 105 / 255, 225 / 255, 1],
         [255 / 255, 248 / 255, 220 / 255, 1],
         [173 / 255, 255 / 255, 47 / 255, 1], [143 / 255, 188 / 255, 143 / 255, 1], [0 / 255, 128 / 255, 0 / 255, 1],
         [34 / 255, 139 / 255, 34 / 255, 1], [46 / 255, 139 / 255, 87 / 255, 1],
         [0 / 255, 100 / 255, 0 / 255, 1]])
    # cellular automata
    # 15 feature color range
    color_range = np.array([[-0.867, -0.733, -0.6, -0.467, -0.333, -0.2, -0.067, 0, 0.067, 0.2, 0.333, 0.467, 0.733,
                             0.867, 1]])

    # feature list which must be as follows: ndarray([colors, color_range], dtype=object)
    feature_list = np.array([colors, color_range], dtype=object)

    # creating and initializing the Cellar Autonoma of deer
    # deer = CaDeer(scale=scale, octaves=octaves, features=15)
    deer = CaDeer(scale=scale, octaves=octaves, features=15)

    # creating the world given the sizes and feature list
    deer.create_world(length=x, width=y, feature_list=feature_list)

    # creates the color version of the world
    deer.color_world()

    # setup for the CA to work correctly, which requires the motility values
    deer.ca_setup(motility_values=motility_values, light_mode=True)

    deer.test1()
    deer.test2()


def numpy_testing():
    world = np.random.uniform(-1, 1, size=(10, 10))
    color_range = np.array([-0.6, -0.2, 0, 0.2, 0.6, 1])
    world = np.where(world < color_range[0], color_range[0], world)
    print(world)
    for i in range(1, color_range.size-1):
        print(i)
        previous = color_range[i-1]
        print(previous)
        current = color_range[i]
        print(current)
        world = np.where(previous < world < current, current, world)
        print(world)


if __name__ == "__main__":
    main()
