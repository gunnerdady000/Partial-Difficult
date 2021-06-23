import numpy as np
from generation import CaDeer

def main():
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
    motility_values = np.array([0.94, 0.59, 1.14, 3.12, 0.46, 0.96, 1.11, 1.02, 1.80, 1.51, 1.90, 2.33, 1.41, 1.98, 3.09])

    # 15 feature color set
    colors = np.array([[128, 128, 0], [85, 107, 47], [107, 142, 35], [210, 180, 140], [154, 205, 50], [218, 165, 32],
                       [144, 238, 144], [65, 105, 225], [255, 248, 220], [173, 255, 47], [143, 188, 143], [0, 128, 0],
                       [34, 139, 34], [46, 139, 87], [0, 100, 0]])

    # 15 feature color range
    color_range = np.array([[-0.867, -0.733, -0.6, -0.467, -0.333, -0.2, -0.067, 0, 0.067, 0.2, 0.333, 0.467, 0.733,
                             0.867, 1]])

    # feature list which must be as follows: ndarray([colors, color_range], dtype=object)
    feature_list = np.array([colors, color_range], dtype=object)

    # creating and initializing the Cellar Autonoma of deer
    deer = CaDeer(scale=scale, octaves=octaves, features=15)

    # creating the world given the sizes and feature list
    deer.create_world(x, y, feature_list)

    # outputting the perlin noise world
    deer.output_world(deer.world, gray=True)

    # creates the color version of the world
    deer.color_world()

    # outputting the world in color after applying the color range values
    deer.output_world(deer.world_color, gray=None)

    # outputting the grayscale version of the color world
    deer.output_world(deer.ca_world, gray=True)

    # setup for the CA to work correctly, which requires the motility values
    deer.ca_setup(motility_values=motility_values)

    # showing the CA pathing of the deer showing Live update and recording the path taken
    deer.pathing(time=10000, live_update=True, path=True, names=names)


if __name__ == "__main__":
    main()
