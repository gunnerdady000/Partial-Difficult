import timeit
import numpy as np
from CaDeerMotility import CaDeer


def main():
    default_case()
    advance_case()
    hacking()


def default_case():
    # class initialization
    deer = CaDeer()

    # class function call, gather features
    deer.gather_features("test_output")

    # create the world, default size is 250 by 250
    deer.create_world()

    # optional call to output the world in black and white, while using a class variable
    deer.output_world(deer.world, True)

    # creates the color version of the world
    deer.color_world()

    # outputting the world in color after applying the color range values
    deer.output_world(deer.world_color)

    # outputting the grayscale version of the color world
    deer.output_world(deer.ca_world, True)

    # run the path finding algorithm
    deer.pathing(10000)

    print("Done, with Default Case")


def advance_case():
    # size of the world
    x, y = 150, 150

    # scaling of the world
    scale = 100.0

    # number of octaves
    octaves = 8

    # creating and initializing the Cellar Autonoma of deer
    deer = CaDeer(scale=scale, octaves=octaves, persistence=0.585, lacunarity=2.68, base=0, features=15)

    # collect the features from Excel
    deer.gather_features("test_output", light_mode=False, color_range=None, input_excel_name="test_input.xlsx",
                         colors=None, motility_values=None, terrain_names=None)

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

    # remove comments to specify a starting point, x and y are opposite of the built in cursor within the plot output
    # deer.starting_pos_x = 71
    # deer.starting_pos_y = 24

    # showing the CA pathing of the deer showing Live update and recording the path taken
    deer.pathing(time=100000, live_update=False, encoding=None, mpfour_output=None)

    print("Done with Advance Case")


def hacking():
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
    deer = CaDeer(scale=scale, octaves=octaves, persistence=0.585, lacunarity=2.68, base=0, features=15)

    # collect the features without the use of an excel file, because we know how to code
    deer.gather_features("test_output", light_mode=False, color_range=color_range,
                         colors=colors, motility_values=motility_values, terrain_names=names)

    # user could save or create a world to use by setting the following comment with a correct ndArray (you fill in)
    # deer.world = (some ndArray)
    # self.x = (length of ndArray)
    # self.y = (length of ndArray)
    # and comment out the deer.create_world()
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

    # specify a starting point, x and y are opposite of the built in cursor within the plot output
    deer.starting_pos_x = 71
    deer.starting_pos_y = 24

    # showing the CA pathing of the deer showing Live update and recording the path taken
    deer.pathing(time=20000, live_update=False, encoding=None, mpfour_output="test_output_two")

    print("Done")


if __name__ == "__main__":
    main()
