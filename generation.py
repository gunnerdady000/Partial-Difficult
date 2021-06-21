import numpy as np
import matplotlib.pyplot as plt
import noise
from scipy import ndimage
from PIL import Image
from numba import jit
from matplotlib import cm


class CaDeer(object):

    def __init__(self, scale=100.0, octaves=6, persistence=None, lacunarity=None, base=None, features=5):
        # number of features to color and use in model
        self.features = features
        # scale of the map
        self.scale = scale
        # holds the number of octaves, close to how many features will appear
        self.octaves = octaves
        # holds the amount of persistence of repeating characteristics
        self.persistence = persistence
        # holds the detail level
        self.lacunarity = lacunarity
        # used to determine starting point of the map
        self.base = base

        # used to get corrected grayscale
        color = cm.get_cmap('gray')
        self.cmap = color.reversed()

    def feature_fill(self):
        # returns the colors needed to color the map and the values to check given the colors wanted
        if self.features == 3:  # green, blue, gray
            self.colors = [[34, 139, 34], [65, 105, 225], [139, 137, 137]]
            self.color_range = [-0.05, 0, 0.36]
        elif self.features == 4:  # yellow, blue, green, gray
            self.colors = [[240, 230, 140], [65, 105, 225], [34, 139, 34], [139, 137, 137]]
            self.color_range = [-0.05, 0, 0.2, 0.36]
        elif self.features == 5:  # yellow, blue, green, gray, white
            self.colors = [[240, 230, 140], [65, 105, 225], [34, 139, 34], [139, 137, 137], [255, 250, 250]]
            self.color_range = [-0.05, 0, 0.2, 0.36, 1]

    def create_world(self, length=250, width=250):
        # number of pixels of the world for a set length
        self.length = length
        # number of pixels of the world for a set width
        self.width = width

        # check to see if we need to fill in with random values
        if self.persistence is None:
            self.persistence = np.random.uniform(0.2, 0.6)
        if self.lacunarity is None:
            self.lacunarity = np.random.uniform(2.2, 3)
        if self.base is None:
            self.base = np.random.randint(0, 25)

        # create the world array
        self.world = np.zeros((self.length, self.width))

        # use perlin noise to generate random world
        for i in range(self.length):
            for j in range(self.width):
                self.world[i][j] = noise.pnoise2(i / self.scale, j / self.scale, self.octaves, self.persistence,
                                                 self.lacunarity, self.length, self.width, self.base)

        # collect the needed colors and ranges
        self.feature_fill()

    def output_world(self, world, gray=None):
        plt.figure()
        if gray is not None:
            print("Grayscale")
            plt.imshow(world, cmap=self.cmap)
        else:
            print("RGB")
            plt.imshow(world.astype(np.uint8))
        plt.show()

    def color_world(self):
        # create RGB from size of the heat map of the world
        self.world_color = np.zeros(self.world.shape + (3,))
        # loop through and create a CA model from the original world
        self.ca_world = np.zeros(self.world.shape)

        # used to simulate elif without else using an array
        found = False
        # create RGB from heat map
        for i in range(self.length):
            for j in range(self.width):
                for k in range(self.features):
                    if self.world[i][j] < self.color_range[k] and not found:
                        # create regions from noise
                        self.ca_world[i][j] = self.color_range[k]
                        # create RBG version from noise
                        self.world_color[i][j] = self.colors[k]
                        # skip until the next cycle
                        found = True

                    # check to see if we need to update the ability to find another color
                    if k == self.features - 1:
                        found = False

    def ca_setup(self, motility_values=None):
        if motility_values is not None:
            self.motility_values = motility_values
        else:
            #                     yellow, blue, green, gray, white
            self.motility_values = [0.94, 1.02, 0.46, 2.33, 3.12]
            #                      [-0.05, 0,    0.2, 0.36, 1]

        # starting values must not be on the edges of the world... for now
        self.starting_pos_x = np.random.randint(1, self.length - 1)
        self.starting_pos_y = np.random.randint(1, self.width - 1)

        # set current position to starting position
        self.current_pos_x = self.starting_pos_x
        self.current_pos_y = self.starting_pos_y

    def moore_neighborhood(self, square):
        # find the average value of the current square
        for i in range(len(square)):
            for j in range(len(square[0])):
                for k in range(self.features):
                    if square[i][j] == self.color_range[k]:
                        square[i][j] = self.motility_values[k]

        average = np.average(square)


        # default for current motility
        current_motility = 100.0
        self.next_position_y = 0
        self.next_position_x = 0

        for i in range(len(square)):
            for j in range(len(square[0])):
                # compare options based off being less than average, while excluding the curr_pos
                check_motility = square[i][j]

                # check to see if the new motility is less than the average
                # ignore the current position in the middle of the grid
                if not(i == 1 and j == 1):
                    # add randomness to increase movement
                    random_step = np.random.normal()
                    if check_motility < average + random_step:
                        # print("print passed average + random_step", average + random_step)
                        if check_motility < current_motility:
                            # print("passed current")
                            current_motility = check_motility
                            self.next_position_x = i
                            self.next_position_y = j

        # update for the fact that position is in the middle of the grid
        self.next_position_x -= 1
        self.next_position_y -= 1

    def pathing(self, time, live_update=False, path=True):
        # make a copy of the world
        buffer = np.copy(self.world_color)

        # set the next position to 0
        self.next_position_x = 0
        self.next_position_y = 0

        # create empty array with all of the taken positions to show total path
        if path:
            path_taken = []

        if live_update:
            plt.show()
            plt.ion()
            plt.figure()

        for t in range(time):

            # output the deer on the colored world
            if live_update:
                plt.clf()
                # use maroon RGB for deer
                buffer[self.current_pos_x][self.current_pos_y] = [128, 0, 0]
                plt.imshow(buffer.astype(np.uint8))
                string = "t {} \n (x,y): ({},{})\n".format(t, self.current_pos_x, self.current_pos_y)
                plt.title(string)
                plt.pause(0.3)

            # add current position to the path taken
            if path:
                current_position = np.array([self.current_pos_x, self.current_pos_y])
                path_taken.append(current_position)

            # find the square that the deer is considering based off of the current position
            # need to add in edge detection
            square_choice = self.ca_world[self.current_pos_x - 1:self.current_pos_x + 2,
                            self.current_pos_y - 1:self.current_pos_y + 2]

            # use Moore neighborhood to select the next position
            self.moore_neighborhood(square_choice)

            # update current position to future position
            self.current_pos_x += self.next_position_x
            self.current_pos_y += self.next_position_y

        if live_update:
            plt.ioff()
            plt.close()

        if path:
            print(path_taken)

        self.output_world(buffer)