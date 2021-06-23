import numpy as np
import matplotlib.pyplot as plt
import noise
from numba import jit
from matplotlib import cm
import matplotlib.patches as mpatches


class CaDeer(object):

    def __init__(self, scale=100.0, octaves=6, persistence=None, lacunarity=None, base=None, features=5):
        # number of features to color and use in model
        if features > 5:
            self.features = features
        else:
            print("Features have been set to default, please enter a number of features greater than 5")
            self.features = 5
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

    def ca_setup(self, motility_values=None):
        if motility_values is not None:
            if motility_values.size == self.features:
                self.motility_values = motility_values
            else:
                print("Motility values have been set to default. Please enter motility values that match the number "
                      "of features.")
                self.default()
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

    def color_append(self):
        # add in color of path
        color = np.vstack([self.colors, [128, 0, 0]])

        # add color of deer
        color = np.vstack([color, [255, 0, 255]])

        # convert from rgb decimal to rgb hex
        colors = ['#{:02x}{:02x}{:02x}'.format(color[i][0], color[i][1], color[i][2]) for i in range(color.shape[0])]

        # return appending colors array
        return colors

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

    def create_world(self, length=250, width=250, feature_list=None):
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
        self.feature_fill(feature_list)

    def default(self):
        # default of 5 features
        self.features = 5
        # default of 5 colors
        self.colors = [[240, 230, 140], [65, 105, 225], [34, 139, 34], [139, 137, 137], [255, 250, 250]]
        # default of 5 color ranges
        self.color_range = [-0.05, 0, 0.2, 0.36, 1]
        # default of 5 molitity values
        self.motility_values = [0.94, 1.02, 0.46, 2.33, 3.12]
        # default of 5 names
        self.names = ['barren', 'water', 'pasture', 'spruce', 'mixed confir']

    def feature_fill(self, feature_list=None):
        # returns the colors needed to color the map and the values to check given the colors wanted
        if feature_list is not None:
            # color size check
            self.colors = np.copy(feature_list[0])
            if self.colors.shape[0] != self.features:
                print("Default values of 5 features has been selected. "
                      "PLease enter number of colors that matches the number of features.")
                self.default()
            # color range check
            self.color_range = np.copy(feature_list[1]).reshape((feature_list[1].size,))
            if len(self.color_range) != self.features:
                print("Default values of 5 features has been selected. "
                      "PLease enter number of color range that matches the number of features.")
                self.default()
        else:               # yellow,          blue,            green,         gray,           white
            self.colors = [[240, 230, 140], [65, 105, 225], [34, 139, 34], [139, 137, 137], [255, 250, 250]]
            self.color_range = [-0.05, 0, 0.2, 0.36, 1]

    def moore_neighborhood(self, square):
        # convert from the heat map values to the respective motility values
        for i in range(len(square)):
            for j in range(len(square[0])):
                for k in range(self.features):
                    if square[i][j] == self.color_range[k]:
                        square[i][j] = self.motility_values[k]

        # find the average value of the current square
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
                if not (i == 1 and j == 1):
                    # add randomness to increase movement
                    if check_motility < average + np.random.normal():
                        if check_motility < current_motility:
                            current_motility = check_motility
                            self.next_position_x = i
                            self.next_position_y = j

        # update for the fact that position is in the middle of the grid
        self.next_position_x -= 1
        self.next_position_y -= 1

    def live_updater(self, buffer, t, colors, motility, prev_pos_x, prev_pos_y):
        # needed for video
        plt.clf()

        if self.current_pos_x == -151:
            self.current_pos_x = np.remainder(self.current_pos_x, self.width)

        if self.current_pos_y == -151:
            self.current_pos_y = np.remainder(self.current_pos_y, self.length)

        # Marron RGB for previous spots for deer [128, 0, 0]
        buffer[prev_pos_x][prev_pos_y] = [128, 0, 0]

        # buffer[prev_pos_x][prev_pos_y] = self.rgba_to_hex(self.world_color[prev_pos_x][prev_pos_y])

        # use pink for current position [255, 0, 255]
        buffer[self.current_pos_x][self.current_pos_y] = [255, 0, 255]

        # add in location and iteration/time passed
        string = "t {} \n (x,y): ({},{})\n s {} o {} p {} l {} b {} f {}".format(t,
                            np.remainder(self.current_pos_x, self.width), np.remainder(self.current_pos_y, self.length),
                            self.scale, self.octaves, np.round(self.persistence, 3), np.round(self.lacunarity, 3),
                            self.base, self.features)

        # plot the updated time/iteration and current coordinates of the deer
        plt.title(string)

        # plot the buffer world image
        plt.imshow(buffer.astype(np.uint8), interpolation='none')

        # create the legend box with names of each color as well the as the motility values
        patches = [mpatches.Patch(color=colors[i], label='{:^5} {:>10}'.format(self.names[i], motility[i])) for i in
                   range(len(self.names))]

        # plot the legend box
        plt.legend(handles=patches, bbox_to_anchor=(1.05, 0.0, 0.3, 1), loc=2, borderaxespad=0.1)  # , mode='expand')

        # show the image
        plt.show()

        # use to determine the time between each iteration
        plt.pause(0.3)

    def pathing(self, time, live_update=False, path=True, names=None):
        # make a copy of the world
        buffer = np.copy(self.world_color)

        # name size check
        if names is not None:
            if len(names) != self.features:
                print("Default values have been set. Please enter an equal amount of names to features.")
                self.default()
            else:
                self.names = names
        else:
            self.default()

        # clear up strings by adding path and deer
        motility = self.string_names()

        # clear up colors by adding path and deer
        colors = self.color_append()

        # set the next position to 0
        self.next_position_x = 0
        self.next_position_y = 0

        # previous position
        prev_pos_x = self.current_pos_x
        prev_pos_y = self.current_pos_y

        # create empty array with all of the taken positions to show total path
        if path:
            path_taken = []

        # used to start the video format
        if live_update:
            plt.show()
            plt.ion()
            plt.figure()

        # loop through the iterations known as time
        for t in range(time):

            # output the deer on the colored world
            if live_update:
                self.live_updater(buffer=buffer, t=t, colors=colors, motility=motility,
                                  prev_pos_x=prev_pos_x, prev_pos_y=prev_pos_y)

            # add current position to the path taken
            if path:
                current_position = np.array([self.current_pos_x, self.current_pos_y])
                path_taken.append(current_position)


            # need to add in edge detection
            #self.edge_detection()

            # find the square that the deer is considering based off of the current position
            square_choice = self.ca_world[self.current_pos_x - 1:self.current_pos_x + 2,
                            self.current_pos_y - 1:self.current_pos_y + 2]

            # use Moore neighborhood to select the next position
            self.moore_neighborhood(square_choice)

            # previous position
            prev_pos_x = self.current_pos_x
            prev_pos_y = self.current_pos_y

            # update current position to future position
            self.current_pos_x += self.next_position_x
            self.current_pos_y += self.next_position_y

        if live_update:
            # used to end the video format
            plt.ioff()
            plt.close()

        if path:
            print(path_taken)

        self.output_world(buffer)

    def rgba_to_hex(self, original_block):
        original_block
        rgba_hex = '0x{:02x}{:02x}{:02x}{:02x}'.format(int(original_block[0]), int(original_block[1]),
                                                      int(original_block[2]), 144)

        return int(rgba_hex, 0)

    def string_names(self):
        # add in the path and deer to the name list
        self.names += ['path', 'deer']

        # converts to a list of strings for motility values
        motility = list(map(str, self.motility_values))

        # add in NA for deer and path, as they do not have their own values on the value map
        motility += ['NA', 'NA']

        # return the motility string
        return motility

    def output_world(self, world, gray=None):
        plt.figure()
        if gray is not None:
            print("Grayscale")
            plt.imshow(world, cmap=self.cmap)
        else:
            print("RGB")
            plt.imshow(world.astype(np.uint8))
        plt.show()