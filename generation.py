import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import noise
import time
from numba import jit
from matplotlib import cm
import matplotlib.patches as mpatches


class CaDeer(object):
    """This is a Cellular Automata Class that is be used to check motility values of deer. The use of Perlin Noise
        allows for repeatable, but random enough generated worlds to check a minimum of 5 motility values in the form of
        features within the Perlin Noise.
        :class:`CaDeer`

        :param scale: Controls the viewing scale of the generated Perlin Noise. Default is 100.0
        :type scale: float, optional
        :param octaves: Number of features that will appear within the distinct float steps generated within
        the Perlin Noise. Default is 6.
        :type octaves: int, optional
        :param persistence: Number of repeating characteristics within the Perlin Noise. Default is None, which will be
        changed to a random float between (0.2, 0.6).
        :type persistence: float, optional
        :param lacunarity: Level of detail found within the Perlin Noise. Default is None, which will be changed to a
        random float between (2.2, 3).
        :type lacunarity: float, optional
        :param base: Determines starting point of the map. Default is None, which will be changed to an integer between
        (2, 25).
        :type base: int, optional
        :param features: Number of features to color and use in model. Must match the number of elements within the
        colors, color_range, feature_list, and names arrays. Default is 5, and will be used for all other functions
        to run the cellular automata.
        :param features: int, optional
    """

    def __init__(self, scale=100.0, octaves=6, persistence=None, lacunarity=None, base=None, features=5):
        """
        Constructor method
        """

        if features > 5:
            self.features = features
        else:
            print("Features have been set to default, please enter a number of features greater than 5")
            self.features = 5

        self.scale = scale
        self.octaves = octaves
        self.persistence = persistence
        self.lacunarity = lacunarity
        self.base = base

        # used to get corrected grayscale
        color = cm.get_cmap('gray')
        self.cmap = color.reversed()

    def excel_read(self, excel_file):
        """Used to gather names, motility values, colors, and color range from an excel file that is passed in by the


            :param excel_file: File path of the Excel File
            :type excel_file: string
            """

        xl = pd.ExcelFile(excel_file)

        # get the data frame
        df = xl.parse(0, skiprows=0)
        data = df.values

        # gather names
        self.names = data[:, 0].tolist()

        # gather motility
        self.motility_values = data[:, 1]

        # gather RGB colors and
        colors = data[:, 2]
        colors = np.asarray([colors[i].replace(' ', '') for i in range(colors.size)])
        colors = np.asarray([np.fromstring(colors[i], dtype=int, sep=',') for i in range(colors.size)])
        self.colors = np.divide(colors, 255)

        # get the color range
        self.color_range = data[:, 3]

    def ca_setup(self, motility_values=None, light_mode=False):
        """Used to setup the starting position of the deer within the simulated world, as well as takes in the
        motility values of the colors used within the generated world. The user may also determine if light mode should
        be used within the output of the generated world.

        :param motility_values: ndArray of floats that are the actual motility values, Must Match the number of
        features. Default is None.
        :type motility_values: ndArray, optional
        :param light_mode: Used to set the output of the RGBA. light_mode = True, RGBA pixels will become more
        transparent as the deer move to the same pixel. light_mode = False, RGBA pixels become less transparent
        as the deer move to the same pixel. Default is False.
        :type light_mode: bool, Optional
        """

        self.motility_values = motility_values

        # check motility_values
        if motility_values is not None:
            if motility_values.size != self.features:
                print("Motility values have been set to default. Please enter motility values that match the number "
                      "of features.")
                self.default()
        else:
            self.default()

        # create dictionary to speed up the program, 1500^2 resulted in a 77% speedup
        range_index = dict(zip(self.color_range, self.motility_values))
        self.range_dictionary = {float(key): range_index[key] for key in range_index}

        # starting values must not be on the edges of the world... for now
        self.starting_pos_x = np.random.randint(1, self.length - 1)
        self.starting_pos_y = np.random.randint(1, self.width - 1)

        # set current position to starting position
        self.current_pos_x = self.starting_pos_x
        self.current_pos_y = self.starting_pos_y

        # determines the output of the pathing
        self.light_mode = light_mode

    def color_append(self, pathing=True):
        """Returns an array of strings that hold the values of colors used within the world and to be used by the
        matplotlib legend. User can add in a brown path by setting pathing=True.

        :param pathing: Determines if the deer path should be added to the legened list
        :type pathing: bool, optional
        :return: A string array of the colors used within the world.
        :rtype: ndArray
        """

        if pathing:
            # add color of deer
            color = np.vstack([self.colors, [255 / 255, 0 / 255, 255 / 255, 1]])
            # delete the alpha out as we do not need it for the main blocks
            color = np.delete(color, 3, axis=1)
        else:
            color = np.delete(self.colors, 3, axis=1)
        # convert from rgb decimal to rgb hex
        colors = ['#{:02x}{:02x}{:02x}'.format(int(color[i][0] * 255), int(255 * color[i][1]), int(255 * color[i][2]))
                  for i in range(color.shape[0])]

        # return appending colors array
        return colors

    def color_world(self):
        """ Used to create the RGBA world as well as color range values of the world.
        """

        # create RGB from size of the heat map of the world
        self.world_color = np.zeros(self.world.shape + (4,))
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
        """ Creates the Perlin Noise given user dimensions as well as feature list that will be used to color the world.

        :param length: Sets the length of the world
        :type length: int, optional
        :param width: Sets the width of the world
        :type width: int, optional
        :param feature_list: Used to set the features of the world. Must be ndArray object that contains two ndArray's,
        that contain the RGB color array and the color range values.
        :type feature_list: object
        """

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
        """ Provides the default values for the features, colors, color_range, motility_values, and names. Default
        settings create a world with 5 features using the barren, water, pasture, spruce, and mixed confir values.
        """

        # default of 5 features
        self.features = 5
        # default of 5 colors
        self.colors = [[240 / 255, 230 / 255, 140 / 255, 1], [65 / 255, 105 / 255, 225 / 255, 1],
                       [34 / 255, 139 / 255, 34 / 255, 1],
                       [139 / 255, 137 / 255, 137 / 255, 1], [255 / 255, 250 / 255, 250 / 255, 1]]
        # default of 5 color ranges
        self.color_range = [-0.05, 0, 0.2, 0.36, 1]
        # default of 5 molitity values
        self.motility_values = [0.94, 1.02, 0.46, 2.33, 3.12]
        # default of 5 names
        self.names = ['barren', 'water', 'pasture', 'spruce', 'mixed confir']
        # default dictionary
        range_index = dict(zip(self.color_range, self.motility_values))
        self.range_dictionary = {float(key): range_index[key] for key in range_index}

    def feature_fill(self, feature_list=None):
        """ Checking function that takes the feature list array that converts the object type into separate arrays used
        for creating the color and the color range values. Will use the default values if the size of the color or color
        range arrays do not match.

        :param feature_list: Used to set the features of the world. Must be ndArray object that contains two ndArray's,
        that contain the RGB color array and the color range values.
        :type feature_list: object
        """

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

        else:  # yellow,          blue,            green,         gray,           white
            self.colors = [[240, 230, 140], [65, 105, 225], [34, 139, 34], [139, 137, 137], [255, 250, 250]]
            self.color_range = [-0.05, 0, 0.2, 0.36, 1]

    def moore_neighborhood(self, square):
        """ Uses a moore neighborhood to determine which new position to move the deer based off of the average of the
        values found within the moore neighborhood. Each outer square of the moore neighborhood is checked against the
        current lowest motility value plus a random normal distribution value.

        :param square: Moore neighborhood array of floats, must be a 3x3 ndArray.
        :type square: ndArray
        """
        # default for current motility
        current_motility = 100.0
        self.next_position_y = 0
        self.next_position_x = 0

        average = np.average(square)

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

    def view_finder(self, square):
        """ This function simulates the viewing of the deer as it uses an extended moore neighborhood to help find the
        best choice of movement. Each of the moore neighborhood values will be the average of the surrounding values,
        which is then passed into the moore_neighborhood function to find the next position.

        :param square: Extended moore neighborhood ndArray of floats, must be 7 by 7.
        :type square: ndArray
        """

        # Python memory hack making me do this :'/
        square = np.copy(square)

        # convert from the heat map values to the respective motility values
        for i in range(len(square[0])):
            for j in range(len(square[0])):
                square[i][j] = self.range_dictionary[square[i][j]]

        # front view
        front_view = np.sum(square[0:2, 0:7])
        square[2][3] = front_view / 14

        # front left view
        front_left_view = np.sum(square[0, 0:5]) + np.sum(square[1, 0:4]) + np.sum(square[2, 0:2]) + np.sum(
            square[3, 0:2]) + square[4][0]
        square[2][2] = front_left_view / 14

        # left view
        left_view = np.sum(square[0:7, 0:2])
        square[3][2] = left_view / 14

        # front right view
        front_right_view = np.sum(square[0, 2:7]) + np.sum(square[1, 3:7]) + np.sum(square[2, 5:7]) + np.sum(
            square[3, 5:7]) + square[4][6]
        square[2][4] = front_right_view / 14

        # right view
        right_view = np.sum(square[0:7, 5:7])
        square[3][4] = right_view / 14

        # back right view
        back_right_view = np.sum(square[6, 2:7]) + np.sum(square[5, 3:7]) + np.sum(square[4, 5:7]) + np.sum(
            square[3, 5:7]) + square[2][6]
        square[4][4] = back_right_view / 14

        # back left view
        back_left_view = np.sum(square[6, 0:5]) + np.sum(square[5, 0:4]) + np.sum(square[4, 0:2]) + np.sum(
            square[3, 0:2]) + square[2][0]
        square[4][2] = back_left_view / 14

        # back view
        back_view = np.sum(square[5:7, 0:7])
        square[4][3] = back_view / 14

        # used to move
        movement = square[2:5, 2:5]

        self.moore_neighborhood(movement)

    def live_updater(self, buffer, t, colors, motility, prev_pos_x, prev_pos_y):
        """ Provides the ability to update the matplotlib output given the current position of the deer. Calls the
        alpha change function to update the current value of the RGBA pixel position.

        :param buffer: Copy of the RGBA world used to show current position of the deer.
        :type buffer: ndArray
        :param t: Current time or iteration of the simulation.
        :type t: int
        :param colors: Color array with the deer being added into the simulation.
        :type colors: ndArray
        :param motility: Motility values being used within the world.
        :type motility: ndArray
        :param prev_pos_x: Previous position of the deer on the x-axis
        :type prev_pos_x: int
        :param prev_pos_y: Previous position of the deer on the y-axis
        :type prev_pos_y: int
        """
        # needed for video
        plt.clf()

        if self.current_pos_x == self.width * -1 - 1:
            self.current_pos_x = np.remainder(self.current_pos_x, self.width)

        if self.current_pos_y == self.length * -1 - 1:
            self.current_pos_y = np.remainder(self.current_pos_y, self.length)

        # Marron RGB for previous spots for deer [128, 0, 0]
        # buffer[prev_pos_x][prev_pos_y] = [128/255, 0/255, 0/255, 1]
        buffer[prev_pos_x][prev_pos_y] = self.world_color[prev_pos_x][prev_pos_y]

        # use pink for current position [255, 0, 255]
        buffer[self.current_pos_x][self.current_pos_y] = [255 / 255, 0 / 255, 255 / 255, 1]

        # add in location and iteration/time passed
        string = "t {} \n (x,y): ({},{})\n s {} o {} p {} l {} b {} f {}".format(t,
                np.remainder(self.current_pos_x, self.width), np.remainder(self.current_pos_y, self.length), self.scale,
                self.octaves, np.round(self.persistence, 3), np.round(self.lacunarity, 3), self.base, self.features
                                                                                 )

        # plot the updated time/iteration and current coordinates of the deer
        plt.title(string)

        # plot the buffer world image
        plt.imshow(buffer)

        # create the legend box with names of each color as well the as the motility values
        patches = [mpatches.Patch(color=colors[i], label='{:^5} {:>10}'.format(self.names[i], motility[i])) for i in
                   range(len(self.names))]

        # plot the legend box
        plt.legend(handles=patches, bbox_to_anchor=(1.05, 0.0, 0.3, 1), loc=2, borderaxespad=0.1)  # , mode='expand')

        # show the image
        plt.show()

        # use to determine the time between each iteration
        plt.pause(0.3)

    def pathing(self, time, live_update=False, names=None):
        """ Given a set amount of iterations this function simulates the deer within the generated world. The use of
        live_update allows the user to see the deer move after each iteration. User must provide a list of the names
        that match the number of features.

        :param time: Total amount of iterations to run the simulation
        :type time: int
        :param live_update: Determines if the simulation should be shown live
        :type live_update: bool
        :param names: List of names for the features. Must match the number of features
        :type names: ndArray
        """
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

            # update the RGBA world with current position of the world
            self.world_color[prev_pos_x][prev_pos_y] = self.alpha_change(self.world_color[prev_pos_x][prev_pos_y])

            # find the square that the deer is considering based off of the current position
            square_choice = self.edge_check(x=self.current_pos_x, y=self.current_pos_y)

            # use Moore neighborhood to select the next position
            self.view_finder(square_choice)

            # previous position
            prev_pos_x = self.current_pos_x
            prev_pos_y = self.current_pos_y

            # update current position to future position
            self.current_pos_x += self.next_position_x
            self.current_pos_y += self.next_position_y

            self.current_pos_x = np.remainder(self.current_pos_x, self.length)
            self.current_pos_y = np.remainder(self.current_pos_y, self.width)

        if live_update:
            # used to end the video format
            plt.ioff()
            plt.close()

        self.output_world(self.world_color)
        self.path_map()

    def string_names(self):
        """ Appends the deer to the name array and returns a list of strings of the motility values.

        :return motility: List of strings of the motility values
        :rtype motility: list

        """

        # add in the path and deer to the name list
        self.names += ['deer']

        # converts to a list of strings for motility values
        motility = list(map(str, self.motility_values))

        # add in NA for deer and path, as they do not have their own values on the value map
        motility += ['NA']

        # return the motility string
        return motility

    def output_world(self, world, gray=False):
        """ Outputs the generated world with the option of having it in a gray scale.

        :param world: Array of RGBA values that represent the world
        :type world: ndArray
        :param gray: Determines if the world should be shown in gray scale.
        :type gray: bool
        """
        plt.figure()

        if gray is True:
            print("Grayscale")
            plt.imshow(world, cmap=self.cmap)
        else:
            print("RGBA")
            plt.imshow(world)
        plt.show()

    def alpha_change(self, pixel):
        """ Changes the current pixel by increasing or decreasing the alpha value depending if light mode has been
        activated. Returns the changed pixel value

        :param pixel: Current pixel that needs to be adjusted as a ndArray with 4 float elements
        :type pixel: ndArray
        :return rgba: Updated pixel value as a ndArray with 4 float elements
        :rtype rgba: ndArray
        """

        # check light mode
        if self.light_mode:
            rgba = [pixel[0], pixel[1], pixel[2], 0.95 * pixel[3]]
        else:
            if pixel[3] <= 0.95:
                rgba = [pixel[0], pixel[1], pixel[2], 1.05 * pixel[3]]
            else:
                rgba = [pixel[0], pixel[1], pixel[2], 1]

        return rgba

    def path_map(self):
        """ Shows the path taken by the simulated deer to the user.
        """

        self.color_append(False)

        for i in range(self.length):
            for j in range(self.width):

                change_apperent = self.world_color[i][j]

                if self.light_mode:
                    change_apperent[3] = np.abs(1 - change_apperent[3])
                    self.world_color[i][j] = change_apperent
                else:
                    if change_apperent[3] <= 0.1:
                        change_apperent[3] = 0.0

        plt.figure()

        colors = self.color_append(False)

        # create the legend box with names of each color as well the as the motility values
        patches = [mpatches.Patch(color=colors[i], label='{:^5} {:>10}'.format(self.names[i], self.motility_values[i]))
                   for i in
                   range(len(self.colors))]

        # plot the legend box
        plt.legend(handles=patches, bbox_to_anchor=(1.05, 0.0, 0.3, 1), loc=2, borderaxespad=0.1)  # , mode='expand')

        plt.imshow(self.world_color)

        plt.show()

    def break_it(self):
        """ Used to test all positions of the world using the edge check function. Writes to a file "breakit.txt".
        """

        # open a file to check output checks
        file1 = open("breakit.txt", "w")
        for i in range(self.width):
            for j in range(self.length):
                check_grid = self.edge_check(x=i, y=j)
                file1.write("({}, {} \n)".format(i, j))
                if check_grid.shape[0] != 7:
                    file1.write("shape[0] != 7 \n")
                if check_grid.shape[1] != 7:
                    file1.write("shape[1] != 7 \n")

        file1.close()

    def edge_check(self, x, y):
        """ Determines which values of the world to use within the 7 by 7 viewing array given the current x and y
        positions of the deer.

        :param x: Current x position of the deer
        :type x: int
        :param y: Current y position of the deer
        :type y: Current y position of the deer
        :return square: 7 by 7 extended moore neighborhood with the current deer position index at the middle of the
        array
        :rtype square: ndArray
        """
        # check to see if indices are outside of the array
        if x + 4 > self.length and y + 4 > self.width:
            square = self.bottom_right_corner(x, y)
        elif x - 3 < 0 and y - 3 < 0:
            square = self.upper_left_corner(x, y)
        elif x + 4 > self.length and y - 3 < 0:
            square = self.bottom_left_corner(x, y)
        elif x - 3 < 0 and y + 4 > self.width:
            square = self.upper_right_corner(x, y)
        elif x + 4 > self.length:
            square = self.bottom(x, y)
        elif x - 3 < 0:
            square = self.upper(x, y)
        elif y + 4 > self.width:
            square = self.right(x, y)
        elif y - 3 < 0:
            square = self.left(x, y)
        else:
            square = self.ca_world[x - 3:x + 4, y - 3:y + 4]

        return square

    def right(self, x, y):
        """ Finds the overlap given the current position is on the right side of the world array. Returns the 7 by 7
        array used by the moore neighborhood.

        :param x: Current x position of the deer
        :type x: int
        :param y: Current y position of the deer
        :type y: int
        :return : 7 by 7 extended moore neighborhood with the current deer position index at the middle of the
        array
        :rtype : ndArray
        """
        # get the left half
        left_half = self.ca_world[x-3:x+4, y-3:self.length]
        # get the right half
        right_half = self.ca_world[x-3:x+4, 0:np.remainder(y+4, self.length)]
        # return the total
        return np.hstack((left_half, right_half))

    def left(self, x, y):
        """ Finds the overlap given the current position is on the left side of the world array. Returns the 7 by 7
        array used by the moore neighborhood.

        :param x: Current x position of the deer
        :type x: int
        :param y: Current y position of the deer
        :type y: int
        :return : 7 by 7 extended moore neighborhood with the current deer position index at the middle of the
        array
        :rtype : ndArray
        """

        # get the right half
        right_half = self.ca_world[x-3:x+4, 0:y+4]
        # get the left half
        left_half = self.ca_world[x-3:x+4, np.remainder(y-3, self.length):self.length]
        # return the total
        return np.hstack((left_half, right_half))

    def upper(self, x, y):
        """ Finds the overlap given the current position is on the top side of the world array. Returns the 7 by 7
                array used by the moore neighborhood.

        :param x: Current x position of the deer
        :type x: int
        :param y: Current y position of the deer
        :type y: int
        :return : 7 by 7 extended moore neighborhood with the current deer position index at the middle of the
        array
        :rtype : ndArray
        """

        # get lower half
        lower_half = self.ca_world[0:x+4, y-3:y+4]
        # get the upper half
        upper_half = self.ca_world[np.remainder(x-3, self.width):self.width, y-3:y+4]
        # return the total
        return np.vstack((upper_half, lower_half))

    def bottom(self, x, y):
        """ Finds the overlap given the current position is on the bottom side of the world array. Returns the 7 by 7
        array used by the moore neighborhood.

        :param x: Current x position of the deer
        :type x: int
        :param y: Current y position of the deer
        :type y: int
        :return : 7 by 7 extended moore neighborhood with the current deer position index at the middle of the
        array
        :rtype : ndArray
        """

        # gather upper half
        upper_half = self.ca_world[x-3:self.width, y-3:y+4]
        # gather lower half
        lower_half = self.ca_world[0:np.remainder(x+4, self.width), y-3:y+4]
        # return the upper and lower
        return np.vstack((upper_half, lower_half))

    def upper_right_corner(self, x, y):
        """ Finds the overlap given the current position is on the upper right corner of the world array.
        Returns the 7 by 7 array used by the moore neighborhood.

        :param x: Current x position of the deer
        :type x: int
        :param y: Current y position of the deer
        :type y: int
        :return : 7 by 7 extended moore neighborhood with the current deer position index at the middle of the
        array
        :rtype : ndArray
        """

        # get lower right quarter
        lower_right_quarter = self.ca_world[np.remainder(x-3, self.width):self.width, 0:np.remainder(y+4, self.length)]
        # get upper right quarter
        upper_right_quarter = self.ca_world[0:x+4, 0:np.remainder(y+4, self.length)]
        # right side
        right_half = np.vstack((upper_right_quarter, lower_right_quarter))
        # lower left quarter
        lower_left_quarter = self.ca_world[0:x+4, y-3:self.length]
        # upper left quarter
        upper_left_quarter = self.ca_world[np.remainder(x-3, self.width):self.width, y-3:self.length]
        # left side
        left_half = np.vstack((upper_left_quarter, lower_left_quarter))
        # combine the sides together
        return np.hstack((left_half, right_half))

    def bottom_left_corner(self, x, y):
        """ Finds the overlap given the current position is on the bottom left corner of the world array.
        Returns the 7 by 7 array used by the moore neighborhood.

        :param x: Current x position of the deer
        :type x: int
        :param y: Current y position of the deer
        :type y: int
        :return : 7 by 7 extended moore neighborhood with the current deer position index at the middle of the
        array
        :rtype : ndArray
        """

        # get the upper right quarter
        upper_right_quarter = self.ca_world[x - 3:self.width, 0:y + 4]
        # get the bottom right quarter
        lower_right_quarter = self.ca_world[0:np.remainder(x + 4, self.width), 0:y+4]
        # right half
        right_half = np.vstack((upper_right_quarter, lower_right_quarter))
        # bottom left quarter
        lower_left_quarter = self.ca_world[0:np.remainder(x+4, self.width), np.remainder(y-3, self.length): self.length]
        # get the upper left quarter
        upper_left_quarter = self.ca_world[x - 3:self.width, np.remainder(y-3, self.length):self.length]
        # create the left half
        left_half = np.vstack((upper_left_quarter, lower_left_quarter))
        # completed 7 by 7 square
        return np.hstack((left_half, right_half))

    def bottom_right_corner(self, x, y):
        """ Finds the overlap given the current position is on the bottom right corner of the world array.
        Returns the 7 by 7 array used by the moore neighborhood.

        :param x: Current x position of the deer
        :type x: int
        :param y: Current y position of the deer
        :type y: int
        :return : 7 by 7 extended moore neighborhood with the current deer position index at the middle of the
        array
        :rtype : ndArray
        """

        # get the upper left quarter
        upper_left_quarter = self.ca_world[x - 3:self.width, y - 3:self.length]
        # bottom left quarter
        lower_left_quarter = self.ca_world[0:np.remainder(x + 4, self.width), y - 3:self.length]
        # create the left half
        left_half = np.vstack((upper_left_quarter, lower_left_quarter))
        # get the upper right quarter
        upper_right_quarter = self.ca_world[x - 3:self.width, 0:np.remainder(y + 4, self.length)]
        # get the bottom right quarter
        lower_right_quarter = self.ca_world[0:np.remainder(x + 4, self.width), 0:np.remainder(y + 4, self.length)]
        # right half
        right_half = np.vstack((upper_right_quarter, lower_right_quarter))
        # completed 7 by 7 square
        return np.hstack((left_half, right_half))

    def upper_left_corner(self, x, y):
        """ Finds the overlap given the current position is on the upper left corner of the world array.
        Returns the 7 by 7 array used by the moore neighborhood.

        :param x: Current x position of the deer
        :type x: int
        :param y: Current y position of the deer
        :type y: int
        :return : 7 by 7 extended moore neighborhood with the current deer position index at the middle of the
        array
        :rtype : ndArray
        """

        # get lower right quarter
        lower_right_quarter = self.ca_world[0:x + 4, 0:y + 4]
        # get upper right quarter
        upper_right_quarter = self.ca_world[np.remainder(x - 3, self.width):self.width, 0:y + 4]
        # right side
        right_half = np.vstack((upper_right_quarter, lower_right_quarter))
        # lower left quarter
        lower_left_quarter = self.ca_world[0:x + 4, np.remainder(y - 3, self.length): self.length]
        # upper left quarter
        upper_left_quarter = self.ca_world[np.remainder(x - 3, self.width):self.width,
                                       np.remainder(y - 3, self.length):self.length]
        # left side
        left_half = np.vstack((upper_left_quarter, lower_left_quarter))
        # combine the sides together
        return np.hstack((left_half, right_half))

    def test1(self):
        square = np.copy(self.ca_world)

        start = time.time()
        for i in range(self.width):
            for j in range(self.length):
                for k in range(self.features):
                    if square[i][j] == self.color_range[k]:
                        square[i][j] = self.motility_values[k]
        end = time.time()
        print(end - start)

    def test2(self):
        range_index = {float(key): self.range_dictionary[key] for key in self.range_dictionary}
        square = np.copy(self.ca_world)

        start = time.time()
        for i in range(self.width):
            for j in range(self.length):
                square[i][j] = range_index[square[i][j]]
        end = time.time()
        print(end - start)