import math

import matplotlib
import numpy as np
from matplotlib.image import imread
from matplotlib.pyplot import imshow
from matplotlib.image import imsave
from enum import Enum

from numpy import ndarray


class ColorModel(Enum):
    rgb = 0
    hsv = 1
    hsi = 2
    hsl = 3
    gray = 4  # picture 2d


class BaseImage:
    data: np.ndarray  # tensor that stores the pixels of image
    color_model: ColorModel  # attribute that stores current color model  of image

    def __init__(self, path: str) -> None:
        """
        initializer that loads an image into the data attribute based on a path
        """
        self.data = imread(path)
        self.color_model = ColorModel.rgb

    def save_img(self, path: str) -> None:
        """
        method that saves image stored in attribute date to file
        """
        imsave(path, self.data)

    def show_img(self) -> None:
        """
        method that shows image stored in attribute date
        """
        imshow(self.data)
        matplotlib.pyplot.show()

    #
    # BaseImage? np.ndarray?
    #
    def get_layer(self, layer_id: int) -> 'BaseImage':
        """
        method that returns layer with the indicated index
        """
        return self.data[:, :, layer_id]

    def to_hsv(self) -> 'BaseImage':
    # def to_hsv(self) -> ndarray:
        """
        metoda dokonujaca konwersji obrazu w atrybucie data do modelu hsv
        metoda zwraca nowy obiekt klasy Baseimage zawierajacy obraz w docelowym modelu barw

        method that converts image in attribute date to hsv
        # method returns new object of class BaseImage that stores image in hsv color model
        """
        # if !(rgb):
        #   to_rgb
        # dtype='uint16'
        # r = np.array(self.data[:, :, 0])
        # g = np.array(self.data[:, :, 1])
        # b = np.array(self.data[:, :, 2])
        # maximum = np.maximum.reduce([r, g, b], dtype='float')
        # minimum = np.minimum.reduce([r, g, b], dtype='float')
        # v = maximum / 255
        # s = np.where(maximum > 0, 1 - minimum / maximum, 0)

        # h = np.where(g >= b,
        #              np.cos((r - 0.5 * g - 0.5 * b) / (np.sqrt(r ** 2 + g ** 2 + b ** 2 - r * g - r * b - g * b))) ** (
        #                  -1),
        #              360 - np.cos(
        #                  (r - 0.5 * g - 0.5 * b) / (np.sqrt(r ** 2 + g ** 2 + b ** 2 - r * g - r * b - g * b))) ** (-1))

        # rf = r.flatten()
        # gf = g.flatten()
        # bf = b.flatten()
        # h = np.empty(0)
        # for r, g, b in zip(rf, gf, bf):
        #     if r == 0 & g == 0 & b == 0:
        #         np.append(h, 0)
        #     elif g >= b:
        #         np.append(h, (math.cos(
        #             (r - 0.5 * g - 0.5 * b) / (math.sqrt(r ** 2 + g ** 2 + b ** 2 - r * g - r * b - g * b))) ** (-1)))
        #     else:
        #         np.append(h, 360 - math.cos(
        #             (r - 0.5 * g - 0.5 * b) / (math.sqrt(r ** 2 + g ** 2 + b ** 2 - r * g - r * b - g * b))) ** (-1))
        # h = h.reshape(r.shape)

        # return np.dstack((h, s, v))
        self.data = self.data.copy()
        for pixel in self.data:
            for color in pixel:
                red, green, blue = color[0], color[1], color[2]
                M = max(red, green, blue)
                m = min(red, green, blue)
                V = M / 255
                S = 1 - m / M if M > 0 else 0
                addition = red ** 2 + green ** 2 + blue ** 2
                subtraction = int(red) * int(green) - int(red) * int(blue) - int(green) * int(blue)
                additionMinusSubtraction = addition - subtraction
                if red == 0 & green == 0 & blue == 0:
                    H = 0
                elif green >= blue:
                    H = np.cos((red - 0.5 * green - 0.5 * blue) / np.sqrt(additionMinusSubtraction)) ** (-1)
                else:
                    H = 360 - np.cos((red - green / 2 - blue / 2) / np.sqrt(additionMinusSubtraction)) ** (-1)
                color[0] = H * 100
                color[1] = S * 100
                color[2] = V * 100

        self.color_model = ColorModel.hsv
        return self



    def to_hsi(self) -> 'BaseImage':
        """
        metoda dokonujaca konwersji obrazu w atrybucie data do modelu hsi
        metoda zwraca nowy obiekt klasy image zawierajacy obraz w docelowym modelu barw
        """
        pass

    def to_hsl(self) -> 'BaseImage':
        """
        metoda dokonujaca konwersji obrazu w atrybucie data do modelu hsl
        metoda zwraca nowy obiekt klasy image zawierajacy obraz w docelowym modelu barw
        """
        pass

    def to_rgb(self) -> 'BaseImage':
        """
        metoda dokonujaca konwersji obrazu w atrybucie data do modelu rgb
        metoda zwraca nowy obiekt klasy image zawierajacy obraz w docelowym modelu barw
        """
        pass


# TESTS

# img_arr = imread('data/lena.jpg')

# imshow(img_arr)
# matplotlib.pyplot.show()

image_test = BaseImage('data/lena.jpg')
image_test.show_img()

image_test.to_hsv()
image_test.show_img()


# testFunkcji = image_test.to_hsv()
# print(testFunkcji)
# imshow(testFunkcji)
# matplotlib.pyplot.show()

# rgb = np.array([[[255, 1, 2], [200, 3, 4]], [[5, 6, 122], [0, 0, 0]]])
# r = rgb[:, :, 0]
# g = rgb[:, :, 1]
# b = rgb[:, :, 2]
# print(r)
# print(g)
# print(b)
# print()

# r = np.array([[2, 2], [0, 1]], dtype='float')
# g = np.array([[2, 2], [0, 1]], dtype='float')
# b = np.array([[1, 3], [4, 0]], dtype='float')

# print(r)
# print(g)
# print(b)
# print()

# rf = r.flatten()
# gf = g.flatten()
# bf = b.flatten()
# h = []
# for r, g, b in zip(rf, gf, bf):
#     if r == 0 & g == 0 & b == 0:
#         h.append(0)
#     elif g >= b:
#         h.append(math.cos((r - 0.5 * g - 0.5 * b) / (math.sqrt(r**2 + g**2 + b**2 - r*g - r*b - g*b))) ** (-1))
#     else:
#         h.append(360 - math.cos((r - 0.5 * g - 0.5 * b) / (math.sqrt(r**2 + g**2 + b**2 - r*g - r*b - g*b))) ** (-1))
# print(h)
#
# h2 = np.where(g >= b,
#              np.cos((r - 0.5 * g - 0.5 * b) / (np.sqrt(r ** 2 + g ** 2 + b ** 2 - r * g - r * b - g * b))) ** (
#                  -1),
#              360 - np.cos(
#                  (r - 0.5 * g - 0.5 * b) / (np.sqrt(r ** 2 + g ** 2 + b ** 2 - r * g - r * b - g * b))) ** (-1))
# print(h2)


