import matplotlib
import numpy as np
from matplotlib import pyplot as plt

from matplotlib.image import imread
from matplotlib.pyplot import imshow
from matplotlib.image import imsave

from enum import Enum
from typing import Any
from math import pi


class ColorModel(Enum):
    rgb = 0
    hsv = 1
    hsi = 2
    hsl = 3
    gray = 4  # picture 2d
    sepia = 5


class BaseImage:
    data: np.ndarray  # tensor that stores the pixels of image
    color_model: ColorModel  # attribute that stores current color model  of image

    def __init__(self, data: Any, color_model: ColorModel) -> None:
        """
        initializer that loads an image into the data attribute (based on a path too)
        """
        if isinstance(data, str):
            self.data = imread(data)
        else:
            self.data = data
        self.color_model = color_model

    def save_img(self, path: str) -> None:
        """
        method that saves image stored in attribute date to file
        """
        imsave(path, self.data)

    def show_img(self) -> None:
        """
        method that shows image stored in attribute date
        """
        if self.color_model == ColorModel.gray:
            plt.imshow(self.data, cmap='gray', vmin=0, vmax=255)
            matplotlib.pyplot.show()
            return
        imshow(self.data)
        matplotlib.pyplot.show()

    def show_img_in_rgb_range(self) -> None:
        """
        method that shows image stored in attribute date(when it's HSV/HSL/HSI color model)
        in the range of numbers available in imshow()
        """
        if self.color_model == ColorModel.hsv or self.color_model == ColorModel.hsi or\
                self.color_model == ColorModel.hsl:
            layer1, layer2, layer3 = self.get_layers()
            layer1 = layer1 / 360
            image_in_range = np.dstack((layer1, layer2, layer3))
            imshow(image_in_range)
            plt.show()

    def get_layer(self, layer_id: int) -> np.ndarray:
        """
        method that returns layer with the indicated index
        """
        return self.data[:, :, layer_id]

    def get_layers(self) -> np.ndarray:
        """
        method that returns layers
        """
        return np.squeeze(np.dsplit(self.data, self.data.shape[-1]))

    def calculate_H(self, r, g, b):
        """
        method calculating H for HSV, HSI and HSL color models
        """
        calc = np.power(r.astype(int), 2) + np.power(g.astype(int), 2) + np.power(b.astype(int), 2) - (
                    r.astype(int) * g.astype(int)) - (r.astype(int) * b.astype(int)) - (g.astype(int) * b.astype(int))
        calc_sqrt = np.sqrt(calc)
        calc_sqrt[calc_sqrt == 0] = 1
        H = np.where(g >= b, np.arccos((r - (g / 2) - (b / 2)) / calc_sqrt) * 180 / pi,
                     360 - np.arccos(((r - (g / 2) - (b / 2)) / calc_sqrt)) * 180 / pi)
        return H

    def to_hsv(self) -> 'BaseImage':
        """
        method that converts image in attribute date to hsv
        method returns new object of class BaseImage that stores image in hsv color model
        """
        if self.color_model != ColorModel.rgb:
            raise Exception("Color model must be RGB")
        r, g, b = self.get_layers()
        MAX = np.max([r, g, b], axis=0)
        MAX[MAX == 0] = 0.001
        MIN = np.min([r, g, b], axis=0)
        H = self.calculate_H(r, g, b)
        S = np.where(MAX == 0, 0, (1 - MIN/MAX))
        V = MAX / 255
        return BaseImage(np.dstack((H, S, V)), ColorModel.hsv)

    def to_hsi(self) -> 'BaseImage':
        """
        method that converts image in attribute date to hsi
        method returns new object of class BaseImage that stores image in hsi color model
        """
        if self.color_model != ColorModel.rgb:
            raise Exception("Color model must be RGB")
        R, G, B = np.float32(self.get_layers())
        sum = R + G + B
        sum[sum == 0] = 255
        r = R / sum
        g = G / sum
        b = B / sum
        MIN = np.min([r, g, b], axis=0)
        H = self.calculate_H(R, G, B)
        I = (R + G + B) / (3 * 255)
        S = 1 - 3 * MIN
        return BaseImage(np.dstack((H, S, I)), ColorModel.hsi)

    def to_hsl(self) -> 'BaseImage':
        """
        method that converts image in attribute date to hsl
        method returns new object of class BaseImage that stores image in hsl color model
        """
        if self.color_model != ColorModel.rgb:
            raise Exception("Color model must be RGB")
        r, g, b = self.get_layers()
        MAX = np.max([r, g, b], axis=0)
        MIN = np.min([r, g, b], axis=0)
        D = (MAX - MIN)/255
        H = self.calculate_H(r, g, b)
        L = (0.5 * (MAX.astype(int) + MIN.astype(int))).astype(int) / 255
        calc = (1 - abs(2 * L - 1))
        calc[calc == 0] = 0.0001
        S = np.where(L > 0, D / calc, 0)
        S[S > 1] = 1
        S[S < 0] = 0.001
        return BaseImage(np.dstack((H, S, L)), ColorModel.hsl)

    def hsv_to_rgb(self) -> 'BaseImage':
        """
        method that converts hsv image in attribute date to rgb
        method returns new object of class BaseImage that stores image in rgb color model
        """
        H, S, V = self.get_layers()
        C = V * S
        X = C * (1 - abs(((H / 60) % 2)-1))
        m = V - C
        r = np.where(H >= 300, C, np.where(H >= 240, X, np.where(H >= 120, 0, np.where(H >= 60, X, C))))
        g = np.where(H >= 240, 0, np.where(H >= 180, X, np.where(H >= 60, C, X)))
        b = np.where(H >= 300, X, np.where(H >= 180, C, np.where(H >= 120, X, 0)))
        r = (r + m) * 255
        g = (g + m) * 255
        b = (b + m) * 255
        # Normalize r g b
        g[g > 255] = 255
        b[b > 255] = 255
        r[r > 255] = 255
        r[r < 0] = 0
        g[g < 0] = 0
        b[b < 0] = 0
        return BaseImage(np.dstack((r, g, b)).astype(np.uint8), ColorModel.rgb)

    def hsi_to_rgb(self) -> 'BaseImage':
        """
        method that converts hsi image in attribute date to rgb
        method returns new object of class BaseImage that stores image in rgb color model
        """
        H, S, I = self.get_layers()
        h = H * np.pi / 180
        s = S
        i = I
        rows = self.data.shape[0]
        columns = self.data.shape[1]
        r = np.zeros((rows, columns))
        g = np.zeros((rows, columns))
        b = np.zeros((rows, columns))
        for k in range(rows):
            for j in range(columns):
                if h[k, j] < np.pi * 2 / 3:
                    x = i[k, j] * (1 - s[k, j])
                    y = i[k, j] * (1 + s[k, j] * np.cos(h[k, j]) / np.cos(np.pi / 3 - h[k, j]))
                    z = 3 * i[k, j] - (x + y)
                    r[k, j] = y
                    g[k, j] = z
                    b[k, j] = x
                if np.pi * 2 / 3 <= h[k, j] < np.pi * 4 / 3:
                    h[k, j] = h[k, j] - np.pi * 2 /3
                    x = i[k, j] * (1 - s[k, j])
                    y = i[k, j] * (1 + s[k, j] * np.cos(h[k, j]) / np.cos(np.pi / 3 - h[k, j]))
                    z = 3 * i[k, j] - (x + y)
                    r[k, j] = x
                    g[k, j] = y
                    b[k, j] = z
                if np.pi * 4 / 3 < h[k, j] < np.pi * 2:
                    h[k, j] = h[k, j] - np.pi * 4 / 3
                    x = i[k, j] * (1 - s[k, j])
                    y = i[k, j] * (1 + s[k, j] * np.cos(h[k, j]) / np.cos(np.pi / 3 - h[k, j]))
                    z = 3 * i[k,j] - (x + y)
                    r[k, j] = z
                    g[k, j] = x
                    b[k, j] = y
        # [0...1] to [0...255]
        r[r > 1] = 1
        g[g > 1] = 1
        b[b > 1] = 1
        r = r * 255
        g = g * 255
        b = b * 255
        #Normalize r g b
        g[g > 255] = 255
        b[b > 255] = 255
        r[r > 255] = 255
        r[r < 0] = 0
        g[g < 0] = 0
        b[b < 0] = 0
        return BaseImage(np.dstack((r, g, b)).astype(np.uint16), ColorModel.rgb)

    def hsl_to_rgb(self) -> 'BaseImage':
        """
        method that converts hsl image in attribute date to rgb
        method returns new object of class BaseImage that stores image in rgb color model
        """
        H, S, L = self.get_layers()
        d = S * (1 - abs(2 * L - 1))
        MIN = 255 * (L - 0.5 * d)
        x = d * (1 - abs(H / 60 % 2 - 1))
        r = np.where(H >= 300, 255 * d + MIN,
                     np.where(H >= 240, 255 * x + MIN,
                              np.where(H >= 180, MIN,
                                       np.where(H >= 120, MIN,
                                                np.where(H >= 60, 255 * x + MIN, 255 * d + MIN)))))
        g = np.where(H >= 300, MIN,
                     np.where(H >= 240, MIN,
                              np.where(H >= 180, 255 * x + MIN,
                                       np.where(H >= 120, 255 * d + MIN,
                                                np.where(H >= 60, 255 * d + MIN, 255 * x + MIN)))))
        b = np.where(H >= 300, 255 * x + MIN,
                     np.where(H >= 240, 255 * d + MIN,
                              np.where(H >= 180, 255 * d + MIN,
                                       np.where(H >= 120, 255 * x + MIN, MIN))))
        # Normalize r g b
        g[g > 255] = 255
        b[b > 255] = 255
        r[r > 255] = 255
        r[r < 0] = 0
        g[g < 0] = 0
        b[b < 0] = 0
        return BaseImage(np.dstack((r, g, b)).astype(np.int16), ColorModel.rgb)

    def to_rgb(self) -> 'BaseImage':
        """
        method that converts image in attribute date to rgb
        method returns new object of class BaseImage that stores image in rgb color model
        """
        match self.color_model:
            case ColorModel.hsv:
                return self.hsv_to_rgb()
            case ColorModel.hsi:
                return self.hsi_to_rgb()
            case ColorModel.hsl:
                return self.hsl_to_rgb()
            case _:
                return self
