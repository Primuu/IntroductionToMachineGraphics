import matplotlib
import numpy as np

from matplotlib.image import imread
from matplotlib.pyplot import imshow
from matplotlib.image import imsave
from enum import Enum
from typing import Any


class ColorModel(Enum):
    rgb = 0
    hsv = 1
    hsi = 2
    hsl = 3
    gray = 4  # picture 2d


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
        imshow(self.data)
        matplotlib.pyplot.show()

    def get_layer(self, layer_id: int) -> 'np.ndarray':
        """
        method that returns layer with the indicated index
        """
        return self.data[:, :, layer_id]

    def to_hsv(self) -> 'BaseImage':
        """
        method that converts image in attribute date to hsv
        method returns new object of class BaseImage that stores image in hsv color model
        """
        if self.color_model != ColorModel.rgb:
            raise Exception("Color model must be RGB")
        red_layer = self.get_layer(0) / 255.0
        green_layer = self.get_layer(1) / 255.0
        blue_layer = self.get_layer(2) / 255.0
        M = np.maximum.reduce([red_layer, green_layer, blue_layer])
        m = np.minimum.reduce([red_layer, green_layer, blue_layer])
        V = M / 255.0
        S = np.where(M > 0, 1 - m / M, 0)
        under_square_expression = np.power(red_layer, 2) + np.power(green_layer, 2) + np.power(blue_layer, 2) - \
                                  red_layer * green_layer - red_layer * blue_layer - green_layer * blue_layer
        H = np.where(green_layer >= blue_layer,
                     np.cos((red_layer - 0.5 * green_layer - 0.5 * blue_layer) / np.sqrt(under_square_expression))
                     ** (-1),
                     360 - np.cos((red_layer - 0.5 * green_layer - 0.5 * blue_layer) / np.sqrt(under_square_expression))
                     ** (-1))

        return BaseImage(np.dstack((H, S, V)), ColorModel.hsv)

    def to_hsi(self) -> 'BaseImage':
        """
        method that converts image in attribute date to hsi
        method returns new object of class BaseImage that stores image in hsi color model
        """
        if self.color_model != ColorModel.rgb:
            raise Exception("Color model must be RGB")
        red_layer = self.get_layer(0) / 255.0
        green_layer = self.get_layer(1) / 255.0
        blue_layer = self.get_layer(2) / 255.0
        M = np.maximum.reduce([red_layer, green_layer, blue_layer])
        m = np.minimum.reduce([red_layer, green_layer, blue_layer])
        I = (red_layer + green_layer + blue_layer) / 3.0
        S = np.where(M > 0, 1 - m / M, 0)
        under_square_expression = np.power(red_layer, 2) + np.power(green_layer, 2) + np.power(blue_layer, 2) - \
                                  red_layer * green_layer - red_layer * blue_layer - green_layer * blue_layer
        H = np.where(green_layer >= blue_layer,
                     np.cos((red_layer - 0.5 * green_layer - 0.5 * blue_layer) / np.sqrt(under_square_expression))
                     ** (-1),
                     360 - np.cos((red_layer - 0.5 * green_layer - 0.5 * blue_layer) / np.sqrt(under_square_expression))
                     ** (-1))

        return BaseImage(np.dstack((H, S, I)), ColorModel.hsi)

    def to_hsl(self) -> 'BaseImage':
        """
        method that converts image in attribute date to hsl
        method returns new object of class BaseImage that stores image in hsl color model
        """
        if self.color_model != ColorModel.rgb:
            raise Exception("Color model must be RGB")
        red_layer = self.get_layer(0) / 255.0
        green_layer = self.get_layer(1) / 255.0
        blue_layer = self.get_layer(2) / 255.0
        M = np.maximum.reduce([red_layer, green_layer, blue_layer])
        m = np.minimum.reduce([red_layer, green_layer, blue_layer])
        d = (M - m) / 255.0
        L = (M + m) / 510.0
        S = np.where(L > 0, d / (1 - np.fabs(2 * L - 1)), 0)
        under_square_expression = np.power(red_layer, 2) + np.power(green_layer, 2) + np.power(blue_layer, 2) - \
                                  red_layer * green_layer - red_layer * blue_layer - green_layer * blue_layer
        H = np.where(green_layer >= blue_layer,
                     np.cos((red_layer - 0.5 * green_layer - 0.5 * blue_layer) / np.sqrt(under_square_expression))
                     ** (-1),
                     360 - np.cos((red_layer - 0.5 * green_layer - 0.5 * blue_layer) / np.sqrt(under_square_expression))
                     ** (-1))

        return BaseImage(np.dstack((H, S, L)), ColorModel.hsl)

    def hsv_to_rgb(self) -> 'BaseImage':
        """
        method that converts hsv image in attribute date to rgb
        method returns new object of class BaseImage that stores image in rgb color model
        """
        H = self.get_layer(0)
        S = self.get_layer(1)
        V = self.get_layer(2)
        M = 255 * V
        m = M * (1 - S)
        z = (M - m) * (1 - np.fabs(((H / 60) % 2) - 1))
        R = np.where(H < 60, M, np.where(H < 120, z + m, np.where(H < 240, m, np.where(H < 300, z + m, np.where(
            H < 360, M, 0)))))
        G = np.where(H < 60, z + m, np.where(H < 240, M, np.where(H < 360, m, 0)))
        B = np.where(H < 120, m, np.where(H < 240, z + m, np.where(H < 300, M, np.where(H < 360, z + m, 0))))
        return BaseImage(np.dstack((R, G, B)), ColorModel.rgb)

    def hsi_to_rgb(self) -> 'BaseImage':
        """
        method that converts hsi image in attribute date to rgb
        method returns new object of class BaseImage that stores image in rgb color model
        """
        H = self.get_layer(0)
        S = self.get_layer(1)
        I = self.get_layer(2)
        IS = I * S
        R = np.where(H == 0, I + 2 * IS, np.where(H < 120, I + IS * np.cos(H) / np.cos(60 - H), np.where(
            H <= 240, I - IS, np.where(H < 360, I + IS * (1 - np.cos(H - 240) / np.cos(300 - H)), 0))))
        G = np.where(H == 0, I - IS, np.where(H < 120, I + IS * (1 - np.cos(H) / np.cos(60 - H)), np.where(
            H == 120, I + 2 * IS, np.where(H < 240, I + IS * np.cos(H - 120) / np.cos(180 - H), np.where(
                H < 360, I - IS, 0)))))
        B = np.where(H <= 120, I - IS, np.where(H < 240, I + IS * (1 - np.cos(H - 120) / np.cos(180 - H)), np.where(
            H == 240, I + 2 * IS, np.where(H < 360, I + IS * np.cos(H - 240) / np.cos(300 - H), 0))))

        return BaseImage(np.dstack((R, G, B)), ColorModel.rgb)

    def hsl_to_rgb(self) -> 'BaseImage':
        """
        method that converts hsl image in attribute date to rgb
        method returns new object of class BaseImage that stores image in rgb color model
        """
        H = self.get_layer(0)
        S = self.get_layer(1)
        L = self.get_layer(2)
        d = S * (1 - np.fabs(2 * L - 1))
        m = 255.0 * (L - 0.5 * d)
        x = d * (1 - np.fabs((H / 60) % 2) - 1)
        d255m = d * 255.0 + m
        x255m = x * 255.0 + m
        R = np.where(H < 60, d255m, np.where(H < 120, x255m, np.where(H < 240, m, np.where(H < 300, x255m, np.where(
            H < 360, d255m, 0)))))
        G = np.where(H < 60, x255m, np.where(H < 120, d255m, np.where(H < 180, d255m, np.where(H < 240, x255m, np.where(
            H < 360, m, 0)))))
        B = np.where(H < 120, m, np.where(H < 180, x255m, np.where(H < 240, d255m, np.where(H < 300, d255m, np.where(
            H < 360, x255m, 0)))))
        return BaseImage(np.dstack((R, G, B)), ColorModel.rgb)

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


image_test = BaseImage('data/lena.jpg', ColorModel.rgb)
image_test.show_img()
x = image_test.to_hsl()
x.show_img()
y = x.to_rgb()
y.show_img()


