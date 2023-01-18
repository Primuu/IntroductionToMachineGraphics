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
    color_model: ColorModel  # attribute that stores current color model of image

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

    def get_H(self, R, G, B):
        """
        method calculating H in color models like HSV, HSI, HSL
        """
        under_square_root = np.power(R, 2) + np.power(G, 2) + np.power(B, 2) - (R * G) - (R * B) - (G * B)
        square_root = np.sqrt(under_square_root)
        square_root[square_root == 0] = 1
        bracket_value = ((R - (G / 2) - (B / 2)) / square_root)
        # (cosine)^(-1) = arcus cosine
        acos = np.arccos(bracket_value) * 180 / pi
        H = np.where(G >= B, acos, 360 - acos)
        return H

    def to_hsv(self) -> 'BaseImage':
        """
        method that converts image in attribute date to hsv
        method returns new object of class BaseImage that stores image in hsv color model
        """
        if self.color_model != ColorModel.rgb:
            raise Exception("Color model must be RGB")
        R, G, B = self.get_layers()
        M = np.max([R, G, B], axis=0)
        m = np.min([R, G, B], axis=0)
        V = M / 255
        S = np.where(M == 0, 0, (1 - m / M))
        R, G, B = R.astype(int), G.astype(int), B.astype(int)
        H = self.get_H(R, G, B)
        return BaseImage(np.dstack((H, S, V)), ColorModel.hsv)

    def to_hsi(self) -> 'BaseImage':
        """
        method that converts image in attribute date to hsi
        method returns new object of class BaseImage that stores image in hsi color model
        """
        if self.color_model != ColorModel.rgb:
            raise Exception("Color model must be RGB")
        R, G, B = np.float32(self.get_layers())
        H = self.get_H(R.astype(int), G.astype(int), B.astype(int))
        R_G_B = R + G + B
        I = (R_G_B) / (3 * 255)
        # changed algorithm for S
        R_G_B[R_G_B == 0] = 255
        r_divided, g_divided, b_divided = R / R_G_B, G / R_G_B, B / R_G_B
        m = np.min([r_divided, g_divided, b_divided], axis=0)
        S = 1 - 3 * m
        return BaseImage(np.dstack((H, S, I)), ColorModel.hsi)

    def to_hsl(self) -> 'BaseImage':
        """
        method that converts image in attribute date to hsl
        method returns new object of class BaseImage that stores image in hsl color model
        """
        if self.color_model != ColorModel.rgb:
            raise Exception("Color model must be RGB")
        R, G, B = self.get_layers()
        M = np.max([R, G, B], axis=0)
        m = np.min([R, G, B], axis=0)
        d = (M - m) / 255
        L = (0.5 * (M.astype(int) + m.astype(int))).astype(int) / 255
        d_divided = d / (1 - abs(2 * L - 1))
        # divisor < 0 ?
        S = np.where(L > 0, d_divided, 0)
        S[S > 1], S[S < 0] = 1, 0
        H = self.get_H(R.astype(int), G.astype(int), B.astype(int))
        return BaseImage(np.dstack((H, S, L)), ColorModel.hsl)

    def hsv_to_rgb(self) -> 'BaseImage':
        """
        method that converts hsv image in attribute date to rgb
        method returns new object of class BaseImage that stores image in rgb color model
        """
        H, S, V = self.get_layers()
        M = 255 * V
        m = M * (1 - S)
        z = (M - m) * (1 - np.fabs(((H / 60) % 2) - 1))
        R = np.where(H < 60, M, np.where(H < 120, z + m, np.where(H < 240, m, np.where(H < 300, z + m, np.where(
            H < 360, M, 0)))))
        G = np.where(H < 60, z + m, np.where(H < 240, M, np.where(H < 360, m, 0)))
        B = np.where(H < 120, m, np.where(H < 240, z + m, np.where(H < 300, M, np.where(H < 360, z + m, 0))))

        R, G, B = R.astype(int), G.astype(int), B.astype(int)
        R[R > 255], G[G > 255], B[B > 255] = 255, 255, 255
        R[R < 0], G[G < 0], B[B < 0] = 0, 0, 0
        return BaseImage(np.dstack((R, G, B)), ColorModel.rgb)


    def hsi_to_rgb(self) -> 'BaseImage':
        """
        method that converts hsi image in attribute date to rgb
        method returns new object of class BaseImage that stores image in rgb color model
        """
        # lesson algorithm
        # H = self.get_layer(0)
        # S = self.get_layer(1)
        # I = self.get_layer(2)
        # IS = I * S
        # R = np.where(H == 0, I + 2 * IS, np.where(H < 120, I + IS * np.cos(H) / np.cos(60 - H), np.where(
        #     H <= 240, I - IS, np.where(H < 360, I + IS * (1 - np.cos(H - 240) / np.cos(300 - H)), 0))))
        # G = np.where(H == 0, I - IS, np.where(H < 120, I + IS * (1 - np.cos(H) / np.cos(60 - H)), np.where(
        #     H == 120, I + 2 * IS, np.where(H < 240, I + IS * np.cos(H - 120) / np.cos(180 - H), np.where(
        #         H < 360, I - IS, 0)))))
        # B = np.where(H <= 120, I - IS, np.where(H < 240, I + IS * (1 - np.cos(H - 120) / np.cos(180 - H)), np.where(
        #     H == 240, I + 2 * IS, np.where(H < 360, I + IS * np.cos(H - 240) / np.cos(300 - H), 0))))
        #
        # return BaseImage(np.dstack((R, G, B)), ColorModel.rgb)

        # changed algorithm
        H, S, I = self.get_layers()
        rows, columns = self.data.shape[:2]
        R, G, B = np.zeros((rows, columns)), np.zeros((rows, columns)), np.zeros((rows, columns))

        for i in range(rows):
            for j in range(columns):
                h = H[i, j] * np.pi / 180
                if h < np.pi * 2 / 3:
                    x = I[i, j] * (1 - S[i, j])
                    y = I[i, j] * (1 + S[i, j] * np.cos(h) / np.cos(np.pi / 3 - h))
                    z = 3 * I[i, j] - (x + y)
                    R[i, j], G[i, j], B[i, j] = y, z, x
                elif np.pi * 2 / 3 <= h < np.pi * 4 / 3:
                    h -= np.pi * 2 / 3
                    x = I[i, j] * (1 - S[i, j])
                    y = I[i, j] * (1 + S[i, j] * np.cos(h) / np.cos(np.pi / 3 - h))
                    z = 3 * I[i, j] - (x + y)
                    R[i, j], G[i, j], B[i, j] = x, y, z
                elif np.pi * 4 / 3 < h < np.pi * 2:
                    h -= np.pi * 4 / 3
                    x = I[i, j] * (1 - S[i, j])
                    y = I[i, j] * (1 + S[i, j] * np.cos(h) / np.cos(np.pi / 3 - h))
                    z = 3 * I[i, j] - (x + y)
                    R[i, j], G[i, j], B[i, j] = z, x, y

        R, G, B = R * 255, G * 255, B * 255
        R[R > 255], G[G > 255], B[B > 255] = 255, 255, 255
        R[R < 0], G[G < 0], B[B < 0] = 0, 0, 0

        return BaseImage(np.dstack((R, G, B)).astype(np.uint16), ColorModel.rgb)

    def hsl_to_rgb(self) -> 'BaseImage':
        """
        method that converts hsl image in attribute date to rgb
        method returns new object of class BaseImage that stores image in rgb color model
        """
        H, S, L = self.get_layers()
        d = S * (1 - np.fabs(2 * L - 1))
        m = 255.0 * (L - 0.5 * d)
        x = d * (1 - abs(H / 60 % 2 - 1))
        d255m = d * 255.0 + m
        x255m = x * 255.0 + m
        R = np.where(H < 60, d255m, np.where(H < 120, x255m, np.where(H < 240, m, np.where(H < 300, x255m, np.where(
            H < 360, d255m, 0)))))
        G = np.where(H < 60, x255m, np.where(H < 120, d255m, np.where(H < 180, d255m, np.where(H < 240, x255m, np.where(
            H < 360, m, 0)))))
        B = np.where(H < 120, m, np.where(H < 180, x255m, np.where(H < 240, d255m, np.where(H < 300, d255m, np.where(
            H < 360, x255m, 0)))))

        R, G, B = R.astype(int), G.astype(int), B.astype(int)
        R[R > 255], G[G > 255], B[B > 255] = 255, 255, 255
        R[R < 0], G[G < 0], B[B < 0] = 0, 0, 0
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
