from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from lab2 import BaseImage, ColorModel


class GrayScaleTransform(BaseImage):
    def __init__(self, data: Any, color_model: ColorModel) -> None:
        super().__init__(data, color_model)

    def to_gray(self) -> BaseImage:
        """
        method that returns grayscale image as an object of the BaseImage class
        """
        if self.color_model != ColorModel.rgb:
            raise Exception("Color model must be RGB")
        red_layer = self.get_layer(0)
        green_layer = self.get_layer(1)
        blue_layer = self.get_layer(2)
        gray = np.floor((red_layer + green_layer + blue_layer) / 3.0)
        return BaseImage(gray, ColorModel.gray)

    def to_sepia(self, alpha_beta: tuple = (None, None), w: int = None) -> BaseImage:
        """
        method that returns a sepia image as an object of the BaseImage class
        depending on the given arguments: alpha and beta or w
        """
        if self.color_model != ColorModel.rgb:
            raise Exception("Color model must be RGB")
        # if alpha_beta
        gray_image = self.to_gray()
        L0 = gray_image.data
        L1 = gray_image.data
        L2 = gray_image.data
        L0 = L0 * alpha_beta[0]
        L2 = L2 * alpha_beta[1]
        return BaseImage(np.dstack((L0, L1, L2)), ColorModel.sepia)


image_test1 = GrayScaleTransform('data/lena.jpg', ColorModel.rgb)
# image_test1.show_img()
# # gray_i = image_test1.to_gray()
# sepia_i = image_test1.to_sepia((1.5, 0.5))
# sepia_i.show_img()

red_layer = image_test1.get_layer(0)
green_layer = image_test1.get_layer(1)
blue_layer = image_test1.get_layer(2)
gray = int((red_layer + green_layer + blue_layer) / 3.0)
print(gray)