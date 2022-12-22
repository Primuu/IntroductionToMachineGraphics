from typing import Any

import numpy as np

from image_library.lab2.lab2_base_image import BaseImage, ColorModel
from image_library.lab3.lab3_grayscale_transform import GrayScaleTransform


class Thresholding(BaseImage):
    def __init__(self, data: Any, color_model: ColorModel):
        super().__init__(data, color_model)

    def threshold(self, value: int) -> BaseImage:
        """
        method that performs segmentation operations using binarization
        """
        if self.color_model != ColorModel.gray and self.color_model != ColorModel.rgb:
            raise Exception("Color model must be RGB or grey")
        if self.color_model == ColorModel.rgb:
            pixels = GrayScaleTransform(self.data, self.color_model).to_gray().data
        else:
            pixels = self.data
        pixels = np.where(pixels < value, 0, 255)
        return self.__class__(pixels, ColorModel.gray)
