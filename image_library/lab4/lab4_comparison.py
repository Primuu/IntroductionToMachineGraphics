import math
from enum import Enum
from typing import Any

import numpy as np

from image_library.lab2.lab2 import BaseImage, ColorModel
from image_library.lab3.lab3 import GrayScaleTransform
from image_library.lab4.lab4_histogram import Histogram


class ImageDiffMethod(Enum):
    mse = 0
    rmse = 1


class ImageComparison(BaseImage):
    """
    Class representing the image, its histogram, and comparison methods
    """
    def __init__(self, data: Any, color_model: ColorModel) -> None:
        super().__init__(data, color_model)

    def histogram(self) -> Histogram:
        """
        method returning object that contains the histogram of the current image (1- or multi-layer)
        """
        return Histogram(self.data)

    def compare_to(self, other: BaseImage, method: ImageDiffMethod) -> float:
        """
        method that returns mse or rmse for two images
        """
        if self.color_model != ColorModel.rgb or other.color_model != ColorModel.rgb:
            raise Exception("Both images must be rgb color model!")
        gray_self = GrayScaleTransform(self.data, ColorModel.rgb).to_gray()
        gray_other = GrayScaleTransform(other.data, ColorModel.rgb).to_gray()
        x_hist = Histogram(gray_self.data).values
        y_hist = Histogram(gray_other.data).values
        n = len(x_hist)
        sum_ = 0
        for i in range(n):
            sum_ += (x_hist[i] - y_hist[i]) ** 2
        sum_ = np.sum(sum_)
        sum_ = sum_ / n
        if method == ImageDiffMethod.rmse:
            sum_ = math.sqrt(sum_)
        return sum_
