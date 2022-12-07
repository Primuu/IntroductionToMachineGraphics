from typing import Any

import numpy as np

from image_library.lab2.lab2 import BaseImage, ColorModel


class ImageAligning(BaseImage):
    """
    class responsible for histogram equalization
    """
    def __init__(self, data: Any, color_model: ColorModel) -> None:
        super().__init__(data, color_model)

    def align_image(self, tail_elimination: bool = True) -> 'BaseImage':
        """
        method returning corrected image based on histogram equalization method
        """
        if self.data.ndim == 2:
            min_pixel = self.data.min()
            max_pixel = self.data.max()
            alignment = min_pixel * 255 / (max_pixel - min_pixel)
            self.data = self.data - alignment
            self.data = self.data.astype(int)
        return self.__class__(self.data, ColorModel.gray)


