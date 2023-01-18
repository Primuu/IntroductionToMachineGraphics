from typing import Any

from image_library.lab2.lab2_base_image import BaseImage, ColorModel
import cv2


class EdgeDetection(BaseImage):

    def __init__(self, data: Any, color_model: ColorModel):
        super().__init__(data, color_model)

    def canny(self, th0: int, th1: int, kernel_size: int) -> BaseImage:
        if self.color_model != ColorModel.gray:
            raise Exception("Color model must be gray")
        edges = cv2.Canny(self.data.astype('uint8'), th0, th1, kernel_size)
        return self.__class__(edges, ColorModel.gray)

