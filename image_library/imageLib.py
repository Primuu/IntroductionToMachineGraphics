from typing import Any

from image_library.lab2.lab2 import ColorModel
from image_library.lab3.lab3 import GrayScaleTransform
from image_library.lab4.lab4_histogram import Histogram


class Image(GrayScaleTransform):
    """
    class that is the main interface of the library
    """
    # TODO:  self.__class__  - dodać do klas użycia dynamicznego odwoałania do klasy
    def __init__(self, data: Any, color_model: ColorModel) -> None:
        super().__init__(data, color_model)
