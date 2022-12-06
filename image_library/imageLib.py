from typing import Any

from image_library.lab2.lab2 import ColorModel
from image_library.lab3.lab3 import GrayScaleTransform
from image_library.lab4.lab4_comparison import ImageComparison


class Image(GrayScaleTransform, ImageComparison):
    """
    class that is the main interface of the library
    """
    # TODO: , ImageAligning) - dodać do "sygnatury"
    def __init__(self, data: Any, color_model: ColorModel) -> None:
        super().__init__(data, color_model)
