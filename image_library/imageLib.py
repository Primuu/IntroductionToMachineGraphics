from typing import Any

from image_library.lab2.lab2_base_image import ColorModel
from image_library.lab3.lab3_grayscale_transform import GrayScaleTransform
from image_library.lab4.lab4_comparison import ImageComparison
from image_library.lab5.lab5_image_aligning import ImageAligning
from image_library.lab6.lab6_image_filtration import ImageFiltration
from image_library.lab7.lab7_thresholding import Thresholding


class Image(GrayScaleTransform, ImageComparison, ImageAligning, ImageFiltration, Thresholding):
    """
    class that is the main interface of the library
    """
    def __init__(self, data: Any, color_model: ColorModel) -> None:
        super().__init__(data, color_model)
