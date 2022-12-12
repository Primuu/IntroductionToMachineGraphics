from typing import Optional

import numpy as np

from image_library.lab2.lab2_base_image import BaseImage


class ImageFiltration:
    def conv_2d(self, image: BaseImage, kernel: np.ndarray, prefix: Optional[float] = None) -> BaseImage:
        """
        kernel: filtr w postaci tablicy numpy
        prefix: przedrostek filtra, o ile istnieje; Optional - forma poprawna obiektowo, lub domyslna wartosc = 1 - optymalne arytmetycznie
        metoda zwroci obraz po procesie filtrowania
        """
        pass




