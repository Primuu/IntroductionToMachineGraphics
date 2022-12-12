from typing import Any

import numpy as np

from image_library.lab2.lab2_base_image import BaseImage, ColorModel
from image_library.lab4.lab4_histogram import Histogram


class ImageAligning(BaseImage):
    """
    class responsible for histogram equalization
    """
    def __init__(self, data: Any, color_model: ColorModel) -> None:
        super().__init__(data, color_model)

    def align_image(self, tail_elimination: bool = True) -> BaseImage:
        """
        method returning corrected image based on histogram equalization method
        """
        def align(layer: np.ndarray) -> np.ndarray:
            layer_copy = np.float64(np.copy(layer))
            min_pixel = layer_copy.min()
            max_pixel = layer_copy.max()
            alignment = 255 / (max_pixel - min_pixel)
            layer_returned = ((layer_copy - min_pixel) * alignment).astype('i')
            return layer_returned

        def align_tail_elimination(layer: np.ndarray) -> np.ndarray:
            cumulated_histogram = Histogram(layer).to_cumulated().values
            range_in_cumulated_hist = cumulated_histogram[-1]
            min_tail = 0
            max_tail = 0
            cumulated_sum_of_hist = 0
            for value in cumulated_histogram:
                if cumulated_sum_of_hist <= 0.05 * range_in_cumulated_hist:
                    min_tail += 1
                if cumulated_sum_of_hist <= 0.95 * range_in_cumulated_hist:
                    max_tail += 1
                cumulated_sum_of_hist = value

            layer_copy = np.float64(np.copy(layer))
            alignment = 255 / (max_tail - min_tail)
            layer_returned = ((layer_copy - min_tail) * alignment).astype('i')
            layer_returned[layer_returned > 255] = 255
            layer_returned[layer_returned < 0] = 0
            return layer_returned

        if tail_elimination is False:
            if self.color_model == ColorModel.gray:
                self.data = align(self.data)
                return self.__class__(self.data, ColorModel.gray)
            else:
                first_layer, second_layer, third_layer = self.get_layers()
                aligned_f_l = align(first_layer)
                aligned_s_l = align(second_layer)
                aligned_t_l = align(third_layer)
                self.data = np.dstack((aligned_f_l, aligned_s_l, aligned_t_l))
                return self.__class__(self.data, self.color_model)
        else:
            if self.color_model == ColorModel.gray:
                self.data = align_tail_elimination(self.data)
                return self.__class__(self.data, ColorModel.gray)
            else:
                first_layer, second_layer, third_layer = self.get_layers()
                aligned_f_l = align_tail_elimination(first_layer)
                aligned_s_l = align_tail_elimination(second_layer)
                aligned_t_l = align_tail_elimination(third_layer)
                self.data = np.dstack((aligned_f_l, aligned_s_l, aligned_t_l))
                return self.__class__(self.data, self.color_model)
