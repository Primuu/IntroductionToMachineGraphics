from typing import Any

import numpy as np

from image_library.lab2.lab2_base_image import BaseImage, ColorModel


class ImageFiltration(BaseImage):
    def __init__(self, data: Any, color_model: ColorModel):
        super().__init__(data, color_model)

    def conv_2d(self, kernel: np.ndarray, prefix: float = 1) -> BaseImage:
        """
        kernel: the filter as numpy array
        prefix: filter prefix, if exists
                (Optional - 'object' correct form OR default value = 1 - arithmetic optimal form)
        method return image after filtration process
        """
        if self.data.ndim == 2:
            output_array = self.convolve(self.data, kernel) * prefix
            output_array[output_array > 255] = 255
            output_array[output_array < 0] = 0
            return self.__class__(output_array.astype('i'), self.color_model)
        else:
            first_layer, second_layer, third_layer = self.get_layers()
            # l_y = layer convolved
            first_l_y = self.convolve(first_layer, kernel) * prefix
            second_l_y = self.convolve(second_layer, kernel) * prefix
            third_l_y = self.convolve(third_layer, kernel) * prefix
            first_l_y[first_l_y > 255], second_l_y[second_l_y > 255], third_l_y[third_l_y > 255] = 255, 255, 255
            first_l_y[first_l_y < 0], second_l_y[second_l_y < 0], third_l_y[third_l_y < 0] = 0, 0, 0
            output_array = np.dstack((first_l_y, second_l_y, third_l_y))
            return self.__class__(output_array.astype('i'), self.color_model)

    def convolve(self, layer: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        rows_number, columns_number = layer.shape
        flattened_layer = np.reshape(layer, (1, layer.size))
        flattened_kernel = np.reshape(kernel, (1, kernel.size))
        flattened_convolved_layer = np.convolve(flattened_layer[0], flattened_kernel[0], 'same')
        convolved_layer = np.reshape(flattened_convolved_layer, (rows_number, columns_number))
        return convolved_layer


identity_prefix = 1
identity = np.array([
    [0, 0, 0],
    [0, 1, 0],
    [0, 0, 0]
])

high_pass_prefix = 1
high_pass = np.array([
    [0, -1, 0],
    [-1, 5, -1],
    [0, -1, 0]
])

low_pass_prefix = 1/9
low_pass = np.array([
    [1, 1, 1],
    [1, 1, 1],
    [1, 1, 1]
])

gaussian_blur_3x3_prefix = 1/16
gaussian_blur_3x3 = np.array([
    [1, 2, 1],
    [2, 4, 2],
    [1, 2, 1]
])

gaussian_blur_5x5_prefix = 1/256
gaussian_blur_5x5 = np.array([
    [1, 4, 6, 4, 1],
    [4, 16, 24, 16, 4],
    [6, 24, 36, 24, 6],
    [4, 16, 24, 16, 4],
    [1, 4, 6, 4, 1],
])

sobel_prefix = 1
sobel_0deg = np.array([
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]
])

sobel_45deg = np.array([
    [0, 1, 2],
    [-1, 0, 1],
    [-2, -1, 0]
])

sobel_90deg = np.array([
    [1, 2, 1],
    [0, 0, 0],
    [-1, -2, -1]
])

sobel_135deg = np.array([
    [2, 1, 0],
    [1, 0, -1],
    [0, -1, -2]
])
