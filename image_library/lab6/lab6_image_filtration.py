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
        values = self.data
        if values.ndim == 2:
            values = np.expand_dims(values, axis=-1)
        if kernel.ndim == 2:
            kernel = np.repeat(np.expand_dims(kernel, axis=-1), values.shape[-1], axis=-1)
        if kernel.shape[-1] == 1:
            kernel = np.repeat(kernel, values.shape[-1], axis=-1)

        kernel = prefix * kernel.astype('float32')

        size_x, size_y = kernel.shape[:2]
        width, height = values.shape[:2]

        output_array = np.zeros((width - size_x + 3,
                                 height - size_y + 3,
                                 values.shape[-1]))

        padded_image = np.pad(values, [
                                        (1, 1),
                                        (1, 1),
                                        (0, 0)])

        for x in range(
                padded_image.shape[0] - size_x + 1):  # -size_x + 1 is to keep the window within the bounds of the image
            for y in range(padded_image.shape[1] - size_y + 1):
                # Creates the window with the same size as the kernel
                window = padded_image[x:x + size_x, y:y + size_y]

                # Sums over the product of the filter and the window
                output_values = np.sum(kernel * window, axis=(0, 1))
                output_array[x, y] = output_values

        output_array[output_array > 255] = 255
        output_array[output_array < 0] = 0

        return self.__class__(output_array.astype('i'), self.color_model)


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
