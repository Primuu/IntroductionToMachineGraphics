import numpy as np
from matplotlib import pyplot as plt
from matplotlib.image import imread

from image_library.imageLib import Image
from image_library.lab2.lab2 import BaseImage, ColorModel
from image_library.lab3.lab3 import GrayScaleTransform
from image_library.lab4.lab4_comparison import ImageComparison, ImageDiffMethod
from image_library.lab4.lab4_histogram import Histogram

lena = BaseImage('../data/lena.jpg', ColorModel.rgb)
lena.show_img()

# Lab 2

# lena_hsv = lena.to_hsv()
# lena_hsv_to_rgb = lena_hsv.hsv_to_rgb()
# lena_hsv_to_rgb.show_img()
#
# lena_hsi = lena.to_hsi()
# lena_hsi_to_rgb = lena_hsi.hsi_to_rgb()
# lena_hsi_to_rgb.show_img()
#
# lena_hsl = lena.to_hsl()
# lena_hsl_to_rgb = lena_hsl.hsl_to_rgb()
# lena_hsl_to_rgb.show_img()

# Lab 3

# gray_scale_transform = GrayScaleTransform('../data/lena.jpg', ColorModel.rgb)
# gray_scale_transform.show_img()
# gray = gray_scale_transform.to_gray()
# gray.show_img()
#
# sepia1 = gray_scale_transform.to_sepia((1.1, 0.9))
# sepia2 = gray_scale_transform.to_sepia((1.5, 0.5))
# sepia3 = gray_scale_transform.to_sepia((1.9, 0.1))
# sepia4 = gray_scale_transform.to_sepia((), 20)
# sepia5 = gray_scale_transform.to_sepia((), 30)
# sepia6 = gray_scale_transform.to_sepia((), 40)
# sepia1.show_img()
# sepia2.show_img()
# sepia3.show_img()
# sepia4.show_img()
# sepia5.show_img()
# sepia6.show_img()

# After adding main interface:
# image = Image('../data/lena.jpg', ColorModel.rgb)
# image.show_img()
# image.to_hsi().show_img()
# image.to_hsi().hsi_to_rgb().show_img()
# image.show_img()

# Lab 4

image_comparison = ImageComparison("../data/lena.jpg", ColorModel.rgb)
image_comparison.histogram().plot()

image_gray = GrayScaleTransform("../data/lena.jpg", ColorModel.rgb)
Histogram(image_gray.to_gray().data).plot()

korwin_lena = Image('../data/lena_korwin.jpg', ColorModel.rgb)
korwin_lena.show_img()
mse = image_comparison.compare_to(korwin_lena, ImageDiffMethod.mse)
rmse = image_comparison.compare_to(korwin_lena, ImageDiffMethod.rmse)
print("MSE: " + str(mse))
print("RMSE: " + str(rmse))
