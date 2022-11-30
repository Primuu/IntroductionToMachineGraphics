import numpy as np
from matplotlib import pyplot as plt
from matplotlib.image import imread

from image_library.imageLib import Image
from image_library.lab2.lab2 import BaseImage, ColorModel
from image_library.lab3.lab3 import GrayScaleTransform
from image_library.lab4.lab4_comparison import ImageComparison, ImageDiffMethod
from image_library.lab4.lab4_histogram import Histogram
from image_library.lab5.lab5_image_aligning import ImageAligning

# lena = BaseImage('../data/lena.jpg', ColorModel.rgb)
# lena.show_img()

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

# Lab 4

lena1 = ImageComparison('../data/lena.jpg', ColorModel.rgb)
# lena1.show_img()
# lena1.histogram().plot()

gray_lena = GrayScaleTransform(lena1.data, ColorModel.rgb).to_gray()
gray_lena.show_img()
Histogram(gray_lena.data).plot()
#
# lena_kropka = Image('../data/lena_kropka.jpg', ColorModel.rgb)
# lena_kropka.show_img()
#
# lena_korwin = Image('../data/lena_korwin.jpg', ColorModel.rgb)
# lena_korwin.show_img()
#
# mse = lena1.compare_to(lena_kropka, ImageDiffMethod.mse)
# rmse = lena1.compare_to(lena_kropka, ImageDiffMethod.rmse)
# print("Lena i lena_kropka: Mse - " + str(mse) + " ,Rmse - " + str(rmse))
#
# mse1 = lena1.compare_to(lena_korwin, ImageDiffMethod.mse)
# rmse1 = lena1.compare_to(lena_korwin, ImageDiffMethod.rmse)
# print("Lena i lena_korwin: Mse - " + str(mse1) + " ,Rmse - " + str(rmse1))


# Lab 5

print(gray_lena.data)

image_aligning = ImageAligning(gray_lena.data, ColorModel.gray).align_image()
print(image_aligning.data)

grey_image_aligning = GrayScaleTransform(image_aligning.data, ColorModel.gray)
grey_image_aligning.show_img()
Histogram(grey_image_aligning.data).plot()
