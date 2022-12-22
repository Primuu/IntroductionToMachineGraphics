import numpy as np

from image_library.imageLib import Image
from image_library.lab2.lab2_base_image import ColorModel
from image_library.lab4.lab4_comparison import ImageDiffMethod
from image_library.lab4.lab4_histogram import Histogram
from image_library.lab6.lab6_image_filtration import *

lena = Image('../data/lena.jpg', ColorModel.rgb)
lena_dot = Image('../data/lena_kropka.jpg', ColorModel.rgb)
lena_confederation = Image('../data/lena_korwin.jpg', ColorModel.rgb)

# lena.show_img()

# Lab 2

# lena_hsv = lena.to_hsv()
# lena_hsv.show_img_in_rgb_range()
# lena_hsv_to_rgb = lena_hsv.to_rgb()
# lena_hsv_to_rgb.show_img()
#
# lena_hsi = lena.to_hsi()
# lena_hsi.show_img_in_rgb_range()
# lena_hsi_to_rgb = lena_hsi.to_rgb()
# lena_hsi_to_rgb.show_img()
#
# lena_hsl = lena.to_hsl()
# lena_hsl.show_img_in_rgb_range()
# lena_hsl_to_rgb = lena_hsl.to_rgb()
# lena_hsl_to_rgb.show_img()

# Lab 3

# lena_grey = lena.to_gray()
# lena_grey.show_img()
#
# sepia1 = lena.to_sepia((1.1, 0.9))
# sepia2 = lena.to_sepia((1.5, 0.5))
# sepia3 = lena.to_sepia((1.9, 0.1))
# sepia4 = lena.to_sepia((), 20)
# sepia5 = lena.to_sepia((), 30)
# sepia6 = lena.to_sepia((), 40)
# sepia1.show_img()
# sepia2.show_img()
# sepia3.show_img()
# sepia4.show_img()
# sepia5.show_img()
# sepia6.show_img()

# Lab 4

# lena_histogram_rgb = lena.histogram()
# lena_histogram_rgb.plot()
#
# lena_grey = Image(lena.to_gray().data, ColorModel.gray)
# lena_grey.show_img()
# lena_histogram_grey = lena_grey.histogram()
# lena_histogram_grey.plot()
#
# lena_dot.show_img()
#
# lena_confederation.show_img()
#
# mse = lena.compare_to(lena_dot, ImageDiffMethod.mse)
# rmse = lena.compare_to(lena_dot, ImageDiffMethod.rmse)
# print("Lena i lena_kropka: Mse - " + str(mse) + " ,Rmse - " + str(rmse))
#
# mse1 = lena.compare_to(lena_confederation, ImageDiffMethod.mse)
# rmse1 = lena.compare_to(lena_confederation, ImageDiffMethod.rmse)
# print("Lena i lena_korwin: Mse - " + str(mse1) + " ,Rmse - " + str(rmse1))

# Lab 5

# # Grey
#
# lena_grey = Image(lena.to_gray().data, ColorModel.gray)
# lena_grey.show_img()
# lena_histogram_grey = lena_grey.histogram()
# lena_histogram_grey.plot()
#
# lena_grey_aligned = lena_grey.align_image(tail_elimination=False)
# lena_grey_aligned.show_img()
# Histogram(lena_grey_aligned.data).plot()
#
# # RGB
# lena.show_img()
# lena_histogram_rgb = lena.histogram()
# lena_histogram_rgb.plot()
# lena_grey = Image(lena.to_gray().data, ColorModel.gray)
# lena_histogram_grey = lena_grey.histogram()
# lena_histogram_grey.plot()
#
# lena_aligned = lena.align_image(tail_elimination=False)
# lena_aligned.show_img()
# Histogram(lena_aligned.data).plot()
# Histogram(Image(lena_aligned.data, ColorModel.rgb).to_gray().data).plot()
#
#
# # Tail elimination
# # Grey
#
# lena_grey = Image(lena.to_gray().data, ColorModel.gray)
# lena_grey.show_img()
# lena_histogram_grey = lena_grey.histogram()
# lena_histogram_grey.plot()
#
# lena_grey_aligned_tail_eli = lena_grey.align_image(tail_elimination=True)
# lena_grey_aligned_tail_eli.show_img()
# Histogram(lena_grey_aligned_tail_eli.data).plot()
#
# # RGB
# lena.show_img()
# lena_grey = Image(lena.to_gray().data, ColorModel.gray)
# lena_histogram_rgb = lena.histogram()
# lena_histogram_rgb.plot()
# lena_histogram_cumulated = lena_grey.histogram().to_cumulated()
# lena_histogram_cumulated.plot()
#
# lena_aligned_tail_eli = lena.align_image(tail_elimination=True)
# lena_aligned_tail_eli.show_img()
# Histogram(lena_aligned_tail_eli.data).plot()
# Histogram(Image(lena_aligned_tail_eli.data, ColorModel.rgb).to_gray().data).plot()

# Lab 6

# conv_test = Image('../data/conv_test.jpg', ColorModel.rgb)
# conv_test = Image((conv_test.data * 255).astype('i'), ColorModel.rgb)
# conv_test.show_img()
#
# identity_conv = conv_test.conv_2d(identity, identity_prefix)
# identity_conv.show_img()
#
# high_pass_conv = conv_test.conv_2d(high_pass, high_pass_prefix)
# high_pass_conv.show_img()
#
# low_pass_conv = conv_test.conv_2d(low_pass, low_pass_prefix)
# low_pass_conv.show_img()
#
# gaussian_blur_3x3_conv = conv_test.conv_2d(gaussian_blur_3x3, gaussian_blur_3x3_prefix)
# gaussian_blur_3x3_conv.show_img()
#
# gaussian_blur_5x5_conv = conv_test.conv_2d(gaussian_blur_5x5, gaussian_blur_5x5_prefix)
# gaussian_blur_5x5_conv.show_img()
#
# # SUDOKU
#
# sudoku = Image('../data/sudoku.jpg', ColorModel.rgb)
# sudoku = Image((sudoku.data * 255).astype('i'), ColorModel.rgb)
# sudoku.show_img()
#
# sudoku_0deg = sudoku.conv_2d(sobel_0deg, sobel_prefix)
# sudoku_0deg.show_img()
#
# sudoku_45deg = sudoku.conv_2d(sobel_45deg, sobel_prefix)
# sudoku_45deg.show_img()
#
# sudoku_90deg = sudoku.conv_2d(sobel_90deg, sobel_prefix)
# sudoku_90deg.show_img()
#
# sudoku_135deg = sudoku.conv_2d(sobel_135deg, sobel_prefix)
# sudoku_135deg.show_img()
#
# detected_edges = ((sudoku_0deg.data + sudoku_45deg.data + sudoku_90deg.data + sudoku_135deg.data) / 4).astype('i')
# print(detected_edges)
# Image(detected_edges, ColorModel.rgb).show_img()

# Lab 7

lena.show_img()

threshold_30 = lena.threshold(30)
threshold_30.show_img()

threshold_70 = lena.threshold(70)
threshold_70.show_img()

threshold_127 = lena.threshold(127)
threshold_127.show_img()

threshold_170 = lena.threshold(170)
threshold_170.show_img()

threshold_220 = lena.threshold(220)
threshold_220.show_img()
