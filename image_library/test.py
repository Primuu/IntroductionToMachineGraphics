from image_library.imageLib import Image
from image_library.lab2.lab2 import ColorModel
from image_library.lab4.lab4_comparison import ImageDiffMethod
from image_library.lab4.lab4_histogram import Histogram

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

# lena.to_gray().show_img()
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

# lena.histogram().plot()
#
# lena.to_gray().show_img()
# Histogram(lena.to_gray().data).plot()
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

# Grey

# lena.to_gray().show_img()
# Histogram(lena.to_gray().data).plot()
#
# lena_grey_aligned = Image(lena.to_gray().data, ColorModel.gray).align_image(tail_elimination=False)
# lena_grey_aligned.show_img()
# Histogram(lena_grey_aligned.data).plot()
#
# # RGB
# lena.show_img()
# Histogram(lena.to_gray().data).plot()
# Histogram(lena.data).plot()
#
# lena_aligned = lena.align_image(tail_elimination=False)
# lena_aligned.show_img()
# Histogram(Image(lena_aligned.data, ColorModel.rgb).to_gray().data).plot()
# Histogram(lena_aligned.data).plot()
#
#
# # Tail elimination
# # Grey
#
# lena.to_gray().show_img()
# Histogram(lena.to_gray().data).plot()
#
# lena_grey_aligned = Image(lena.to_gray().data, ColorModel.gray).align_image(tail_elimination=True)
# lena_grey_aligned.show_img()
# Histogram(lena_grey_aligned.data).plot()

# RGB
lena.show_img()
Histogram(lena.to_gray().data).plot()
Histogram(lena.data).plot()

lena_aligned = lena.align_image(tail_elimination=True)
lena_aligned.show_img()
Histogram(Image(lena_aligned.data, ColorModel.rgb).to_gray().data).plot()
Histogram(lena_aligned.data).plot()
