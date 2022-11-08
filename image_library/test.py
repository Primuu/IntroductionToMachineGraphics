from image_library.imageLib import Image
from image_library.lab2.lab2 import BaseImage, ColorModel
from image_library.lab3.lab3 import GrayScaleTransform

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