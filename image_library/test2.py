import matplotlib
import numpy as np
import cv2
from matplotlib import pyplot as plt

from image_library.imageLib import *

lake = Image('../data/lena.jpg', ColorModel.rgb)
lake = Image('../data/lake.jpg', ColorModel.rgb)
mountains = Image('../data/mountains.jpg', ColorModel.rgb)
order = Image('../data/order.jpg', ColorModel.rgb)

# 1

# lena.show_img()
# lake.show_img()
# mountains.show_img()
# order.show_img()

mountains.show_img()
mountains_gray = mountains.to_gray()
mountains_gray.show_img()
mountains_gray.save_img('../data/bin/trash1.jpg')

# lena.show_img()
#
# R, G, B = lena.get_layers()
# R, G, B = R.astype(int), G.astype(int), B.astype(int)
# R[R > 255], G[G > 255], B[B > 255] = 255, 255, 255
# R[R < 0], G[G < 0], B[B < 0] = 0, 0, 0
# lena = Image(np.dstack((R, G, B)).astype('uint8'), ColorModel.hsv)
#
# lena.save_img('../data/bin/trash1.jpg')






