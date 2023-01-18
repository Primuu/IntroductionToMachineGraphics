import cv2
import matplotlib

import matplotlib.pyplot as plt
import numpy as np

# img_color = cv2.imread('../../data/mauritius.jpg', cv2.IMREAD_COLOR)
# plt.imshow(img_color)
# matplotlib.pyplot.show()
# # print(img_color.shape)
#
# img_grayscale = cv2.imread('../../data/mauritius.jpg', cv2.IMREAD_GRAYSCALE)
# plt.imshow(img_grayscale, cmap='gray')
# matplotlib.pyplot.show()
# # print(img_grayscale.shape)
#
# # IMREAD_UNCHANGED for images with alpha canal
# img_with_alpha = cv2.imread('../../data/mauritius.jpg', cv2.IMREAD_UNCHANGED)
# plt.imshow(img_with_alpha)
# matplotlib.pyplot.show()
# # print(img_with_alpha.shape)


# # color model conversion
# img_color = cv2.imread('../../data/mauritius.jpg', cv2.IMREAD_COLOR)
# img_rgb = cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB)
# plt.imshow(img_rgb)
# matplotlib.pyplot.show()
#
# # other:
# # COLOR_BGR2GRAY
# # COLOR_BGR2XYZ
# # COLOR_BGR2HSV
# # COLOR_BGR2Lab
# # COLOR_BGR2Luv
# # COLOR_BGR2HLS
# # COLOR_BGR2YUV


# THRESHOLDING
# OTSU

# lena_gray = cv2.imread('../../data/lena.jpg', cv2.IMREAD_GRAYSCALE)
#
# plt.imshow(lena_gray, cmap='gray')
# matplotlib.pyplot.show()
#
# _, thresh_otsu = cv2.threshold(lena_gray, thresh=0, maxval=255, type=cv2.THRESH_BINARY + cv2.THRESH_OTSU)
#
# plt.imshow(thresh_otsu, cmap='gray')
# matplotlib.pyplot.show()

# ADAPTIVE

# # adaptiveMethod - parametr stałej określającej metodę wyznaczania lokalnych progów
# # ADAPTIVE_THRESH_MEAN_C - wyznaczanie progu na podstawie średniej wartości sąsiednich pikseli
# # ADAPTIVE_THRESH_GAUSSIAN_C - wyznaczanie progu na podstawie sumy ważonej sąsiednich pikseli,
# # gdzie wagi pochodzą z rozkładu gaussowskiego
# # thresholdType - metoda binaryzacji blockSize - wielkośc obszaru sąsiedztwa
# # C - stała odejmowana od obliczonej średniej arytmetycznej lub ważonej
#
# th_adaptive = cv2.adaptiveThreshold(lena_gray, maxValue=255, adaptiveMethod=cv2.ADAPTIVE_THRESH_MEAN_C,
#                                     thresholdType=cv2.THRESH_BINARY, blockSize=13, C=8)
#
# plt.imshow(th_adaptive, cmap='gray')
# matplotlib.pyplot.show()

# EDGE DETECTING - CANNY

# canny_edges = cv2.Canny(
#       lena_gray,
#       16,  # prog histerezy 1
#       40,  # prog histerezy 2
#       3  # wielkoscc filtra sobela
# )
# plt.imshow(canny_edges, cmap='gray')
# matplotlib.pyplot.show()

# HISTOGRAM ALIGNING

# lake_color = cv2.imread('../../data/swiecajty.jpg', cv2.IMREAD_COLOR)
#
# plt.imshow(lake_color)
# matplotlib.pyplot.show()
#
# lake_gray = cv2.cvtColor(lake_color, cv2.COLOR_BGR2GRAY)
# plt.imshow(lake_gray , cmap='gray')
# matplotlib.pyplot.show()
#
# clahe = cv2.createCLAHE(
#     clipLimit=2.0,
#     tileGridSize=(4, 4)
# )
# equalized_lake_gray = clahe.apply(lake_gray)
#
# plt.subplot(221)
# plt.imshow(lake_gray, cmap='gray')
#
# plt.subplot(222)
# plt.hist(lake_gray.ravel(), bins=256, range=(0, 256), color='gray')
#
# plt.subplot(223)
# plt.imshow(equalized_lake_gray, cmap='gray')
#
# plt.subplot(224)
# plt.hist(equalized_lake_gray.ravel(), bins=256, range=(0, 256), color='gray')
#
# plt.show()
# matplotlib.pyplot.show()

# Histogram correction in color images

# lake_rgb = cv2.cvtColor(lake_color, cv2.COLOR_BGR2RGB)
# lake_lab = cv2.cvtColor(lake_color, cv2.COLOR_BGR2LAB)
#
# clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
# lake_lab[..., 0] = clahe.apply(lake_lab[..., 0])
# lake_color_equalized = cv2.cvtColor(lake_lab, cv2.COLOR_LAB2RGB)
#
# plt.subplot(221)
# plt.imshow(lake_rgb)
#
# plt.subplot(222)
# plt.hist(lake_rgb[..., 0].ravel(), bins=256, range=(0, 256), color='b')
# plt.hist(lake_rgb[..., 1].ravel(), bins=256, range=(0, 256), color='g')
# plt.hist(lake_rgb[..., 2].ravel(), bins=256, range=(0, 256), color='r')
#
# plt.subplot(223)
# plt.imshow(lake_color_equalized)
#
# plt.subplot(224)
# plt.hist(lake_color_equalized[..., 0].ravel(), bins=256, range=(0, 256), color='b')
# plt.hist(lake_color_equalized[..., 1].ravel(), bins=256, range=(0, 256), color='g')
# plt.hist(lake_color_equalized[..., 2].ravel(), bins=256, range=(0, 256), color='r')
#
# plt.show()
# matplotlib.pyplot.show()

# Hough transforms

# lines_img = cv2.imread('../../data/lines.jpg', cv2.IMREAD_GRAYSCALE)
# plt.imshow(lines_img, cmap='gray')
# matplotlib.pyplot.show()
#
# _, lines_thresh = cv2.threshold(
#     lines_img,
#     thresh=0,
#     maxval=255,
#     type=cv2.THRESH_BINARY + cv2.THRESH_OTSU
# )
#
# plt.imshow(lines_thresh, cmap='gray')
# matplotlib.pyplot.show()
#
#
# lines_edges = cv2.Canny(lines_thresh, 20, 50, 3)
# plt.imshow(lines_edges, cmap='gray')
# matplotlib.pyplot.show()
#
# lines = cv2.HoughLinesP(
#     lines_edges,
#     2,
#     np.pi / 180,
#     30
# )
# print(len(lines))
# result_lines_img = cv2.cvtColor(lines_img, cv2.COLOR_GRAY2RGB)
# for line in lines:
#   x0, y0, x1, y1 = line[0]
#   cv2.line(result_lines_img, (x0, y0), (x1, y1), (0, 255, 0), 5)
# plt.imshow(result_lines_img)
# matplotlib.pyplot.show()

# Circle detection

# checkers_img = cv2.imread('../../data/checkers.jpg')
# checkers_gray = cv2.cvtColor(checkers_img, cv2.COLOR_BGR2GRAY)
# checkers_color = cv2.cvtColor(checkers_img, cv2.COLOR_BGR2RGB)
#
# circles = cv2.HoughCircles(
#     checkers_gray,
#     method=cv2.HOUGH_GRADIENT,
#     dp=2,
#     minDist=60,
#     minRadius=20,
#     maxRadius=100
# )
# print(len(circles[0]))
#
# for (x, y, r) in circles.astype(int)[0]:
#   cv2.circle(checkers_color, (x, y), r, (0, 255, 0), 4)
#
# plt.imshow(checkers_color)
# matplotlib.pyplot.show()
