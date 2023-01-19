from image_library.imageLib import *
from image_library.lab6.lab6_image_filtration import *

book = Image('../data/book.jpg', ColorModel.rgb)
book.show_img()

# result 1

conv_test = book
conv_test = Image((conv_test.data * 255).astype('i'), ColorModel.rgb)

result_aligned = conv_test.align_image(tail_elimination=True)
result_aligned.show_img()

result_aligned = book.align_image(tail_elimination=True)
result_aligned.show_img()

