import numpy as np
from matplotlib import pyplot as plt


class Histogram:
    """
    class representing the histogram of a given image
    """
    values: np.ndarray  # attribute that stores the values of the histogram of a given image

    def __init__(self, pixels: np.ndarray) -> None:
        if pixels.ndim == 2:
            self.values = np.histogram(pixels, bins=256, range=(0, 255))[0]
        else:
            first_layer = pixels[:, :, 0]
            second_layer = pixels[:, :, 1]
            third_layer = pixels[:, :, 2]
            first_layer_as_histogram = np.histogram(first_layer, bins=256, range=(0, 255))[0]
            second_layer_as_histogram = np.histogram(second_layer, bins=256, range=(0, 255))[0]
            third_layer_as_histogram = np.histogram(third_layer, bins=256, range=(0, 255))[0]
            self.values = np.dstack((first_layer_as_histogram, second_layer_as_histogram, third_layer_as_histogram))

    def plot(self) -> None:
        """
        method displaying a histogram from the values attribute
        """
        if self.values.ndim == 1:
            plt.figure()
            plt.title("Grayscale Histogram")
            plt.xlabel("grayscale value")
            plt.ylabel("pixel count")
            plt.xlim([0, 255])
            bin_edges = np.linspace(0, 254.9, 256)
            plt.plot(bin_edges, self.values, color="black")
            plt.show()
        else:
            plt.figure(figsize=(12, 9))
            bin_edges = np.linspace(0, 254.9, 256)
            plt.subplot(131)

            plt.title("red layer")
            plt.xlim([0, 255])
            plt.ylabel("pixel count")
            plt.xlabel("redscale value")
            plt.plot(bin_edges, self.values[:, :, 0].flatten(), color="red")

            plt.subplot(132)
            plt.title("green layer")
            plt.xlabel("greenscale value")
            plt.xlim([0, 255])
            plt.plot(bin_edges, self.values[:, :, 1].flatten(), color="green")

            plt.subplot(133)
            plt.title("blue layer")
            plt.xlabel("bluescale value")
            plt.xlim([0, 255])
            plt.plot(bin_edges, self.values[:, :, 2].flatten(), color="blue")

            plt.show()

    def to_cumulated(self) -> 'Histogram':
        """
        method that returns cumulative histogram based on the internal state of the object
        """
        pass
