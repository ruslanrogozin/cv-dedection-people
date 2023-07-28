import numpy as np
import skimage
from skimage import io, transform
from matplotlib import pyplot as plt
from PIL import Image

class Normalize:

    def __init__(self, mean=128, std=128):
        self.mean = mean
        self.std = std

    def __call__(self, image):
        img = skimage.img_as_float(np.array(image))

        io.imshow(img)
        plt.show()
        image = (img * 256 - self.mean) / self.std
        io.imshow(image)
        plt.show()
        return image
