from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
import math
from config import config


__all__ = ['SquarePad', 'ContentSquarePad', 'RotationCorrection', 'GaussianBlur']


class SquarePad():
    """
    pad to the square shape
    """

    def __init__(self, fill=0):
        self.fill = 0

    def __call__(self, img):
        w, h = img.size
        l = max(w, h)
        if self.fill == 0:
            color = 0
        elif self.fill == 'mean':
            color = np.mean(np.asarray(img), axis=(0, 1)).astype(np.int)
        square = Image.new('RGB', (l, l), color=color)
        square.paste(img, ((l - w) // 2, (l - h) // 2))
        return square


class ContentSquarePad():
    """
    pad to square shapes by repeating the input image
    """

    def __init__(self):
        pass

    def __call__(self, im):
        im = np.asarray(im)
        h, w, _ = im.shape
        rotated = False
        if h > w:
            h, w = w, h
            im = im.transpose((1, 0, 2))
            rotated = True
        s = math.floor(math.ceil(w/h) / 2) * 2 + 1
        im = np.tile(im, (s, 1, 1))
        h, w, _ = im.shape
        mid = math.floor(h / 2)
        side = mid-math.floor(w/2)
        im = im[side:side+w, :, :]
        if rotated:
            im = im.transpose((1, 0, 2))
        return Image.fromarray(im)


class RotationCorrection():
    """
    make sure the width is not shorter than the height
    """

    def __init__(self):
        pass

    def __call__(self, img):
        w, h = img.size
        if h > w:
            return img.transpose(Image.ROTATE_90)
        else:
            return img


class GaussianBlur():

    def __init__(self, size=5, sigma=5, p=0.5):
        self.size = size
        self.sigma = sigma
        self.p = p

    def __call__(self, img):
        if np.random.rand() > self.p:
            return img
        img = np.asarray(img)
        if isinstance(self.size, list):
            size = np.random.choice(self.size)
        else:
            size = self.size
        if isinstance(self.sigma, list):
            sigma = np.random.choice(self.sigma)
        else:
            sigma = self.sigma
        img = cv2.GaussianBlur(img, (size, size), sigma)
        return Image.fromarray(img)
