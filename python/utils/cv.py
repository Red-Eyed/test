import cv2
import numpy as np


def norm_minmax(img: np.ndarray):
    img = img.astype(np.float32)
    img = (img - img.min()) / (img.max() - img.min())

    return img


def show(name: str, img: np.ndarray):
    img = norm_minmax(img) * 255

    cv2.imshow(name, img.astype(np.uint8))
