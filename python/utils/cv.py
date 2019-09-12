import cv2
import numpy as np
import matplotlib.pyplot as plt


def norm_minmax(img: np.ndarray):
    img = img.astype(np.float32)
    img = (img - img.min()) / (img.max() - img.min())

    return img


def show(name: str, img: np.ndarray, cmap: str = "gray"):
    img = norm_minmax(img) * 255

    plt.figure(name)
    plt.imshow(img.astype(np.uint8), cmap=cmap)
