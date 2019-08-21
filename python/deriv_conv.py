import numpy as np
import cv2


def create_sin(shape):
    img = np.ones(shape)
    img = np.linspace(0, shape)
    cv2.imshow("lol", img)
