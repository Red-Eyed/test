from time import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn import functional as F

from utils.cv import show


def sin2d(size, period, phase):
    plane = torch.arange(0., size, 1.)
    _, plane = torch.meshgrid([plane, plane])
    plane = torch.sin(period * plane + phase)
    return plane


if __name__ == '__main__':
    torch.set_default_tensor_type(torch.FloatTensor)
    img = sin2d(800, 1 / 100, 0)
    show("orig", img.numpy())

    w, h = 10, 10
    # mean filter
    kernel = torch.ones(w * h).reshape((1, 1, w, h))
    img_tensor = torch.tensor(img).reshape((1, 1, *img.shape))

    start2 = time()
    out = F.conv2d(img_tensor, kernel)

    filter_mean = np.squeeze(out.cpu().numpy())
    show("mean", filter_mean)

    # gauss filter
    sigma = 10
    dx = 0.1
    dy = 0.1
    x = torch.arange(-6, 6, dx)
    y = torch.arange(-6, 6, dy)
    x2d, y2d = torch.meshgrid([x, y])
    kernel = torch.exp(-(x2d ** 2 + y2d ** 2) / (2 * sigma ** 2))
    show("gauss kernel", kernel.numpy(), cmap="jet")
    kernel = kernel.reshape((1, 1, *kernel.shape))
    out = F.conv2d(img_tensor, kernel)
    show("gauss", np.squeeze(out.cpu().numpy()))

    # derivative
    kernel = torch.tensor([1., -1.]).reshape(1, 1, 1, 2)
    out = F.conv2d(img_tensor, kernel)
    show("derive", np.squeeze(out.cpu().numpy()))

    plt.show()
    cv2.waitKey()
