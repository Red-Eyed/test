from time import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn import functional as F

from utils.cv import show

if __name__ == '__main__':
    img = cv2.imread("/home/vstupakov/Downloads/image (1).png", cv2.IMREAD_GRAYSCALE).astype(np.float32)
    show("orig", img)
    torch.set_default_tensor_type(torch.FloatTensor)

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
    plt.imshow(kernel, cmap="jet")
    kernel = kernel.reshape((1, 1, *kernel.shape))

    out = F.conv2d(img_tensor, kernel)
    filter_gauss = np.squeeze(out.cpu().numpy())
    show("gauss", filter_gauss)

    plt.show()
    cv2.waitKey()
