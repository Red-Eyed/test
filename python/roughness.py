import cv2
import matplotlib.pyplot as plt
import numpy as np

from utils.cv import show, norm_minmax


def preprocess(img: np.ndarray):
    img = norm_minmax(img)

    mask1 = img > np.mean(img) / 1.5
    mask2 = np.logical_not(mask1)

    img1 = img * mask1
    img2 = img * mask2

    std1, mean1 = np.std(img1), np.mean(img1)
    std2, mean2 = np.std(img2), np.mean(img2)

    img1 /= std1
    img2 /= std2

    diff = np.abs(mean1 - mean2)
    if mean1 > mean2:
        img2 += diff * mask2
    else:
        img1 += diff * mask1

    img3 = img1 + img2

    return img3


def roughness(img):
    s1 = cv2.Sobel(img, cv2.CV_32F, 1, 0, 5)
    s2 = cv2.Sobel(img, cv2.CV_32F, 0, 1, 5)
    sobel = 0.5 * s1 + 0.5 * s2

    sobel = np.abs(sobel)
    mean, std = np.mean(sobel), np.std(sobel)

    mask1 = sobel > (mean + std)
    mask2 = sobel < (mean - std)
    mask = np.logical_or(mask1, mask2)

    masked = sobel * mask
    non_zero = np.sum(masked > 0)

    rough = non_zero / np.prod(img.shape)

    return rough


if __name__ == '__main__':
    img = cv2.imread("/home/redeyed/Desktop/Tile_001_00131.png", cv2.IMREAD_GRAYSCALE).astype(np.float32)
    show("orig", img)

    preprocessed = preprocess(img)
    show("preprocessed", preprocessed)

    percent = np.round(roughness(preprocessed), 2)
    print(f"roughness = {percent}")

    plt.show()
