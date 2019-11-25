import numpy as np
from scipy.signal import convolve


def find_peaks1d(array: np.ndarray):
    kernel = np.float32([1, -1])
    darray = convolve(array, kernel, "same")

    sign = np.where(darray <= 0, -1, 1)
    dsign = convolve(sign, kernel, "same")

    # removing zero padding
    dsign[[0, -1]] = 0

    peaks = dsign == dsign.min()

    return np.uint32(peaks)


def running_avarage(array: np.ndarray, points_num):
    kernel = np.ones(points_num) / points_num
    result = convolve(array, kernel, "same")
    return result
