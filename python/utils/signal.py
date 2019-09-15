import numpy as np
from scipy.signal import convolve


def find_peaks1d(array: np.ndarray):
    derivative = convolve(array, [1.0, -1.0], "same")

    sign = np.sign(derivative)
    sign_change = convolve(sign, [1.0, -1.0], "same")
    sign_change[[0, -1]] = 0
    peaks = sign_change == sign_change.min()

    return peaks