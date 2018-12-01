import numpy as np


def img_cumsum(img):
    return (np.float32(img) - img.mean()).cumsum(axis=0)
