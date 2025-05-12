import numpy as np


def percentage_error(val, grid):
    return np.abs((val - grid) / val)
