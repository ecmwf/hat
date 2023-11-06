"""
HYDROSTATS DECORATORS

Decorators to run checks on hydrological functions

NOTE consider adding the "hydrostat" decorators to new custom functions
to get checks and safe guards for free

e.g.

@hydrostat
def my_cool_new_function(s,o):
    # do something
    return result

"""

import functools

import numpy as np


def is_two_numpy_arrays(func):
    # checks the args are two numpy arrays
    @functools.wraps(func)
    def wrapper(arr1, arr2, *args, **kwargs):
        if not (isinstance(arr1, np.ndarray) and isinstance(arr2, np.ndarray)):
            raise TypeError("Both inputs must be numpy arrays")
        if args or kwargs:
            raise TypeError("Only two inputs are allowed")
        return func(arr1, arr2)

    return wrapper


def to_float(func):
    # force floating point numpy arrays
    @functools.wraps(func)
    def wrapper(arr1: np.ndarray, arr2: np.ndarray):
        return func(arr1.astype(float), arr2.astype(float))

    return wrapper


def filter_nan(func):
    @functools.wraps(func)
    def wrapper(arr1: np.ndarray, arr2: np.ndarray):
        nan_mask1 = np.isnan(arr1)
        nan_mask2 = np.isnan(arr2)
        filtered_mask = ~(nan_mask1 | nan_mask2)
        if not np.any(filtered_mask):
            return None
        return func(arr1[filtered_mask], arr2[filtered_mask])

    return wrapper


def hydrostat(func):
    """
    single decorator which chains individual decorators

    NOTE decorators are applied in order (from top to bottom)
    """

    @is_two_numpy_arrays
    # @to_float
    @filter_nan
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    return wrapper
