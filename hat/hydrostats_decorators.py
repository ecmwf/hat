"""
Decorators to run checks on hydrological functions
"""

import functools

import numpy as np


def is_two_numpy_arrays(func):
    @functools.wraps(func)
    def wrapper(arr1, arr2, *args, **kwargs):
        if not (isinstance(arr1, np.ndarray) and isinstance(arr2, np.ndarray)):
            raise TypeError("Both inputs must be numpy arrays")
        if args or kwargs:
            raise TypeError("Only two inputs are allowed")
        return func(arr1, arr2)

    return wrapper


def to_float(func):
    @functools.wraps(func)
    def wrapper(arr1: np.ndarray, arr2: np.ndarray):
        return func(arr1.astype(np.float64), arr2.astype(np.float64))
    return wrapper


def filter_nan(func):
    @functools.wraps(func)
    def wrapper(arr1: np.ndarray, arr2: np.ndarray):
        filtered_mask = ~(np.isnan(arr1) | np.isnan(arr2))
        if not np.any(filtered_mask):
            return None
        return func(arr1[filtered_mask], arr2[filtered_mask])
    return wrapper


def hydrostat(func):
    @is_two_numpy_arrays
    @filter_nan
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    return wrapper
