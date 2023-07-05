"""
HYDROSTATS DECORATORS

Decorators to run checks on hydrological functions

NOTE consider adding the "hydrostat" decorators to new custom functions
to get extract checks and safe guards for free

e.g.

@hydrostat
def my_cool_new_function(s,o):
    # do something
    return result

"""

import functools

import numpy as np


def is_two_numpy_arrays(func):
    """decorator to check the arguments are two numpy arrays"""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if len(args) != 2:
            raise TypeError("Not enough arguments. Expected 2 numpy arrays.")

        if not (isinstance(args[0], np.ndarray) and isinstance(args[1], np.ndarray)):
            raise TypeError(f"Both arguments to {func} must be numpy arrays.")

        return func(*args, **kwargs)

    # this variable is used to test if all (including new future)
    # analytical functions have this decorator
    wrapper._is_two_numpy_arrays = True
    return wrapper


def filter_nan(func):
    """decorator to filter not a number (NaN) from arrays"""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        s, o = args[:2]

        data = np.array([s.flatten(), o.flatten()])
        data = np.transpose(data)
        data = data[~np.isnan(data).any(1)]

        # TODO document this behaviour..
        # NOTE don't raise if all nans? or return empty dataset

        s, o = data[:, 0], data[:, 1]

        return func(s, o, **kwargs)

    wrapper._filtered_nan = True
    return wrapper


# def handle_divide_by_zero_error(func):
#     """decorator to return NaN if dividing by zero"""
#     @functools.wraps(func)
#     def wrapper(*args, **kwargs):

#         s, o = args[:2]

#         try:
#             func(s,o, **kwargs)
#         except ZeroDivisionError:
#             return 0

#     wrapper._handle_zero_errors = True
#     return wrapper


def hydrostat(func):
    """
    single decorator which chains individual decorators

    NOTE decorators are applied in order (from top to bottom)
    """

    @is_two_numpy_arrays
    @filter_nan
    # @handle_divide_by_zero_error
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    return wrapper
