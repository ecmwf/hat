import numpy as np
import pytest

from hat import hydrostats_decorators


def add_arrays(arr1: np.ndarray, arr2: np.ndarray) -> np.ndarray:
    return arr1 + arr2


def divide_arrays(arr1: np.ndarray, arr2: np.ndarray) -> np.ndarray:
    return arr1 / arr2


def test_is_two_numpy_arrays():
    """test @is_two_numpy_arrays decorator"""
    dummy_func = add_arrays
    decorated = hydrostats_decorators.is_two_numpy_arrays(dummy_func)

    is_numpy = np.array([1, 2, 3])

    # two numpy arrays
    assert np.allclose(decorated(is_numpy, is_numpy), is_numpy * 2)

    # non-numpy arguments
    with pytest.raises(TypeError):
        decorated(is_numpy, "not a numpy array")
    with pytest.raises(TypeError):
        decorated("not a numpy array", is_numpy)

    # not two arguments
    with pytest.raises(TypeError):
        decorated(is_numpy)
    with pytest.raises(TypeError):
        decorated(is_numpy, is_numpy, is_numpy)


def test_filter_nan():
    """test @filter_nan decorator"""
    dummy_func = add_arrays
    decorated = hydrostats_decorators.filter_nan(dummy_func)

    arr = np.array([1, 2, 3])
    nan1 = np.array([np.nan, 2, 3])
    nan2 = np.array([1, np.nan, 3])
    nan3 = np.array([1, 2, np.nan])
    nans = np.array([np.nan, np.nan, np.nan])

    # two clean arrays
    assert np.allclose(decorated(arr, arr), arr * 2)

    # nan in various positions
    assert np.allclose(decorated(arr, nan1), np.array([4, 6]))
    assert np.allclose(decorated(arr, nan2), np.array([2, 6]))
    assert np.allclose(decorated(arr, nan3), np.array([2, 4]))
    assert np.allclose(decorated(nan1, arr), np.array([4, 6]))
    assert np.allclose(decorated(nan2, arr), np.array([2, 6]))
    assert np.allclose(decorated(nan3, arr), np.array([2, 4]))

    # nan in both arrays at same element
    assert np.allclose(decorated(nan1, nan1), np.array([4, 6]))
    assert np.allclose(decorated(nan2, nan2), np.array([2, 6]))
    assert np.allclose(decorated(nan3, nan3), np.array([2, 4]))

    # all nans
    assert decorated(arr, nans) is None
    assert decorated(nans, arr) is None


# def test_handle_divide_by_zero_error():
#     """test for @handle_divide_by_zero_error decorator
#     - tries to gracefully handle divide by zero errors
#     - only raises error if every element is divided by zero
#     """
#     dummy_func = divide_arrays
#     decorated = hydrostats_decorators.handle_divide_by_zero_error(dummy_func)

#     ones = np.ones(3)
#     zeros = np.zeros(3)
#     arr = np.array([1., 2., 3.])

#     # ones divide by ones
#     assert np.allclose(decorated(ones, ones), ones)

#     # zeros divided by ones (will not raise an error)
#     assert np.allclose(decorated(zeros, ones), zeros)

#     # all zeros division error
#     with pytest.raises(ZeroDivisionError):
#         decorated(ones, zeros)

#     # zeros division in not all elements
#     result = decorated(arr, np.array([0., 1., 1.]))
#     assert np.isnan(result[0])
#     assert result[1] == 2.0
#     assert result[2] == 3.0


def test_hydrostat():
    "tests for @hydrostat decorator, which is a chain of previous decorators"
    decorated_add = hydrostats_decorators.hydrostat(add_arrays)
    decorated_divide = hydrostats_decorators.hydrostat(divide_arrays)

    zeros = np.zeros(3)
    ones = np.ones(3)
    arr = np.array([1, 2, 3])
    nan1 = np.array([np.nan, 2, 3])
    nans = np.array([np.nan, np.nan, np.nan])

    # simple addition
    assert np.allclose(decorated_add(zeros, zeros), zeros)
    assert np.allclose(decorated_add(ones, ones), ones * 2)

    # simple division
    assert np.allclose(decorated_divide(ones, ones), ones)
    assert np.allclose(decorated_divide(zeros, ones), zeros)

    # addition with nans
    assert np.allclose(decorated_add(nan1, nan1), [4, 6])
    assert np.allclose(decorated_add(nan1, arr), [4, 6])
    assert np.allclose(decorated_add(arr, nan1), [4, 6])

    # # all zero division
    # with pytest.raises(ZeroDivisionError):
    #     print(decorated_divide(ones, zeros))

    # all nans
    assert decorated_divide(nans, nans) is None
