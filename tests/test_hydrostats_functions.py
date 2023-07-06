import numpy as np

import hat.hydrostats_functions as funcs

"""DUMMY DATA"""


def ones():
    "simulation and observation dummy data, all ones"
    s = np.ones(5)
    o = np.ones(5)

    return (s, o)


def dummy_data():
    "simulation and observation dummy data"
    s = np.array([1, 2, 3, 4, 5])
    o = np.array([4, 5, 6, 7, 8])

    return (s, o)


""" TESTS """


def test_apb():
    # has been decorated
    assert hasattr(funcs.apb, "_is_two_numpy_arrays")
    assert hasattr(funcs.apb, "_filtered_nan")

    s, o = ones()

    assert np.isclose(funcs.apb(s, o), 0)
    assert np.isclose(funcs.apb(s * 1.5, o), 50)
    assert np.isclose(funcs.apb(s * 2, o), 100)


def test_apb2():
    # has been decorated
    assert hasattr(funcs.apb2, "_is_two_numpy_arrays")
    assert hasattr(funcs.apb2, "_filtered_nan")

    s, o = ones()

    assert np.isclose(funcs.apb2(s, o), 0)
    assert np.isclose(funcs.apb2(s * 1.5, o), 50)
    assert np.isclose(funcs.apb2(s * 2, o), 100)


def test_bias():
    # has been decorated
    assert hasattr(funcs.bias, "_is_two_numpy_arrays")
    assert hasattr(funcs.bias, "_filtered_nan")

    s, o = ones()

    assert np.isclose(funcs.bias(s, o), 0)
    assert np.isclose(funcs.bias(s, o * 0), 1)


def test_br():
    # has been decorated
    assert hasattr(funcs.br, "_is_two_numpy_arrays")
    assert hasattr(funcs.br, "_filtered_nan")

    s, o = ones()

    assert np.isclose(funcs.br(s, o), 1)


def test_correlation():
    # has been decorated
    assert hasattr(funcs.correlation, "_is_two_numpy_arrays")
    assert hasattr(funcs.correlation, "_filtered_nan")

    s = np.array([1, 2, 3, 4, 5])
    o = np.array([2, 3, 4, 5, 6])

    result = funcs.correlation(s, o)

    assert np.isclose(result, 1)


def test_kge():
    # has been decorated
    assert hasattr(funcs.kge, "_is_two_numpy_arrays")
    assert hasattr(funcs.kge, "_filtered_nan")

    s, o = dummy_data()

    assert np.isclose(funcs.kge(s, o), -0.1180339887498949)


def test_index_agreement():
    # has been decorated
    assert hasattr(funcs.index_agreement, "_is_two_numpy_arrays")
    assert hasattr(funcs.index_agreement, "_filtered_nan")

    s, o = dummy_data()

    funcs.index_agreement(s, o)

    assert np.isclose(funcs.index_agreement(s, o), 0.5544554455445545)


def test_mae():
    # has been decorated
    assert hasattr(funcs.mae, "_is_two_numpy_arrays")
    assert hasattr(funcs.mae, "_filtered_nan")

    s, o = ones()

    assert np.isclose(funcs.mae(s, o), 0)


def test_ns():
    # has been decorated
    assert hasattr(funcs.ns, "_is_two_numpy_arrays")
    assert hasattr(funcs.ns, "_filtered_nan")

    s, o = dummy_data()

    assert np.isclose(funcs.ns(s, o), -3.5)


def test_nslog():
    # has been decorated
    assert hasattr(funcs.nslog, "_is_two_numpy_arrays")
    assert hasattr(funcs.nslog, "_filtered_nan")

    s, o = dummy_data()

    assert np.isclose(funcs.nslog(s, o), -11.59036828886655)


def test_pc_bias():
    # has been decorated
    assert hasattr(funcs.pc_bias, "_is_two_numpy_arrays")
    assert hasattr(funcs.pc_bias, "_filtered_nan")

    s, o = dummy_data()

    assert np.isclose(funcs.pc_bias(s, o), -50.0)


def test_pc_bias2():
    # has been decorated
    assert hasattr(funcs.pc_bias2, "_is_two_numpy_arrays")
    assert hasattr(funcs.pc_bias2, "_filtered_nan")

    s, o = dummy_data()

    assert np.isclose(funcs.pc_bias2(s, o), -50)


def test_rmse():
    # has been decorated
    assert hasattr(funcs.rmse, "_is_two_numpy_arrays")
    assert hasattr(funcs.rmse, "_filtered_nan")

    s, o = ones()

    assert np.isclose(funcs.rmse(s, o), 0)


def test_rsr():
    # has been decorated
    assert hasattr(funcs.rsr, "_is_two_numpy_arrays")
    assert hasattr(funcs.rsr, "_filtered_nan")

    s, o = dummy_data()

    assert np.isclose(funcs.rsr(s, o), 2.1213203435596424)


def test_vr():
    # has been decorated
    assert hasattr(funcs.vr, "_is_two_numpy_arrays")
    assert hasattr(funcs.vr, "_filtered_nan")

    s, o = dummy_data()

    assert np.isclose(funcs.vr(s, o), 0)
