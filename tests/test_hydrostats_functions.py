# import numpy as np

# import hat.hydrostats_functions as funcs

# """DUMMY DATA"""


# def ones():
#     "simulation and observation dummy data, all ones"
#     s = np.ones(5)
#     o = np.ones(5)

#     return (s, o)


# def dummy_data():
#     "simulation and observation dummy data"
#     s = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
#     o = np.array([4.0, 5.0, 6.0, 7.0, 8.0])

#     return (s, o)


# """ TESTS """


# def test_apb():
#     s, o = ones()

#     assert np.isclose(funcs.apb(s, o), 0)
#     assert np.isclose(funcs.apb(s * 1.5, o), 50)
#     assert np.isclose(funcs.apb(s * 2, o), 100)


# def test_apb2():
#     s, o = ones()

#     assert np.isclose(funcs.apb2(s, o), 0)
#     assert np.isclose(funcs.apb2(s * 1.5, o), 50)
#     assert np.isclose(funcs.apb2(s * 2, o), 100)


# def test_bias():
#     s, o = ones()

#     assert np.isclose(funcs.bias(s, o), 0)
#     assert np.isclose(funcs.bias(s, o * 0), 1)


# def test_br():
#     s, o = ones()

#     assert np.isclose(funcs.br(s, o), 1)


# def test_correlation():
#     s = np.array([1, 2, 3, 4, 5])
#     o = np.array([2, 3, 4, 5, 6])

#     result = funcs.correlation(s, o)

#     assert np.isclose(result, 1)


# def test_kge():
#     s, o = dummy_data()

#     assert np.isclose(funcs.kge(s, o), -0.1180339887498949)


# def test_index_agreement():
#     s, o = dummy_data()

#     funcs.index_agreement(s, o)

#     assert np.isclose(funcs.index_agreement(s, o), 0.5544554455445545)


# def test_mae():
#     s, o = ones()

#     assert np.isclose(funcs.mae(s, o), 0)


# def test_ns():
#     s, o = dummy_data()

#     assert np.isclose(funcs.ns(s, o), -3.5)


# def test_nslog():
#     s, o = dummy_data()

#     assert np.isclose(funcs.nslog(s, o), -11.59036828886655)


# def test_pc_bias():
#     s, o = dummy_data()

#     assert np.isclose(funcs.pc_bias(s, o), -50.0)


# def test_pc_bias2():
#     s, o = dummy_data()

#     assert np.isclose(funcs.pc_bias2(s, o), -50)


# def test_rmse():
#     s, o = ones()

#     assert np.isclose(funcs.rmse(s, o), 0)


# def test_rsr():
#     s, o = dummy_data()

#     assert np.isclose(funcs.rsr(s, o), 2.1213203435596424)


# def test_vr():
#     s, o = dummy_data()

#     assert np.isclose(funcs.vr(s, o), 0)
