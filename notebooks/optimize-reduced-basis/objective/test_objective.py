from objective import get_global_index

import numpy as np

q_range = [1, 8]
q_size = 100
q_params = np.linspace(*q_range, q_size)

chi_range = [-0.8, 0.8]
chi_size = 20
chi_params = np.linspace(*chi_range, chi_size)


def test_global_index_1d_case1():
    index = 10
    automatic_index = get_global_index({"q_index": q_size}, {"q_index": index})
    assert q_params[index] == q_params[automatic_index]


def test_global_index_1d_case2():
    index = 15
    automatic_index = get_global_index({"q_index": q_size}, {"q_index": index})
    assert q_params[index] == q_params[automatic_index]


def test_global_index_1d_case3():
    index = 15
    automatic_index = get_global_index({"q_index": q_size}, {"q_index": index})
    assert q_params[index] == q_params[automatic_index]


def test_global_index_2d_case1():
    q_index = 15
    chi_index = 4

    params_2d = [[q, chi] for q in q_params for chi in chi_params]
    automatic_index = get_global_index(
        {"q_index": q_size, "chi_index": chi_size},
        {"q_index": q_index, "chi_index": chi_index},
    )
    assert params_2d[automatic_index] == [q_params[q_index], chi_params[chi_index]]


def test_global_index_2d_case2():
    chi_index = 4
    q_index = 15

    params_2d = [[chi, q] for chi in chi_params for q in q_params]
    automatic_index = get_global_index(
        {"chi_index": chi_size, "q_index": q_size},
        {"chi_index": chi_index, "q_index": q_index},
    )
    assert params_2d[automatic_index] == [chi_params[chi_index], q_params[q_index]]


def test_global_index_3d():
    q_index = 24
    chi1_index = 4
    chi2_index = 8

    params_3d = [
        [q, chi1, chi2] for q in q_params for chi1 in chi_params for chi2 in chi_params
    ]
    automatic_index = get_global_index(
        {"q_index": q_size, "chi1_index": chi_size, "chi2_index": chi_size},
        {"q_index": q_index, "chi1_index": chi1_index, "chi2_index": chi2_index},
    )
    assert params_3d[automatic_index] == [
        q_params[q_index],
        chi_params[chi1_index],
        chi_params[chi2_index],
    ]
