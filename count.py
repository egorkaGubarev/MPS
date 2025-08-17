import numpy as np


def overlap_from_left(mps_1, mps_2):
    d = mps_1[0].shape[0]
    over = np.eye(1)
    for site in range(len(mps_1)):
        if len(mps_1[site].shape) == 3:
            core = 0
            for sigma in range(d):
                core += np.conjugate(mps_2[site][sigma]).T @ over @ mps_1[site][sigma]

            over = core
        else:
            over = np.conjugate(mps_2[site]).T @ over @ mps_1[site]
    return np.array(over)[0][0]


def overlap_from_right(mps_1, mps_2):
    d = mps_1[0].shape[0]
    over = np.eye(1)
    for site in reversed(range(len(mps_1))):
        if len(mps_1[site].shape) == 3:
            core = 0
            for sigma in range(d):
                core += mps_2[site][sigma] @ over @ np.conjugate(mps_1[site][sigma]).T

            over = core
        else:
            over = mps_2[site] @ over @ np.conjugate(mps_1[site]).T
    return np.array(over)[0][0]


def matrix_element(c_1, c_2, o):
    el = 0
    sites = o.shape[0]
    d = o.shape[1]
    for i in range(d ** sites):
        sigma_i = count_sigma(i, d, sites)
        for j in range(d ** sites):
            sigma_j = count_sigma(j, d, sites)

            prod = 1
            for k in range(sites):
                prod *= o[k, int(sigma_i[k]), int(sigma_j[k])]

            el += np.conjugate(c_1[i]) * c_2[j] * prod

    return el


def matrix_element_from_mps_from_left(mps_1, mps_2, o):
    d = mps_1[0].shape[0]
    el = np.eye(1)
    shift = 0
    for site in range(len(mps_1)):
        if len(mps_1[site].shape) == 3:
            core = 0
            for sigma_1 in range(d):
                prod = np.conjugate(mps_1[site][sigma_1]).T @ el
                for sigma_2 in range(d):
                    core += o[site - shift, sigma_1, sigma_2] * prod @ mps_2[site][sigma_2]
            el = core
        else:
            el = np.conjugate(mps_1[site]).T @ el @ mps_2[site]
            shift = 1
    return np.array(el)[0, 0]


def matrix_element_from_mps_from_right(mps_1, mps_2, o):
    d = mps_1[0].shape[0]
    el = np.eye(1)
    mat_am = len(mps_1)
    shift = mat_am - o.shape[0]
    for site in reversed(range(mat_am)):
        if len(mps_1[site].shape) == 3:
            core = 0
            for sigma_1 in range(d):
                prod = mps_1[site][sigma_1] @ el
                for sigma_2 in range(d):
                    core += o[site - shift, sigma_1, sigma_2] * prod @ np.conjugate(mps_2[site][sigma_2]).T
            el = core
        else:
            el = mps_1[site] @ el @ np.conjugate(mps_2[site]).T
            shift = 0
    return np.array(el)[0, 0]


def matrix_element_loc_from_mps(mps, o, loc):
    d = mps[0].shape[0]
    el = 0
    loc_matr = mps[loc]
    sing = mps[loc + 1]
    for sigma_1 in range(d):
        prod = loc_matr[sigma_1] @ sing
        for sigma_2 in range(d):
            el += o[sigma_1, sigma_2] * np.trace(np.conjugate(loc_matr[sigma_2] @ sing).T @ prod)
    return el


def count_sigma(x, d, sites):
    sigma = np.zeros(sites)
    digit = 1
    while x > 0:
        sigma[sites - digit] = x % d
        x //= d
        digit += 1
    return sigma
