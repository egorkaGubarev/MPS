import numpy as np


def approx(m, r):
    u, s, v = np.linalg.svd(m)

    u = u[:, :r]
    s = s[:r]
    v = v[:r, :]

    return u @ np.diag(s) @ v


def count_overlap(mps_1, mps_2, d, sites):
    over = 0
    for sigma in range(d):
        over += np.conjugate(mps_2[0][sigma]).T @ mps_1[0][sigma]

    for site in range(1, sites):
        core = 0
        for sigma in range(d):
            core += np.conjugate(mps_2[site][sigma]).T @ over @ mps_1[site][sigma]

        over = core

    return np.array(over)[0][0]


def count_overlap_mixed(mps_1, mps_2, d, sites, singul_bond):
    over = 0
    for sigma in range(d):
        over += np.conjugate(mps_2[0][sigma]).T @ mps_1[0][sigma]

    for site in range(1, singul_bond + 1):
        core = 0
        for sigma in range(d):
            core += np.conjugate(mps_2[site][sigma]).T @ over @ mps_1[site][sigma]

        over = core

    over = np.conjugate(mps_2[singul_bond + 1]).T @ over @ mps_1[singul_bond + 1]

    for site in range(singul_bond + 2, sites + 1):
        core = 0
        for sigma in range(d):
            core += np.conjugate(mps_2[site][sigma]).T @ over @ mps_1[site][sigma]

        over = core

    return np.array(over)[0][0]


def count_matrix_element(c_1, c_2, o, sites, d):
    el = 0
    for i in range(d ** sites):
        sigma_i = count_sigma(i, d, sites)
        for j in range(d ** sites):
            sigma_j = count_sigma(j, d, sites)

            prod = 1
            for k in range(sites):
                prod *= o[k, int(sigma_i[k]), int(sigma_j[k])]

            el += np.conjugate(c_1[i]) * c_2[j] * prod

    return el


def count_matrix_element_from_mps(mps_1, mps_2, o, sites, d):
    el = 0
    for sigma_1 in range(d):
        for sigma_2 in range(d):
            el += o[0, sigma_2, sigma_1] * np.conjugate(mps_2[0][sigma_2]).T @ mps_1[0][sigma_1]

    for site in range(1, sites):
        core = 0
        for sigma_1 in range(d):
            for sigma_2 in range(d):
                core += o[site, sigma_2, sigma_1] * np.conjugate(mps_2[site][sigma_2]).T @ el @ mps_1[site][sigma_1]

        el = core

    return np.array(el)[0, 0]


def count_sigma(x, d, sites):
    sigma = np.zeros(sites)
    digit = 1
    while x > 0:
        sigma[sites - digit] = x % d
        x //= d
        digit += 1
    return sigma
