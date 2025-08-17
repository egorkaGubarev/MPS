import numpy as np


def approx(m, r):
    u, s, v = np.linalg.svd(m)

    u = u[:, :r]
    s = s[:r]
    v = v[:r, :]

    return u @ np.diag(s) @ v


def create_oper(sites, d):
    o = np.zeros((sites, d, d))
    for k in range(sites):
        o[k, :, :] = create_oper_loc(d)
    return o


def create_oper_loc(d):
    o = np.diag(np.random.random(d))
    for i in range(d):
        for j in range(i + 1, d):
            el = np.random.random()
            o[i, j] = el
            o[j, i] = np.conjugate(el)
    return o
