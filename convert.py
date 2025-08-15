import numpy as np

import utils


def to_left_mps_with_qr(c, sites, d):
    steps = sites - 1
    a = []
    rank = 1
    psi = c.reshape(d, d ** (sites - 1))

    for i in range(steps):
        q, psi = np.linalg.qr(psi)
        rank_prev = rank
        rank = q.shape[1]
        a.append(np.zeros((d, rank_prev, rank)))

        for string in range(q.shape[0] // d):
            for sigma in range(d):
                a[i][sigma, string, :] = q[string * d + sigma, :]

        if i < steps - 1:
            psi = psi.reshape(d * rank, d ** (sites - i - 2))

    a.append(np.zeros((d, d, 1)))
    for sigma in range(d):
        a[steps][sigma, :, 0] = psi[:, sigma]

    return a


def to_left_mps_with_svd(c, sites, d):
    steps = sites - 1
    a = []
    rank = 1
    psi = c.reshape(d, d ** (sites - 1))

    for i in range(steps):
        u, s, v = np.linalg.svd(psi, full_matrices=False)
        rank_prev = rank
        rank = u.shape[1]
        a.append(np.zeros((d, rank_prev, rank)))

        for string in range(u.shape[0] // d):
            for sigma in range(d):
                a[i][sigma, string, :] = u[string * d + sigma, :]

        psi = np.diag(s) @ v
        if i < steps - 1:
            psi = psi.reshape(d * rank, d ** (sites - i - 2))

    a.append(np.zeros((d, d, 1)))
    for sigma in range(d):
        a[steps][sigma, :, 0] = psi[:, sigma]

    return a


def to_mixed_mps_with_svd(c, sites, d, singul_bond):
    a = []
    b = []

    rank = 1
    psi = c.reshape(d, d ** (sites - 1))
    s = None
    v = None

    for i in range(singul_bond + 1):
        u, s, v = np.linalg.svd(psi, full_matrices=False)
        rank_prev = rank
        rank = u.shape[1]
        a.append(np.zeros((d, rank_prev, rank)))

        for string in range(u.shape[0] // d):
            for sigma in range(d):
                a[i][sigma, string, :] = u[string * d + sigma, :]

        if i < singul_bond:
            psi = (np.diag(s) @ v).reshape(d * rank, d ** (sites - i - 2))

    a.append(np.diag(s).reshape(rank, rank))
    psi = v.reshape(rank * d ** (sites - singul_bond - 2), d)
    rank = 1

    for i in range(sites - singul_bond - 2):
        u, s, v = np.linalg.svd(psi, full_matrices=False)
        rank_prev = rank
        rank = u.shape[1]
        b.append(np.zeros((d, rank, rank_prev)))
        columns_in_block = v.shape[1] // d

        for sigma in range(d):
            b[i][sigma, :, :] = v[:, columns_in_block * sigma: columns_in_block * (sigma + 1)]

        psi = u @ np.diag(s)

        if i < sites - singul_bond - 3:
            psi = psi.reshape(-1, d * rank)

    b.append(np.zeros((d, a[-1].shape[1], rank)))

    if sites - singul_bond == 2:
        for sigma in range(d):
            b[sites - singul_bond - 2][sigma, :, 0] = psi[:, sigma]
    else:
        for string in range(psi.shape[0] // d):
            for sigma in range(d):
                b[-1][sigma, string, :] = psi[string * d + sigma, :]

    return a + list(reversed(b))


def to_right_mps_with_qr(c, sites, d):
    steps = sites - 1
    b = []
    rank = 1
    psi = c.reshape(d ** (sites - 1), d)

    for i in range(steps):
        q, psi = np.linalg.qr(np.conjugate(psi.T))
        q = np.conjugate(q.T)
        psi = np.conjugate(psi.T)
        rank_prev = rank
        rank = q.shape[0]
        b.append(np.zeros((d, rank, rank_prev)))
        columns_in_block = q.shape[1] // d

        for sigma in range(d):
            b[i][sigma, :, :] = q[:, columns_in_block * sigma: columns_in_block * (sigma + 1)]

        if i < steps - 1:
            psi = psi.reshape(d ** (sites - i - 2), d * rank)

    b.append(np.zeros((d, 1, rank)))
    for sigma in range(d):
        b[steps][sigma, 0, :] = psi[sigma, :]

    return list(reversed(b))


def to_right_mps_with_svd(c, sites, d):
    steps = sites - 1
    b = []
    rank = 1
    psi = c.reshape(d ** (sites - 1), d)

    for i in range(steps):
        u, s, v = np.linalg.svd(psi, full_matrices=False)
        rank_prev = rank
        rank = u.shape[1]
        b.append(np.zeros((d, rank, rank_prev)))
        columns_in_block = v.shape[1] // d

        for sigma in range(d):
            b[i][sigma, :, :] = v[:, columns_in_block * sigma: columns_in_block * (sigma + 1)]

        psi = u @ np.diag(s)
        if i < steps - 1:
            psi = psi.reshape(d ** (sites - i - 2), d * rank)

    b.append(np.zeros((d, 1, rank)))
    for sigma in range(d):
        b[steps][sigma, 0, :] = psi[sigma, :]

    return list(reversed(b))


def from_mps(a):
    sites = len(a)
    d = a[0].shape[0]

    c = np.zeros(d ** sites)
    for i in range(d ** sites):
        sigma = utils.count_sigma(i, d, sites)
        mps = np.eye(1)

        for j in range(sites):
            mps_new = mps @ a[j][int(sigma[j]), :, :]
            mps = mps_new

        c[i] = mps[0, 0]

    return c


def from_mps_mixed(a, singul_bond):
    sites = len(a) - 1
    d = a[0].shape[0]

    c = np.zeros(d ** sites)
    for i in range(d ** sites):
        sigma = utils.count_sigma(i, d, sites)
        mps = np.eye(1)

        for j in range(singul_bond + 1):
            mps_new = mps @ a[j][int(sigma[j]), :, :]
            mps = mps_new

        mps = mps @ a[singul_bond + 1][:, :]

        for j in range(singul_bond + 2, sites + 1):
            mps_new = mps @ a[j][int(sigma[j - 1]), :, :]
            mps = mps_new

        c[i] = mps[0, 0]

    return c
