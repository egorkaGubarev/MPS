import numpy as np

def approx(m, r):
    u, s, v = np.linalg.svd(m)

    u = u[:, :r]
    s = s[:r]
    v = v[:r, :]

    return u @ np.diag(s) @ v

def convert_to_mps(c, sites, d):
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

def convert_from_mps(a):
    sites =len(a)
    d = a[0].shape[0]

    c = np.zeros(d ** sites)
    for i in range(d ** sites):
        sigma = count_sigma(i, d, sites)
        mps = np.eye(1)

        for j in range(sites):
            mps_new = mps @ a[j][int(sigma[j]), :, :]
            mps = mps_new

        c[i] = mps[0, 0]

    return c

def count_sigma(x, d, sites):
    sigma = np.zeros(sites)
    digit = 1
    while x > 0:
        sigma[sites - digit] = x % d
        x //= d
        digit += 1
    return sigma
