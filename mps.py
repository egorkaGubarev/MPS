import numpy as np

import convert
import utils

L = 10
d = 2

c_1 = np.random.random(d ** L)
c_2 = np.random.random(d ** L)

o = np.zeros((L, d, d))
for k in range(L):
    o[k, :, :] = np.diag(np.random.random(d))
    for i in range(d):
        for j in range(i + 1, d):
            el = np.random.random()
            o[k, i, j] = el
            o[k, j, i] = np.conjugate(el)

c_1 /= np.linalg.norm(c_1)
c_2 /= np.linalg.norm(c_2)

mps_1 = convert.to_left_mps_with_svd(c_1, L, d)
mps_2 = convert.to_left_mps_with_svd(c_2, L, d)

print('-----')
print(f'element = {np.round(utils.count_matrix_element(c_1, c_2, o, L, d), 5)}')
print(f'element = {np.round(utils.count_matrix_element_from_mps(mps_1, mps_2, o, L, d), 5)}')
