import numpy as np
import time

import convert
import count
import utils

L = 10
d = 2
bound = 2
class_needed = True

c_1 = np.random.random(d ** L)
c_2 = np.random.random(d ** L)

c_1 /= np.linalg.norm(c_1)
c_2 /= np.linalg.norm(c_2)

start = time.time()

mps_1 = convert.to_left_mps_with_svd(c_1, L, d)
mps_2 = convert.to_left_mps_with_svd(c_2, L, d)

bound_time = time.time()

mps_1_bound = convert.to_left_mps_with_svd(c_1, L, d, bound)
mps_2_bound = convert.to_left_mps_with_svd(c_2, L, d, bound)

end = time.time()

print(f'MPS  generation time: {np.round(bound_time - start, 2)} s')
print(f'Bond MPS  generation time: {np.round(end - bound_time, 2)} s')

o = utils.create_oper(L, d)

if class_needed:
    start = time.time()
    print(f'Class: {np.round(count.matrix_element(c_1, c_2, o), 5)}')
classic = time.time()
print(f'MPS: {np.round(count.matrix_element_from_mps_from_left(mps_1, mps_2, o), 5)}')
mps_time = time.time()
print(f'Bound MPS: {np.round(count.matrix_element_from_mps_from_left(mps_1_bound, mps_2_bound, o), 5)}')
bound_mps_time = time.time()

if class_needed:
    print(f'Class time: {np.round(classic - start, 2)} s')
print(f'MPS  time: {np.round(mps_time - classic, 2)} s')
print(f'Bound MPS  time: {np.round(mps_time - classic, 2)} s')
