import numpy as np
import random

min_dim_pow = 2
max_dim_pow = 3

Na = 2 ** random.randint(min_dim_pow, max_dim_pow)
Nb = 2 ** random.randint(min_dim_pow, max_dim_pow)

psi = np.random.random((Na, Nb))
u, s, v = np.linalg.svd(psi)

print(f'Na: {Na}')
print(f'Nb: {Nb}')
print(f'psi: {psi}')
print(f'u: {u}')
print(f's: {s}')
print(f'v: {v}')
