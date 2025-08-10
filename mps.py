import numpy as np

import utils

L = 10
d = 2

c = np.random.random(d ** L)
a = utils.convert_to_mps(c, L, d)

print(a[0])
