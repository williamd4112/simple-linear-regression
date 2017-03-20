import numpy as np
from util import load_csv

import sys

y = load_csv(sys.argv[1])
t = load_csv(sys.argv[2])

assert len(y) == len(t)

mse = np.sum((y - t)**2) / (2 *(len(y)))

print mse
