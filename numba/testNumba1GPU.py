import numpy as np
from numba import vectorize
from timeit import default_timer as time


@vectorize(['float32(float32, float32)'], target='cuda')
def Add(a, b):
  return a + b

# Initialize arrays
N = 100000000
A = np.ones(N, dtype=np.float32)
B = np.ones(A.shape, dtype=A.dtype)
C = np.empty_like(A, dtype=A.dtype)

# Add arrays on GPU
s = time()
C = Add(A, B)
e = time()
tcuda = e - s
print('cuda: %f' % tcuda)
