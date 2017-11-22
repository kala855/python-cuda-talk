from timeit import default_timer as time
from numba import cuda
import numpy as np

bpg = 100
tpb = 32

n = bpg * tpb

@cuda.jit
def matmul(A, B, C):
    """Perform square matrix multiplication of C = A * B
    """
    i, j = cuda.grid(2)
    if i < C.shape[0] and j < C.shape[1]:
        tmp = 0.
        for k in range(A.shape[1]):
            tmp += A[i, k] * B[k, j]
        C[i, j] = tmp


A = np.array(np.random.random((n, n)), dtype=np.float32)
B = np.array(np.random.random((n, n)), dtype=np.float32)
C = np.empty_like(A)
print("N = %d x %d" % (n, n))
stream = cuda.stream()
with stream.auto_synchronize():
	dA = cuda.to_device(A, stream)
	dB = cuda.to_device(B, stream)
	dC = cuda.to_device(C, stream)
	s = time()
	matmul[(bpg, bpg), (tpb, tpb), stream](dA, dB, dC)
	e = time()
	dC.to_host(stream)
tcuda = e - s
# Host compute
Amat = np.matrix(A)
Bmat = np.matrix(B)
s = time()
Cans = Amat * Bmat
e = time()
tcpu = e - s
# Check result
assert np.allclose(C, Cans)
print('cpu:  %f' % tcpu)
print('cuda: %f' % tcuda)
print('cuda speedup: %.2fx' % (tcpu / tcuda))
