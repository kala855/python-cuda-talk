import cupy as cp
import numpy as np

from timeit import default_timer as time

dA = cp.random.rand(2500,2500)
dB = cp.random.rand(2500,2500)
s = time()
dC = cp.dot(dA,dB)
e = time()
tcuda = e - s

A = np.random.rand(2500,2500)
B = np.random.rand(2500,2500)
s = time()
C = np.dot(A,B)
e = time()
tcpu = e - s

print('cpu:  %f' % tcpu)
print('cuda: %f' % tcuda)
print('cuda speedup: %.2fx' % (tcpu / tcuda))
