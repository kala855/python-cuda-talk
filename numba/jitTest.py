from numba import jit

@jit
def f(x,y):
	# A somewhat trivial example
	return x + y
