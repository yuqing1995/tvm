'''
Implement FIR filter with TVM 
'''
from __future__ import absolute_import, print_function
import time
import tvm
import numpy as np
#declear the global enviroment
tgt = 'llvm'
# implement a N-tap FIR filter, (N=11)
n = 11
m = 3
#define C as a coefficient function for FIR
C = tvm.placeholder((n,), name='C')
X = tvm.placeholder((m,), name='X')
shift_reg = tvm.compute(C.shape, lambda i: X[m-1-i] ,name='shift_reg')
print (shift_reg)
#R = tvm.compute(X.shape, lambda k: tvm.sum(X[k]*C[m-1-k]), name='R')


s = tvm.create_schedule(shift_reg.op)
#s2 = tvm.create_schedule(R.op)

f = tvm.lower(s, [C, X, R], name="myadd")
fadd = tvm.build(f, target = "llvm")
#n = 1024
array = [53, 0, -91, 0, 313, 500, 313, 0, -91, 0, 53]
c = tvm.nd.array(array)
x = tvm.nd.array(np.random.uniform(size=m).astype(X.dtype))
sh = tvm.nd.array(np.zeros(n, dtype=C.dtype))
fadd(c, x, sh)

