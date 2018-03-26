# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pycuda.autoinit
import pycuda.driver as drv
import numpy
import time
from pycuda.compiler import SourceModule

start = time.time()

n = 400

mod = SourceModule("""
__global__ void multiply_them(float *dest, float *a)
{
  const int i = threadIdx.x;
  dest[i] = .9*a[i] + .05*a[(i-1)%400] + .05*a[(i+1)%400];
}
""")

multiply_them = mod.get_function("multiply_them")

a = numpy.random.rand(n).astype(numpy.float32)

dest = numpy.zeros_like(a)
multiply_them(
        drv.Out(dest), drv.In(a),
        block=(n,1,1), grid=(1,1))

end = time.time()
#print dest-a*b
#print a*b
print dest[0:10]
print a[0:10]
print sum(a)
print sum(dest)
print(end - start)