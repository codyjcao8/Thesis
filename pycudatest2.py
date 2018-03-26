# -*- coding: utf-8 -*-
"""
Created on Fri Dec  1 01:32:28 2017

@author: cody
"""

import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy
import time

start = time.time()

a = numpy.random.rand(4,4)
a = a.astype(numpy.float32)

a_gpu = cuda.mem_alloc(a.nbytes)

cuda.memcpy_htod(a_gpu,a)

mod = SourceModule("""
    __global__ void doublify(float *a)
    {
        int idx = threadIdx.x + threadIdx.y*4;
        a[idx] *= 2;
    }
""")

func = mod.get_function("doublify")
func(a_gpu, block=(4,4,1))


a_doubled = numpy.empty_like(a)
cuda.memcpy_dtoh(a_doubled, a_gpu)

end = time.time()

print a_doubled
print a
print end - start