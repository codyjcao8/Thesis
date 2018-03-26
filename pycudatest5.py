# -*- coding: utf-8 -*-
"""
Created on Fri Dec  8 07:53:30 2017

@author: cody
"""

import pycuda.autoinit
import pycuda.driver as drv
import numpy
import time
import matplotlib.pyplot as plt
from pycuda.compiler import SourceModule


N = 25


mod = SourceModule("""


__global__ void diffuse(double *dest,double *dest2,double *dest3, double *dest4, double *a)
{
  const int i = threadIdx.x;
  const int j = threadIdx.y;
  const int k = blockIdx.x;
  const int l = blockIdx.y;
  dest[i] = i;
  dest2[j] = j;
  dest3[k] = k;
  dest4[l] = l;
}

""")

diffuse = mod.get_function("diffuse")


a = numpy.zeros(N)
TX = numpy.zeros(N)
TY = numpy.zeros(N)
TBX = numpy.zeros(N)
TBY = numpy.zeros(N)

diffuse(
    drv.Out(TX),drv.Out(TY), drv.Out(TBX), drv.Out(TBY), drv.In(a),
    block=(N,N,1),grid=(N,N))


print TX
print "_________________________"
print TY
print "_________________________"
print TBX
print "_________________________"
print TBY
print "_________________________"