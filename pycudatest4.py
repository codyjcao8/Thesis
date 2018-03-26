# -*- coding: utf-8 -*-
"""
Created on Sun Dec  3 16:28:53 2017

@author: cody
"""

import pycuda.autoinit
import pycuda.driver as drv
import numpy
import time
import matplotlib.pyplot as plt
from pycuda.compiler import SourceModule

start = time.time()

numpy.random.seed(134)

length = 400

"""
- send code to Prof Johnson
- work on 2D
    - send to GPU as 1D
    - update using modulo
    - reformat on host
"""

G = [[2,0],[0,1]]


mod = SourceModule("""
__global__ void fitness(double *dest, double *x)
{
  const int i = threadIdx.x;
  const int a = 2;
  const int b = 0;
  const int c = 0;
  const int d = 1;
  dest[i] = x[i] + 0.5*x[i]*(((a-b)*x[i] + b) - ( (a+d-c-b)*x[i]*x[i] + (c+b-2*d)*x[i] + d));
}

__global__ void diffuse(double *dest, double *a)
{
  const int i = threadIdx.x;
  dest[i] = .9*a[i] + .05*a[(i-1)%400] + .05*a[(i+1)%400];
}
""")


fitness = mod.get_function("fitness")
diffuse = mod.get_function("diffuse")


dest = numpy.random.rand(length).astype(numpy.float64)


#half and half initialization
for i in range(length/2):
    dest[i] = 0
    dest[length-1-i] = 1



plt.plot(dest,'k')


while 1:
    for r in range(10000):    
        #calculating fitness
        fit = numpy.zeros_like(dest)
        fitness(
            drv.Out(fit), drv.In(dest),
            block=(length,1,1), grid=(1,1))
        
        result = numpy.zeros_like(fit)
        
        #diffusion step
        diffuse(
            drv.Out(result), drv.In(fit),
            block=(length,1,1), grid=(1,1))    
    
    plt.clf()
    dest = result.copy()
    axx= plt.subplot(1,1,1)    
    axx.set_ylim([0,1.5])
    axx.set_xlim([0,400])
    plt.plot(result,'r',linewidth=3)
    print(dest[100:110])    
    plt.pause(.000001)
    

end = time.time()  

#print dest[0:10]
#print a[0:10]