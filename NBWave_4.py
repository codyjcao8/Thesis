# -*- coding: utf-8 -*-
"""
Created on Sat Dec  2 15:41:54 2017

@author: stewjo
"""


import matplotlib.pyplot as plt
 
import pycuda.autoinit
import pycuda.driver as drv
import numpy

from pycuda.compiler import SourceModule


# kernal declaration
mod = SourceModule("""
__global__ void nichbail(float *ax, float *ay, float *bx, float *by)
{
  const int i = threadIdx.x;
  
  // nich bail
  bx[i]= 2.0*ax[i]*exp(-ay[i]);
  by[i]= ax[i]*(1.0-exp(-ay[i]));
  
}

__global__ void migrate(float *ax, float *ay, float *bx, float *by)
{
  const int i = threadIdx.x;
  
  // migrate    
  float mux= .95;
  float muy= .05;
  ax[i]= (1-mux)*bx[i] + (mux/2.0)*(bx[(i-1+300)%300]+bx[(i+1)%300]);
  ay[i]= (1-muy)*by[i] + (muy/2.0)*(by[(i-1+300)%300]+by[(i+1)%300]);
}
""")


nichbail = mod.get_function("nichbail")
migrate = mod.get_function("migrate")


numpy.random.seed(25993)


ax = numpy.zeros(300).astype(numpy.float32)
ay = numpy.zeros(300).astype(numpy.float32)


ax[110:116]= .1*numpy.array( [0.60, 3.95, 17.36, 30.57, 26.78, 15.79], dtype=numpy.float32)
ay[113:118]= .1*numpy.array([0.16, 7.15, 24.94, 12.26, 0.22], dtype=numpy.float32)


ax += .05*numpy.random.rand(300).astype(numpy.float32)
ay += .05*numpy.random.rand(300).astype(numpy.float32)


bx = numpy.zeros_like(ax)
by = numpy.zeros_like(ay)


plt.figure()


while 1:
    nichbail(
            drv.InOut(ax), drv.InOut(ay), drv.InOut(bx), drv.InOut(by),
            block=(300,1,1), grid=(1,1))
    migrate(
            drv.InOut(ax), drv.InOut(ay), drv.InOut(bx), drv.InOut(by),
            block=(300,1,1), grid=(1,1))
    
    #small noise
    ax += .00005*numpy.random.rand(300).astype(numpy.float32)
    ay += .00005*numpy.random.rand(300).astype(numpy.float32)
        
            
    plt.clf()  
    axx= plt.subplot(2,1,1)    
    axx.set_ylim([0,100])
    plt.plot(ax,'k',linewidth=3)
    axy= plt.subplot(2,1,2)          
    axy.set_ylim([0,100])
    plt.plot(ay,linewidth=3)
    plt.show()
    plt.pause(0.000001)
