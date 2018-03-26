# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 12:52:06 2018

@author: cody
"""

import pycuda.autoinit
import pycuda.driver as drv
import numpy
import matplotlib.pyplot as plt
from pycuda.compiler import SourceModule
import time


numpy.random.seed(1200057)
#numpy.random.seed(12457)

def test(x):
    return x**(1/.61)


def distance(x1,y1,x2,y2):
    temp = (x1-x2)**2 + (y1-y2)**2
    return temp**0.5


mod = SourceModule("""
#include <math.h>

__global__ void fitness(double *x, double *y, double eps, double a, double b, double c, double d)
{
  const int i = threadIdx.x + blockDim.x * blockIdx.x;
  
  double tempx = x[i];
  double tempy = y[i]; 
  
  double pop = tempx + tempy; 
  double lambda1 = 2;
  double lambda2 = .02;
  double avgfit = lambda1 - lambda2*pop;
  
  y[i] = tempy + tempy*eps*(   (tempx*c + tempy*d) / pop  +  avgfit );
  x[i] = tempx + tempx*eps*(   (tempx*a + tempy*b) / pop  +  avgfit );

}


__global__ void diffuse_display(double *picture, double *destx, double *desty, double *x, double *y, double mu1, double mu2, double eps, int side)
{
  const int i = threadIdx.x + blockDim.x * blockIdx.x;
  const int ROW    = i/side;
  const int COLUMN = i%side;
  
  const int N   =   ROW*side               + (COLUMN+side-1)%side;
  const int S   =   ROW*side               + (COLUMN +1)%side;
  const int E   = ((ROW+1)%side)*side      +  COLUMN;
  const int W   = ((ROW+side-1)%side)*side +  COLUMN;
  const int NE  = ((ROW+1)%side)*side      + (COLUMN+side-1)%side;
  const int NW  = ((ROW-1+side)%side)*side + (COLUMN+side-1)%side;
  const int SE  = ((ROW+1)%side)*side      + (COLUMN+1)%side;
  const int SW  = ((ROW+side-1)%side)*side + (COLUMN+1)%side;
  
  double mu1eps = mu1*eps;
  double mu2eps = mu2*eps;
  
  double d = 4+4/sqrt(2.0);   
  
  destx[i] = (1-mu1eps)*x[i] + (( mu1eps ) / d ) * (x[N]+x[S]+x[W]+x[E]) + ( (1/sqrt(2.0) ) * mu1eps / d )*(x[NE]+x[NW]+x[SE]+x[SW]);
  desty[i] = (1-mu2eps)*y[i] + (( mu2eps ) / d ) * (y[N]+y[S]+y[W]+y[E]) + ( (1/sqrt(2.0) ) * mu2eps / d )*(y[NE]+y[NW]+y[SE]+y[SW]);
  
  picture[i] = destx[i]/(destx[i]+desty[i]);
}
""")


fitness = mod.get_function("fitness")
diffuse_display = mod.get_function("diffuse_display")


side = 256
length = side*side

a   = 1.7375
mu1 = 1
mu2 = 20
eps = .01

#initial population values
init_pop1 = (2+a)/.02
init_pop2 = 100

b,c,d = 0,0,1


#   parameters of the circle
radius = 22
middle = length/2 + 128
middle_x = middle/side
middle_y = middle%side


s1_freq = numpy.zeros(length).astype(numpy.float64)
s2_freq = numpy.zeros(length).astype(numpy.float64)


#   creates the inital circle
for i in range(length):
    x = i/side
    y = i%side
    if distance(x,y,middle_x,middle_y) <= radius:
        s1_freq[i] = init_pop1
        s2_freq[i] = 0
    else:
        s1_freq[i] = 0
        s2_freq[i] = init_pop2


#   picture
picture = numpy.zeros(length)
for i in range(length):
    picture[i] = s1_freq[i]/(s1_freq[i]+s2_freq[i])


result = numpy.reshape(picture,(side,side))

graph = []

p = plt.imshow(result,interpolation = 'nearest')
plt.gca().invert_yaxis()
plt.set_cmap('Greys')
plt.pause(2)

time = 0

try:
    while 1:

        time += 1
        for count in range(1000):    
            fitness(
                drv.InOut(s1_freq), drv.InOut(s2_freq), numpy.float64(eps), numpy.float64(a), numpy.float64(b), numpy.float64(c),numpy.float64(d),
                block = (1024,1,1), grid=(length/1024,1))
    
            dest1 = s1_freq.copy()
            dest2 = s2_freq.copy()        
            
            diffuse_display(
                drv.Out(picture), drv.Out(dest1), drv.Out(dest2), drv.In(s1_freq), drv.In(s2_freq), numpy.float64(mu1), numpy.float64(mu2), numpy.float64(eps),
                numpy.int64(side), block = (1024,1,1), grid=(length/1024,1))
                
            s1_freq = dest1.copy()
            s2_freq = dest2.copy()
            
        graph.append(sum(picture[middle-128:middle+128]))
        
        print(time  , sum(picture))
        gridpic = numpy.reshape(picture,(side,side))
        p.set_data(gridpic)
        plt.pause(0.00001)
        
except KeyboardInterrupt:
    pass



plt.figure()
xx = range(len(graph))
plt.plot(xx,graph)