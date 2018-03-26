# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 23:43:42 2018

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

__global__ void fitness(double *x, double *y, double eps, double a, double b, double c, double d, double *picture,
   double *destx, double *desty, double mu1, double mu2,  int side)
{
  const int i = threadIdx.x + blockDim.x * blockIdx.x;
  
  double tempx = x[i];
  double tempy = y[i];
  
  double avgfit = (x[i]*(x[i]*a + y[i]*c) + y[i]*(x[i]*b+y[i]*d))/pow(x[i]+y[i],2);
  
  //double pop = tempx + tempy; 
  //double lambda1 = 5;
  //double lambda2 = .1;
  //avgfit = lambda1 - lambda2*pop;
  
  y[i] = tempy + tempy*eps*(   (tempx*c + tempy*d)/(tempx+tempy)  -  avgfit );
  x[i] = tempx + tempx*eps*(   (tempx*a + tempy*b)/(tempx+tempy)  -  avgfit );

  __syncthreads();
  
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
  
  double denom = 4+4/sqrt(2.0);
      
  
  destx[i] = (1-mu1eps)*x[i] + (( mu1eps ) / denom ) * (x[N]+x[S]+x[W]+x[E]) + ( (1/sqrt(2.0) ) * mu1eps / denom )*(x[NE]+x[NW]+x[SE]+x[SW]);
  desty[i] = (1-mu2eps)*y[i] + (( mu2eps ) / denom ) * (y[N]+y[S]+y[W]+y[E]) + ( (1/sqrt(2.0) ) * mu2eps / denom )*(y[NE]+y[NW]+y[SE]+y[SW]);

  
  picture[i] = destx[i]/(destx[i]+desty[i]);
}
""")


fitness = mod.get_function("fitness")


side = 256
length = side*side

a   = 1.3
mu1 = 12
mu2 = 10
eps = .01

#initial population values
init_pop1 = 50
init_pop2 = 50

b,c,d = 0,0,1

#parameters of the circle
radius = 15
middle = length/2 + 128
middle_x = middle/side
middle_y = middle%side




s1_freq = numpy.zeros(length).astype(numpy.float64)
s2_freq = numpy.zeros(length).astype(numpy.float64)


#creates the inital circle
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


#   destx and desty
destx = numpy.zeros(length).astype(numpy.float64)
desty = numpy.zeros(length).astype(numpy.float64)


result = numpy.reshape(picture,(side,side))

dest1 = numpy.zeros(length)
dest2 = numpy.zeros(length)

p = plt.imshow(result,interpolation = 'nearest')
plt.gca().invert_yaxis()
plt.set_cmap('Greys')
plt.pause(2)

time = 0
while 1:
    picture = numpy.zeros(length)
    time += 1
    for count in range(1000):
        
        fitness(
            drv.InOut(s1_freq), drv.InOut(s2_freq), numpy.float64(eps), numpy.float64(a), numpy.float64(b), numpy.float64(c),numpy.float64(d),drv.Out(picture), drv.Out(dest1),
            drv.Out(dest2), numpy.float64(mu1), numpy.float64(mu2),numpy.int64(side),
            block = (1024,1,1), grid=(length/1024,1))

            
        s1_freq = dest1.copy()
        s2_freq = dest2.copy()

    print(time)    
    
    gridpic = numpy.reshape(picture,(side,side))
    p.set_data(gridpic)
    plt.pause(0.00001)
