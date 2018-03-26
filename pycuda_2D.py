# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 22:11:59 2017

@author: cody
"""

import pycuda.autoinit
import pycuda.driver as drv
import numpy
import matplotlib.pyplot as plt
from pycuda.compiler import SourceModule


numpy.random.seed(1574657)


"""
work on implementing different diffusion rates from cressman, vickers

"""


def makeSquare(array, sizeOfSquare, offset,dimension):
    n = dimension*dimension/2 + offset
    for i in range(n,n+sizeOfSquare):
        for k in range(sizeOfSquare):
            array[i+k*dimension] = 1

mod = SourceModule("""
__global__ void fitness(double *dest, double *x, double eps)
{
  const int i = threadIdx.x + blockDim.x * blockIdx.x;
  const float a = 1.05;
  const float b = 0;
  const float c = 0;
  const float d = 1;
  dest[i] = x[i] + eps*x[i]*(((a-b)*x[i] + b) - ( (a+d-c-b)*x[i]*x[i] + (c+b-2*d)*x[i] + d));
}

__global__ void diffuse_neumann(double *dest, double *a, int side)
{
  const int i = threadIdx.x + blockDim.x * blockIdx.x;
  //if(i>=length) return;
  const int ROW = i/side;
  const int COLUMN = i%side;
  const int above =  ROW*side      + (COLUMN+side-1)%side;
  const int below =  ROW*side      + (COLUMN     +1)%side;
  const int right =  ((ROW     +1)%side)*side +  COLUMN;
  const int left =   ((ROW+side-1)%side)*side +  COLUMN;
  dest[i] = .9*a[i] + .025*a[above]+ .025*a[below]+ .025*a[left]+ .025*a[right];
}

__global__ void diffuse_moore(double *dest, double *a, int side, double eps, double mu)
{
  const int i = threadIdx.x + blockDim.x * blockIdx.x;
  //if(i>=length) return;

  const int ROW = i/side;
  const int COLUMN = i%side;
  const int N =   ROW*side               + (COLUMN+side-1)%side;
  const int S =   ROW*side               + (COLUMN +1)%side;
  const int E = ((ROW+1)%side)*side      +  COLUMN;
  const int W = ((ROW+side-1)%side)*side +  COLUMN;
  const int NE= ((ROW+1)%side)*side      + (COLUMN+side-1)%side;
  const int NW= ((ROW-1+side)%side)*side + (COLUMN+side-1)%side;
  const int SE= ((ROW+1)%side)*side      + (COLUMN+1)%side;
  const int SW= ((ROW+side-1)%side)*side + (COLUMN+1)%side;
  const double d = 4+4/sqrt(2.0);
  double mueps = mu*eps;
  
  dest[i] = (1-mueps)*a[i] + ((mueps/d)*(a[N]+a[S]+a[W]+a[E]) + ((1/sqrt(2.0))*mueps/d)*(a[NE] + a[NW] + a[SE] + a[SW]));  
}
""")


fitness = mod.get_function("fitness")
diffuse_neumann = mod.get_function("diffuse_neumann")
diffuse_moore = mod.get_function("diffuse_moore")


side = 256
length = side*side


mu = .3
eps = .1


dest = numpy.random.rand(length).astype(numpy.float64)
#dest = numpy.zeros(length,dtype = numpy.float64)


#for i in range(len(dest)/2):
#    dest[i] = 1.05/2.05
#    dest[len(dest)-i-1] = 1.05/2.05


#makeSquare(dest, 100, 100, side)


result = numpy.reshape(dest,(side,side))


p = plt.imshow(result,interpolation = 'nearest')
plt.gca().invert_yaxis()
plt.set_cmap('Greys')
plt.pause(2)


while 1:
    for count in range(50):
        temp = dest.copy()
        fitness(
            drv.Out(dest), drv.In(temp), numpy.float64(eps),
            block = (1024,1,1), grid=(64,1))
        
        temp = dest.copy()
        diffuse_moore(
            drv.Out(dest), drv.In(temp), numpy.int64(side), numpy.float64(eps), numpy.float64(mu),
            block = (1024,1,1), grid=(64,1))
        
    picture = numpy.reshape(dest,(side,side))
    p.set_data(picture)
    plt.pause(0.00001)