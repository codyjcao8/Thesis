# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 15:34:11 2018

@author: cody
"""


import pycuda.autoinit
import pycuda.driver as drv
import numpy
import matplotlib.pyplot as plt
from pycuda.compiler import SourceModule
import time
import csv
import os

numpy.random.seed(1200057)

def test(x):
    return x**(1/.61)


def makeSquare(array, sizeOfSquare, offset,dimension):
    n = dimension*dimension/2 + offset
    for i in range(n,n+sizeOfSquare):
        for k in range(sizeOfSquare):
            array[i+k*dimension] = 1


def makeSquare(array, sizeOfSquare, offset,dimension,players):
    n = dimension*dimension/2 + offset
    for i in range(n,n+sizeOfSquare):
        for k in range(sizeOfSquare):
            array[i+k*dimension] = players

def distance(x1,y1,x2,y2):
    temp = (x1-x2)**2 + (y1-y2)**2
    return temp**0.5


mod = SourceModule("""
#include <math.h>

__global__ void fitness(double *s1, double *s2, double eps, double a, double b, double c, double d)
{
  const int i = threadIdx.x + blockDim.x * blockIdx.x;
  
  double tempx = s1[i];
  double tempy = s2[i];
  
  double avgfit = (s1[i]*(s1[i]*a + s2[i]*c) + s2[i]*(s1[i]*b+s2[i]*d))/pow(s1[i]+s2[i],2);
  
  //double pop = tempx + tempy; 
  //double lambda1 = 5;
  //double lambda2 = .1;
  //avgfit = lambda1 - lambda2*pop;
  
  s2[i] = tempy + tempy*eps*(   (tempx*c + tempy*d)/(tempx+tempy)  -  avgfit );
  s1[i] = tempx + tempx*eps*(   (tempx*a + tempy*b)/(tempx+tempy)  -  avgfit );

}


__global__ void diffuse_display(double *picture,double *x1, double *x2, double *s1, double *s2, double mu1, double mu2, double eps, int length)
{
  const int i = threadIdx.x + blockDim.x * blockIdx.x;
  
  double mu1eps = mu1 * eps;
  double mu2eps = mu2 * eps;
  
  if (i == 0){
     x1[i] = mu1eps*s1[i+1] + (1-mu1eps)*s1[i];
     x2[i] = mu2eps*s2[i+1] + (1-mu2eps)*s2[i];
  }
  else if (i == length-1){
     x1[i] = mu1eps*s1[i-1] + (1-mu1eps)*s1[i];
     x2[i] = mu2eps*s2[i-1] + (1-mu2eps)*s2[i];
  }
  else{
     x1[i] = 0.5*mu1eps*(s1[i+1]+s1[i-1]) + (1-mu1eps)*s1[i];
     x2[i] = 0.5*mu2eps*(s2[i+1]+s2[i-1]) + (1-mu2eps)*s2[i];
  }
  
  picture[i] = x1[i]/(x1[i] + x2[i]);
}
""")


fitness = mod.get_function("fitness")
diffuse_display = mod.get_function("diffuse_display")


#parameters

side = 50
length = side*side
center = length/2


pop     = 10

mu1     = 12
mu2     = 9

eps     = .01

a       = 1.2
b,c,d   = 0,0,1

block   = 500
grid    = 5

#grid * block = length

s1_freq = numpy.zeros(length).astype(numpy.float64)
s2_freq = numpy.zeros(length).astype(numpy.float64)

for i in range(length/2):
    s1_freq[i] = pop
    s2_freq[i+length/2] = pop

"""
# record payoff ratio, diffusion ratio, populations, winner
data = []
"""


T = s1_freq/(s1_freq + s2_freq)
T = numpy.expand_dims(T,axis = 0)
initDisplay = T.copy()
initDisplay = numpy.tile(initDisplay,(400,1))
p = plt.imshow(initDisplay,interpolation = 'nearest')
plt.set_cmap('Greys')
plt.pause(2)

while 1:
    
    for count in range(1000):    
    
        picture = numpy.zeros(length)
    
       
        
        fitness(
            drv.InOut(s1_freq), drv.InOut(s2_freq), numpy.float64(eps), numpy.float64(a),numpy.float64(b),numpy.float64(c),numpy.float64(d),
            block=(block,1,1), grid = (grid,1)
        )
     
        x1 = s1_freq.copy()
        x2 = s2_freq.copy()
        
        diffuse_display(
            drv.Out(picture), drv.Out(x1), drv.Out(x2), drv.In(s1_freq),drv.In(s2_freq), numpy.float64(mu1), numpy.float64(mu2),numpy.float64(eps),
            numpy.int64(length), block = (block,1,1), grid = (grid,1)
        )
  
        s1_freq = x1.copy()
        s2_freq = x2.copy()
    
    
    print(picture[center-25], picture[center+25])
    T = picture.copy()
    T = numpy.expand_dims(T,axis = 0)
    display = numpy.tile(T,(400,1))
    p.set_data(display)
    plt.pause(0.0001)



