# -*- coding: utf-8 -*-
"""
Created on Mon Mar 26 07:04:15 2018

@author: cody
"""


import matplotlib.animation as animation
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

__global__ void fitness(double *x, double *y, double eps, double a, double b, double c, double d)
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


# parameters of the circle
radius = 13
middle = length/2 + 128
middle_x = middle/side
middle_y = middle%side


s1_freq = numpy.zeros(length).astype(numpy.float64)
s2_freq = numpy.zeros(length).astype(numpy.float64)


# parameters to test
pop     = 50


mu1     = 20
mu2     = 1
eps     = .01

alpha = 3.6581536769866942

b,c,d   = 0,0,1



for i in range(length):
    x = i/side
    y = i%side
    if distance(x,y,middle_x,middle_y) <= radius:
        s1_freq[i] = pop
        s2_freq[i] = 0
    else:
        s1_freq[i] = 0
        s2_freq[i] = pop
    


# record payoff ratio, diffusion ratio, populations, winner
graph = []
graph2 = []
picture = s1_freq/(s1_freq + s2_freq)

result = numpy.reshape(picture, (side, side))

p = plt.imshow(result,interpolation = 'nearest')
plt.gca().invert_yaxis()
plt.set_cmap('Greys')
plt.pause(1)


for loops in range(20):
    picture = numpy.zeros(length)
                
    for count in range(500):    
        fitness(
            drv.InOut(s1_freq), drv.InOut(s2_freq), numpy.float64(eps), numpy.float64(alpha), numpy.float64(b), numpy.float64(c),numpy.float64(d),
            block = (1024,1,1), grid=(length/1024,1))

        dest1 = s1_freq.copy()
        dest2 = s2_freq.copy()        

        diffuse_display(
            drv.Out(picture), drv.Out(dest1), drv.Out(dest2), drv.In(s1_freq), drv.In(s2_freq), numpy.float64(mu1), numpy.float64(mu2),
            numpy.float64(eps),numpy.int64(side), block = (1024,1,1), grid=(length/1024,1))
            
        s1_freq = dest1.copy()
        s2_freq = dest2.copy()
    
    
    print(picture[middle + radius], loops)

    graph.append(picture[ middle - 48 : middle + 48])
    graph2.append(s1_freq[middle - 48 : middle + 48])   
    gridpic = numpy.reshape(picture,(side,side))
    p.set_data(gridpic)
    plt.pause(0.00001)
                
plt.close()




cwd = os.getcwd()
filename = "vickers2D_wavegraph_"+str(alpha)+"_"+str(mu1)+"_"+str(mu2)+".txt"

filepath = cwd + "/" + filename

f = open(filename, 'w+')
for i in range(len(graph)):
    f.write(str(graph[i]) + "\n")

"""
if not os.path.isfile(filepath):
    f = open(filename,'w+')
    f.write("epsilon, alpha, beta, pop, dif1, dif2, difratio, winner, time"  + "\n")
    for i in range(len(data)):
        temp = data[i][0],data[i][1],data[i][2],data[i][3],data[i][4],data[i][5],data[i][6],data[i][7], data[i][8]
        temp2= str(temp)
        text = temp2[1:-1]
        f.write( str(text)[1:len(str(data))-1] + "\n")
"""
f.close()
