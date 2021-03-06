# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 19:01:33 2018

Runs simulations of replicator-diffusion to test for intervals of non-convergence

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

__global__ void fitness(double *x, double *y, double eps, double a, double b, double c, double d)
{
  const int i = threadIdx.x + blockDim.x * blockIdx.x;
  
  double tempx = x[i];
  double tempy = y[i];
  
  //double avgfit = (x[i]*(x[i]*a + y[i]*c) + y[i]*(x[i]*b+y[i]*d))/pow(x[i]+y[i],2);
  
  double pop = tempx + tempy; 
  double lambda1 = 2;
  double lambda2 = .02;
  double avgfit = lambda1 - lambda2*pop;
  
  y[i] = tempy + tempy*eps*(   (tempx*c + tempy*d)/(tempx+tempy)  +  avgfit );
  x[i] = tempx + tempx*eps*(   (tempx*a + tempy*b)/(tempx+tempy)  +  avgfit );

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
radius = 15
middle = length/2 + 128
middle_x = middle/side
middle_y = middle%side


s1_freq = numpy.zeros(length).astype(numpy.float64)
s2_freq = numpy.zeros(length).astype(numpy.float64)


# parameters to test
mulist  = numpy.linspace(5, 15, 11)



init_pop2 = 100
mu2       = 10.0
eps       = .01


b,c,d   = 0,0,1


# record payoff ratio, diffusion ratio, populations, winner
data = []


for mu in mulist:
    hi = 1.9
    lo = .5
    nextmu = False    
    
    while 1:
        alpha = (hi + lo)/2
        
        pop1 = (alpha + 2)/.02        
        
        if nextmu:
            break
        
        s1_freq = numpy.zeros(length).astype(numpy.float64)
        s2_freq = numpy.zeros(length).astype(numpy.float64)
        
         
        
        
    #   creates the initial clump
        for i in range(length):
            x = i/side
            y = i%side
            if distance(x,y,middle_x,middle_y) <= radius:
                s1_freq[i] = pop1
                s2_freq[i] = 0
            else:
                s1_freq[i] = 0
                s2_freq[i] = init_pop2
                #   picture
        picture = numpy.zeros(length)
    
        for i in range(length):
            picture[i] = s1_freq[i]/(s1_freq[i]+s2_freq[i])
        
        result = numpy.reshape(picture,(side,side))
        p = plt.imshow(result,interpolation = 'nearest')
        plt.gca().invert_yaxis()
        plt.set_cmap('Greys')
        plt.pause(2)
        
        time = 0
        
        while 1:
            time += 1
            picture = numpy.zeros(length)
                
            for count in range(1000):    
                fitness(
                        drv.InOut(s1_freq), drv.InOut(s2_freq), numpy.float64(eps), numpy.float64(alpha), numpy.float64(b), numpy.float64(c),numpy.float64(d),
                        block = (1024,1,1), grid=(length/1024,1))

                dest1 = s1_freq.copy()
                dest2 = s2_freq.copy()        

                diffuse_display(
                        drv.Out(picture), drv.Out(dest1), drv.Out(dest2), drv.In(s1_freq), drv.In(s2_freq), numpy.float64(mu), numpy.float64(mu2),
                        numpy.float64(eps),numpy.int64(side),
                        block = (1024,1,1), grid=(length/1024,1))
            
                s1_freq = dest1.copy()
                s2_freq = dest2.copy()
    
            print(time  , picture[32935], picture[middle+5] )
            if picture[middle + 5] < 0.5:
                #   record that strategy 2 won
                data.append([eps, alpha, d, pop1, init_pop2, mu, mu2,mu/mu2, 2, time])
                lo = alpha
                break
                
            if picture[32935] > 0.5:
                #   record that strategy 1 won
                data.append([eps, alpha, d, pop1, init_pop2, mu, mu2, mu/mu2, 1, time])
                hi = alpha
                break
            
            if time == 100:
                #   something                  
                data.append([eps, alpha, d, pop1, init_pop2, mu, mu2, mu/mu2, 3, time])
                nextmu = True
                break 
            
            gridpic = numpy.reshape(picture,(side,side))
            p.set_data(gridpic)
            plt.pause(0.00001)
                
        plt.close()




cwd = os.getcwd()
filename = "cressman2D.txt"

filepath = cwd + "/" + filename


if not os.path.isfile(filepath):
    f = open(filename,'w+')
    f.write("epsilon, alpha, beta, initpop1, initpop2, dif1, dif2, difratio, winner, time"  + "\n")
    for i in range(len(data)):
        temp = data[i][0],data[i][1],data[i][2],data[i][3],data[i][4],data[i][5],data[i][6],data[i][7], data[i][8], data[i][9]
        temp2= str(temp)
        text = temp2[1:-1]
        f.write( str(text)[1:len(str(data))-1] + "\n")
else:
    f = open(filename, 'a+')
    for i in range(len(data)):
        temp = data[i][0],data[i][1],data[i][2],data[i][3],data[i][4],data[i][5],data[i][6],data[i][7], data[i][8], data[i][9]
        temp2= str(temp)
        text = temp2[1:-1]
        f.write( str(text)[1:len(str(data))-1] + "\n")

f.close()