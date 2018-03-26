# -*- coding: utf-8 -*-
"""
Created on Sun Mar  4 10:04:50 2018

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


mod = SourceModule("""
#include <math.h>

__global__ void fitness(double *s1, double *s2, double eps, double a, double b, double c, double d)
{
  const int i = threadIdx.x + blockDim.x * blockIdx.x;
  
  double tempx = s1[i];
  double tempy = s2[i];
  
  double avgfit = (s1[i]*(s1[i]*a + s2[i]*c) + s2[i]*(s1[i]*b+s2[i]*d))/pow(s1[i]+s2[i],2);
   
  
  s2[i] = tempy + tempy*eps*(   (tempx*c + tempy*d)/(tempx+tempy)  - avgfit );
  s1[i] = tempx + tempx*eps*(   (tempx*a + tempy*b)/(tempx+tempy)  - avgfit );

}


__global__ void diffuse_display(double *picture,double *x1, double *x2, double *s1, double *s2, double mu1, double mu2, double eps, int length)
{
  const int i = threadIdx.x + blockDim.x * blockIdx.x;
  
  double mu1eps = mu1 * eps;
  double mu2eps = mu2 * eps;
  
  if (i != 0 and i != length-1){
      x1[i] = 0.5*mu1eps*(s1[i+1]+s1[i-1]) + (1-mu1eps)*s1[i];
      x2[i] = 0.5*mu2eps*(s2[i+1]+s2[i-1]) + (1-mu2eps)*s2[i];
  }
  else if (i == 0){
     x1[i] = mu1eps*s1[i+1] + (1-mu1eps)*s1[i];
     x2[i] = mu2eps*s2[i+1] + (1-mu2eps)*s2[i];
  }
  else{
    x1[i] = mu1eps*s1[i-1] + (1-mu1eps)*s1[i];
    x2[i] = mu2eps*s2[i-1] + (1-mu2eps)*s2[i];
  }
  
  picture[i] = x1[i]/(x1[i] + x2[i]);
}
""")


fitness = mod.get_function("fitness")
diffuse_display = mod.get_function("diffuse_display")


#   parameters

side = 50
length = side*side
center = length/2

mu1list     = [20]
mu2list     = [1]



eps     = .01

maxiterations = 25

b,c,d   = 0,0,1

block   = 500
grid    = 5


count   = 0


pop = 50


data    = []

try:
    for index in range(len(mu1list)):
        nextmu = False
        hi     = 8
        lo     = .2
        mu1    = mu1list[index]
        mu2    = mu2list[index]
        iterations = 0
        while 1:
            a = (hi+lo)/2
            
              
            if nextmu:
                break
            
            
            if iterations == maxiterations:
                data.append([eps, a, pop, mu1, mu2, mu1/mu2, 3, time])
                break
            
            s1_freq = numpy.zeros(length).astype(numpy.float64)
            s2_freq = numpy.zeros(length).astype(numpy.float64)
    
            for i in range(length/2):
                s1_freq[i]          = pop
                s2_freq[i+length/2] = pop
            
            picture = s1_freq/(s1_freq + s2_freq)
            
            time = 0
            
    
    
            while 1:
                time += 1
                picture = numpy.zeros(length)
                for count in range(1000):
                    
                    fitness(
                        drv.InOut(s1_freq), drv.InOut(s2_freq), numpy.float64(eps), numpy.float64(a),
                        numpy.float64(b),numpy.float64(c),numpy.float64(d),
                        block=(block,1,1), grid = (grid,1))
            
                    x1 = s1_freq.copy()
                    x2 = s2_freq.copy()
            
    
                    diffuse_display(
                        drv.Out(picture), drv.Out(x1),drv.Out(x2), drv.In(s1_freq),drv.In(s2_freq), numpy.float64(mu1), numpy.float64(mu2),numpy.float64(eps),
                        numpy.int64(length), block = (block,1,1), grid = (grid,1))
            
                    s1_freq = x1.copy()
                    s2_freq = x2.copy()
                    
                print(time, picture[center-75], picture[center+75], mu1, mu2)
                
                
                
                
                if picture[center - 75] < 0.5:
                    #   record that strategy 2 won
                    data.append([eps, a, pop, mu1, mu2, mu1/mu2, 2, time])
                    lo = a
                    iterations += 1
                    break
                    
                if picture[center + 75] > 0.5:
                    #   record that strategy 1 won
                    data.append([eps, a, pop, mu1, mu2, mu1/mu2, 1, time])
                    hi = a
                    iterations += 1                
                    break
                
                if time == 300 and hi - lo >= .18:
                                                                #   narrow the interval in which the wave doesn't move (moves too slowly)
                    
                    nextmu = True                               #   so that we move on to the next set of diffusion rates
                    fiveALPHA = numpy.linspace(lo,hi,num=5)     #   break interval into five evenly spaced values
                    
                    for newA in fiveALPHA:                      #   loop through each of the new alpha values
    
                                                                #   initializing both sides of the wave
                        
                        s1_freq = numpy.zeros(length).astype(numpy.float64)
                        s2_freq = numpy.zeros(length).astype(numpy.float64)
    
                        for i in range(length/2):
                            s1_freq[i]          = pop
                            s2_freq[i+length/2] = pop
            
                        newtime = 0
                        
                        while 1:
                                                                #   main subloop 
                            newtime += 1
                            newpicture = numpy.zeros(length)
                            for count in range(1000):
                    
                                fitness(
                                    drv.InOut(s1_freq), drv.InOut(s2_freq), numpy.float64(eps), numpy.float64(newA),numpy.float64(b),
                                    numpy.float64(c),numpy.float64(d), block=(block,1,1), grid = (grid,1))
            
                                x1 = s1_freq.copy()
                                x2 = s2_freq.copy()
            
    
                                diffuse_display(
                                    drv.Out(newpicture), drv.Out(x1),drv.Out(x2), drv.In(s1_freq),drv.In(s2_freq), numpy.float64(mu1),
                                    numpy.float64(mu2),numpy.float64(eps), numpy.int64(length), block = (block,1,1), grid = (grid,1))
                        
                                s1_freq = x1.copy()
                                s2_freq = x2.copy()
                    
                            print(newtime, newpicture[center-75],newpicture[center + 75], "new subloop")
                
                
                            if newpicture[center - 75] < 0.5:
                                #   record that strategy 2 won
                                data.append([eps, newA, pop, mu1, mu2, mu1/mu2, 2, newtime])
                                break
                                
                            if newpicture[center + 75] > 0.5:
                                #   record that strategy 1 won
                                data.append([eps, newA, pop, mu1, mu2, mu1/mu2, 1, newtime])
                                break
                            
                            if newtime == 300:
                                #   record slow convergence __ wave is close to not moving
                                data.append([eps, newA, pop, mu1, mu2, mu1/mu2, 3, newtime])
                                break
                            
                elif time == 300 and hi - lo < .18:
                    data.append([eps, a, pop, mu1, mu2, mu1/mu2, 3, time])
                    nextmu = True
                    break




finally:
    cwd = os.getcwd()
    filename = "vickers1Ddif20.txt"
    
    filepath = cwd + "/ " + filename
    
    
    if not os.path.isfile(filepath):
        f = open(filename,'w+')
        f.write("epsilon, alpha, pop, dif1, dif2, difratio, winner, time"  + "\n")
        for i in range(len(data)):
            temp = data[i][0],data[i][1],data[i][2],data[i][3],data[i][4],data[i][5],data[i][6],data[i][7]
            temp2= str(temp)
            text = temp2[1:-1]
            f.write( str(text)[1:len(str(data))-1] + "\n")
    else:
        f = open(filename, 'a+')
        for i in range(len(data)):
            temp = data[i][0],data[i][1],data[i][2],data[i][3],data[i][4],data[i][5],data[i][6],data[i][7]
            temp2= str(temp)
            text = temp2[1:-1]
            f.write( str(text)[1:len(str(data))-1] + "\n")
    
    f.close()
