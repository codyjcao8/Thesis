# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 20:15:12 2018

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
  
  double N = tempx + tempy;
  double fN = 2 - 0.02*N;  
  
  s2[i] = tempy + tempy*eps*(   (tempx*c + tempy*d)/(tempx+tempy)  + fN );
  s1[i] = tempx + tempx*eps*(   (tempx*a + tempy*b)/(tempx+tempy)  + fN );

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


#parameters

side = 50
length = side*side

pop     = 100

mu1list = numpy.linspace(9,11,15)
mu2     = 10

eps     = .01


b,c,d   = 0,0,1

block = 500
grid = 5


s1_freq = numpy.zeros(length).astype(numpy.float64)
s2_freq = numpy.zeros(length).astype(numpy.float64)

for i in range(length/2):
    s1_freq[i] = pop
    s2_freq[i+length/2] = pop


count = 0




pop = 50


for mu1 in mu1list:
    nextmu = False
    hi     = 1.5
    lo     = 1
    
    while 1:
        a = (hi+lo)/2
        
        if nextmu:
            break
        
        s1_freq = numpy.zeros(length).astype(numpy.float64)
        s2_freq = numpy.zeros(length).astype(numpy.float64)

        for i in range(length/2):
            s1_freq[i] = pop
            s2_freq[i+length/2] = pop
        
        picture = s1_freq/(s1_freq + s2_freq)
        
        time = 0
        

        T = s1_freq/(s1_freq + s2_freq)
        T = numpy.expand_dims(T,axis = 0)
        initDisplay = T.copy()
        initDisplay = numpy.tile(initDisplay,(400,1))
        p = plt.imshow(initDisplay,interpolation = 'nearest')
        plt.set_cmap('Greys')
        plt.pause(2)


        while 1:
            time += 1
            for count in range(1000):
                
                fitness(
                    drv.InOut(s1_freq), drv.InOut(s2_freq), numpy.float64(eps), numpy.float64(a),numpy.float64(b),numpy.float64(c),numpy.float64(d),
                    block=(block,1,1), grid = (grid,1))
        
                x1 = s1_freq.copy()
                x2 = s2_freq.copy()
        

                diffuse_display(
                    drv.Out(picture), drv.Out(x1),drv.Out(x2), drv.In(s1_freq),drv.In(s2_freq), numpy.float64(mu1), numpy.float64(mu2),numpy.float64(eps),
                    numpy.int64(length), block = (block,1,1), grid = (grid,1))
        
                s1_freq = x1.copy()
                s2_freq = x2.copy()
                
            print(time, sum(picture))
            
            
            
            
            

for loop in range(500):
    
    for i in range(1000):    
    
        picture = numpy.zeros(length)
    
        
        fitness(
            drv.InOut(s1_freq), drv.InOut(s2_freq), numpy.float64(eps), numpy.float64(a),numpy.float64(b),numpy.float64(c),numpy.float64(d),
            block=(block,1,1), grid = (grid,1)
            )
        
        x1 = s1_freq.copy()
        x2 = s2_freq.copy()
        

        diffuse_display(
            drv.Out(picture), drv.Out(x1),drv.Out(x2), drv.In(s1_freq),drv.In(s2_freq), numpy.float64(mu1), numpy.float64(mu2),numpy.float64(eps),
            numpy.int64(length), block = (block,1,1), grid = (grid,1)
            )
        
        s1_freq = x1.copy()
        s2_freq = x2.copy()



    count += 1

    if count == 5:
        data5 = picture.copy()
        
    if count == 50:
        data50 = picture.copy()
        
    if count == 150:
        data150 = picture.copy()
    
    T = picture.copy()
    T = numpy.expand_dims(T,axis = 0)
    display = numpy.tile(T,(400,1))
    p.set_data(display)
    plt.pause(0.0001)


plt.close()

plt.plot(range(len(data5[500:1500])),data5[500:1500])
plt.plot(range(len(data50[500:1500])),data50[500:1500])
plt.plot(range(len(data150[500:1500])),data150[500:1500])
plt.axis([0,1500,-.1,1.01])
plt.show()
  


"""
cwd = os.getcwd()
filename = "data_test"+str(pop)+"_int.txt"

filepath = cwd + "\\" + filename


if not os.path.isfile(filepath):
    f = open(filename,'w+')
    f.write("epsilon, alpha, beta, initpop1, initpop2, dif1, dif2, winner, time"  + "\n")
    for i in range(len(data)):
        temp = data[i][0],data[i][1],data[i][2],data[i][3],data[i][4],data[i][5],data[i][6],data[i][7], data[i][8]
        temp2= str(temp)
        text = temp2[1:-1]
        f.write( str(text)[1:len(str(data))-1] + "\n")
else:
    f = open(filename, 'a+')
    for i in range(len(data)):
        temp = data[i][0],data[i][1],data[i][2],data[i][3],data[i][4],data[i][5],data[i][6],data[i][7], data[i][8]
        temp2= str(temp)
        text = temp2[1:-1]
        f.write( str(text)[1:len(str(data))-1] + "\n")

f.close()
"""