# -*- coding: utf-8 -*-
"""
Created on Fri Mar 16 08:47:54 2018

@author: cody

automate simulations to test blob size... testing diffusion 
"""

import pycuda.autoinit
import pycuda.driver as drv
import numpy
import matplotlib.pyplot as plt
from pycuda.compiler import SourceModule
import time as TIME
import csv
import os

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
middle = length/2 + 128
middle_x = middle/side
middle_y = middle%side


#   payoffs
b,c,d = 0,0,1

eps = .01


#   different diffusion ratios
#dif_1 = numpy.linspace(1,20,20)
#dif_2 = numpy.linspace(20,1,20)


dif_1 = [2 ,  4 , 5 , 7 , 8,  9 , 10 , 11, 12, 13, 14, 15]
dif_2 = [19, 17, 16 ,14, 13, 12 , 11 , 10,  9,  8,  7,  6]


#   attempt threshold
attempt_thresh = 15

#   starting interval for alpha's
a_low = .2
a_high = 5

#   testing different initial circles
blobsizes = numpy.linspace(25,30,6)


#   store all the aggregate data
data = [] 


#   starting population for both strategies
pop = 50


try:
    #   iterate through blob sizes
    for blob in blobsizes:          
    
        #   store blob specific data 
        bdata = []

        for i in range(len(dif_1)): #   iterate through the different diffusion/diffusion ratios
            hi = a_high
            lo = a_low
            
            mu1 = dif_1[i]
            mu2 = dif_2[i]
            
            nextmu = False
            
            attempts = 0            
            
            while 1:
                attempts += 1
                alpha = (hi + lo)/2.0
                
                if nextmu:
                    break
                
                
                #    handles strange cases in which convergence would always take place before 60 iterations
                #    might be caused by some 'acceleration' of dominance due to diffusion rates
                if attempts == attempt_thresh: 
                    data.append([eps, alpha, pop, mu1, mu2, mu1/mu2, blob, 3, time])
                    bdata.append([eps, alpha, pop, mu1, mu2, mu1/mu2, blob, 3, time])
                    break
                
                s1_freq = numpy.zeros(length).astype(numpy.float64)
                s2_freq = numpy.zeros(length).astype(numpy.float64)
                
                #   create the initial circle
                for i in range(length):
                    x = i/side
                    y = i%side
                    if distance(x,y,middle_x, middle_y) <= blob:
                        s1_freq[i] = pop
                        s2_freq[i]
                    else:
                        s1_freq[i] = 0
                        s2_freq[i] = pop

                picture = s1_freq/(s1_freq + s2_freq)                
                time = 0
                sync_inc = 0
                sync_dec = 0
                while 1:
                    
                    start = TIME.time()
                    
                    before_a = picture[middle + 7]
                    before_b = picture[middle + 128]
                    for count in range(1000):
                        fitness(drv.InOut(s1_freq), drv.InOut(s2_freq), numpy.float64(eps), numpy.float64(alpha),
                            numpy.float64(b), numpy.float64(c), numpy.float64(d),
                            block = (1024,1,1), grid = (length/1024,1))
                            
                        dest1 = s1_freq.copy()
                        dest2 = s2_freq.copy()
                        
                        diffuse_display(
                            drv.Out(picture), drv.Out(dest1), drv.Out(dest2), drv.In(s1_freq), drv.In(s2_freq),
                            numpy.float64(mu1),numpy.float64(mu2), numpy.float64(eps),numpy.int64(side),
                            block = (1024,1,1), grid=(length/1024,1))
            
                        s1_freq = dest1.copy()
                        s2_freq = dest2.copy()
                    
                    time += 1
                    end = TIME.time()
                    
                    #   a check to ensure that everything runs smoothly
                    
                    
                    
                    after_a = picture[middle + 7]
                    after_b = picture[middle + 128]
                    print(after_a - before_a, after_b - before_b, blob, mu1, attempts, time)
                    
                    if after_a >= before_a and after_b >= before_b:
                        sync_inc += 1
                        sync_dec = 0
                    elif after_a <= before_a and after_b <= before_b:
                        sync_dec += 1
                        sync_inc = 0
                    elif after_a > before_a and after_b < before_b:
                        sync_inc = 0
                        sync_dec = 0
                    else:
                        sync_inc = 0
                        sync_dec = 0
                    
                    
                    if sync_inc == 25 or after_b > 0.5:
                        data.append([eps, alpha, pop, mu1, mu2, mu1/mu2, blob, 1, time])
                        hi = alpha
                        break
                    
                    if sync_dec == 25 or picture[middle] < 0.5:
                        data.append([eps, alpha, pop, mu1, mu2, mu1/mu2, blob, 2, time])
                        lo = alpha
                        break
                        
                        
                    
                    

        #   saves a text file specific to the blob size
    
#        cwd = os.getcwd()
#        filename = "vickers2D_blob"+str(blob)+"_3.txt"
#
#        filepath = cwd + "/" + filename
#
#
#        if not os.path.isfile(filepath):
#            f = open(filename,'w+')
#            f.write("epsilon, alpha, beta, pop, dif1, dif2, difratio, blobsize, winner, time, converged"  + "\n")
#            for i in range(len(bdata)):
#                temp = bdata[i][0], bdata[i][1], bdata[i][2], bdata[i][3],bdata[i][4],bdata[i][5],bdata[i][6],bdata[i][7], bdata[i][8], bdata[i][9],bdata[i][10]
#                temp2= str(temp)
#                text = temp2[1:-1]
#                f.write( str(text)[1:len(str(bdata))-1] + "\n")
#        else:
#            f = open(filename, 'a+')
#            for i in range(len(bdata)):
#                temp = bdata[i][0], bdata[i][1], bdata[i][2], bdata[i][3],bdata[i][4],bdata[i][5],bdata[i][6],bdata[i][7], bdata[i][8], bdata[i][9],bdata[i][10]
#                temp2= str(temp)
#                text = temp2[1:-1]
#                f.write( str(text)[1:len(str(bdata))-1] + "\n")
#
#        f.close()








#   saves a text file with parameters
finally:
    cwd = os.getcwd()
    filename = "vickers2D_blob1-3.txt"

    filepath = cwd + "/" + filename


    if not os.path.isfile(filepath):
        f = open(filename,'w+')
        f.write("epsilon, alpha, pop, dif1, dif2, difratio, blobsize, winner, time"  + "\n")
        for i in range(len(data)):
            temp = data[i][0],data[i][1],data[i][2],data[i][3],data[i][4],data[i][5],data[i][6],data[i][7], data[i][8]
            temp2= str(temp)
            text = temp2[1:-1]
            f.write( str(text)[1:len(str(data))-1] + "\n")
    else:
        f = open(filename, 'a+')
        for i in range(len(data)):
            temp = data[i][0],data[i][1],data[i][2],data[i][3],data[i][4],data[i][5],data[i][6],data[i][7],data[i][8]
            temp2= str(temp)
            text = temp2[1:-1]
            f.write( str(text)[1:len(str(data))-1] + "\n")

    f.close()
    




