# -*- coding: utf-8 -*-
"""
Created on Fri Mar 23 11:54:25 2018

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
middle = length/2 + 128
middle_x = middle/side
middle_y = middle%side


#   payoffs
b,c,d = 0,0,1

eps = .01


#   different diffusion ratios
dif_1 = numpy.linspace(1,20,20)
dif_2 = numpy.linspace(20,1,20)



#   max number of iterations
max_iters = 25

#   starting interval for alpha's
a_low = .5
a_high = 2

#   testing different initial circles
blobsizes = numpy.linspace(13,22,10)


#   store all the aggregate data
data = [] 
tempdata = []

#   starting population for strategy 2
pop2 = 100


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
            
            
            iters = 0            
            
            while 1:
                iters += 1
                alpha = (hi + lo)/2.0
                
                
                #   carrying capacity for pop1                
                pop1 = (alpha + 2)/.02
                                
                
                if iters == max_iters: 
                    data.append([eps, alpha, pop1,pop2, mu1, mu2, mu1/mu2, blob, 3])
                    tempdata.append([eps, alpha, pop1,pop2, mu1, mu2, mu1/mu2, blob, 3])
                    bdata.append([eps, alpha, pop1, pop2, mu1, mu2, mu1/mu2, blob, 3])
                    break
                
                s1_freq = numpy.zeros(length).astype(numpy.float64)
                s2_freq = numpy.zeros(length).astype(numpy.float64)
                
                #   create the initial circle
                for i in range(length):
                    x = i/side
                    y = i%side
                    if distance(x,y,middle_x, middle_y) <= blob:
                        s1_freq[i] = pop1
                        s2_freq[i] = 0
                    else:
                        s1_freq[i] = 0
                        s2_freq[i] = pop2

                
                time = 0
                
                while 1:
                    picture = numpy.zeros(length)
                                        
                    print(blob, iters, alpha, mu1, mu2)
                    for count in range(1000):
                        fitness(
                            drv.InOut(s1_freq), drv.InOut(s2_freq), numpy.float64(eps), numpy.float64(alpha),
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
                    
                    if time == 1:
                        before = sum(picture)
                    
                    if time == 4:
                        after  = sum(picture)

                        if before > after:
                            tempdata.append( [eps, alpha, pop1, pop2, mu1, mu2, mu1/mu2, blob, 2])
                            bdata.append([eps, alpha, pop1, pop2, mu1, mu2, mu1/mu2, blob, 2])
                            lo = alpha
                            break
                        else:
                            tempdata.append( [eps, alpha, pop1, pop2, mu1, mu2, mu1/mu2, blob, 1])
                            bdata.append([eps, alpha, pop1, pop2, mu1, mu2, mu1/mu2, blob, 1])
                            hi = alpha
                            break



        #   saves a text file specific to the blob size
    
        cwd = os.getcwd()
        filename = "cressman2D_blob"+str(blob)+".txt"

        filepath = cwd + "/" + filename


        if not os.path.isfile(filepath):
            f = open(filename,'w+')
            f.write("epsilon, alpha, pop1, pop2, dif1, dif2, difratio, blobsize, winner"  + "\n")
            for i in range(len(bdata)):
                temp = bdata[i][0], bdata[i][1], bdata[i][2], bdata[i][3],bdata[i][4],bdata[i][5],bdata[i][6],bdata[i][7], bdata[i][8]
                temp2= str(temp)
                text = temp2[1:-1]
                f.write( str(text)[1:len(str(bdata))-1] + "\n")
        else:
            f = open(filename, 'a+')
            for i in range(len(bdata)):
                temp = bdata[i][0], bdata[i][1], bdata[i][2], bdata[i][3],bdata[i][4],bdata[i][5],bdata[i][6],bdata[i][7], bdata[i][8]
                temp2= str(temp)
                text = temp2[1:-1]
                f.write( str(text)[1:len(str(bdata))-1] + "\n")

        f.close()








#   saves a text file with parameters
finally:
    cwd = os.getcwd()
    filename = "cressman2D_blob.txt"

    filepath = cwd + "/" + filename


    if not os.path.isfile(filepath):
        f = open(filename,'w+')
        f.write("epsilon, alpha, pop1, pop2, dif1, dif2, difratio, blobsize, winner"  + "\n")
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
    





