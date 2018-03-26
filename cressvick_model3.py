# -*- coding: utf-8 -*-
"""
Created on Thu Feb  1 16:44:42 2018

@author: cody
"""


import pycuda.autoinit
import pycuda.driver as drv
import numpy
import matplotlib.pyplot as plt
from pycuda.compiler import SourceModule


numpy.random.seed(3387)
#numpy.random.seed(12457)

mod = SourceModule("""
#include <math.h>

__global__ void fitness1(double *x, double *y, double *z, double eps)
{
  const int i = threadIdx.x + blockDim.x * blockIdx.x;
  
  const float a1 = 0;
  const float a2 = -1;  
  const float a3 = 1;
  
  const float b1 = 1;
  const float b2 = 0;
  const float b3 = -1;
  
  const float c1 = -1;
  const float c2 = 1;
  const float c3 = 0;
  
  double tempx = x[i];
  double tempy = y[i];
  double tempz = z[i];
  
  double pop = tempx + tempy + tempz;    
  
  double xsq = pow(tempx,2);
  double ysq = pow(tempy,2);
  double zsq = pow(tempz,2);
  
  double avgfit = xsq*a1 + ysq*b2 + zsq*c3 + tempx*tempy*(b1+a2) + tempx*tempz*(c1+a3) + tempy*tempz*(c2+b3);
  avgfit = avgfit/(pop*pop);  
  
  double lambda1 = 50;
  double lambda2 = .01;
  avgfit = lambda1 - lambda2*pop;
  
  x[i] = tempx + tempx*eps*(   (tempx*a1 + tempy*b1 + tempz*c1)/pop  -  avgfit  );
  y[i] = tempy + tempy*eps*(   (tempx*a2 + tempy*b2 + tempz*c2)/pop  -  avgfit  );
  z[i] = tempz + tempz*eps*(   (tempx*a3 + tempy*b3 + tempz*c3)/pop  -  avgfit  ); 
}


__global__ void diffuse_display(double *destx, double *desty, double *destz, double *x, double *y, double *z, double eps, int side)
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
  
  //diffusion rates
  double mu1 = .5;
  double mu2 = .5;
  double mu3 = .5;  
  
  double mu1eps = mu1*eps;
  double mu2eps = mu2*eps;
  double mu3eps = mu3*eps;    
  
  double d = 4+4/sqrt(2.0);
      
  destx[i] = (1-mu1eps)*x[i] + (( mu1eps ) / d ) * (x[N]+x[S]+x[W]+x[E]) + ( (1/sqrt(2.0) ) * mu1eps / d )*(x[NE]+x[NW]+x[SE]+x[SW]);
  desty[i] = (1-mu2eps)*y[i] + (( mu2eps ) / d ) * (y[N]+y[S]+y[W]+y[E]) + ( (1/sqrt(2.0) ) * mu2eps / d )*(y[NE]+y[NW]+y[SE]+y[SW]);
  destz[i] = (1-mu3eps)*z[i] + (( mu3eps ) / d ) * (z[N]+z[S]+z[W]+z[E]) + ( (1/sqrt(2.0) ) * mu3eps / d )*(z[NE]+z[NW]+z[SE]+z[SW]);
}

""")


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



fitness1 = mod.get_function("fitness1")
diffuse_display = mod.get_function("diffuse_display")


side = 256
length = side*side

eps = .01

#   *x, *y, *z
s1_freq = numpy.random.randint(1,100,size=length).astype(numpy.float64)       
s2_freq = numpy.random.randint(1,100,size=length).astype(numpy.float64)       
s3_freq = numpy.random.randint(1,100,size=length).astype(numpy.float64)


#s1_freq = numpy.zeros(length).astype(numpy.float64)
#s2_freq = numpy.zeros(length).astype(numpy.float64)
#s3_freq = numpy.zeros(length).astype(numpy.float64)
#
#for i in range(5504):
#    s1_freq[i] = 50
#    s2_freq[i + 5504] = 50
#
#for i in range(11008,16384):
#    s3_freq[i] = 50



s1_rat = s1_freq/(s1_freq + s2_freq + s3_freq)
s2_rat = s2_freq/(s1_freq + s2_freq + s3_freq)
s3_rat = s3_freq/(s1_freq + s2_freq + s3_freq)


#   picture
tuples = zip(s1_rat, s2_rat, s3_rat)
picture = numpy.reshape(tuples,(side,side,3))


#   destx and desty
destx = numpy.zeros(length).astype(numpy.float64)
desty = numpy.zeros(length).astype(numpy.float64)
destz = numpy.zeros(length).astype(numpy.float64)


p = plt.imshow(picture,interpolation = 'nearest')
plt.gca().invert_yaxis()
plt.pause(2)



datax = numpy.random.randint(0,side)
datay = numpy.random.randint(0,side)

rock = []
paper = []
scissor = []


try:
    while 1:
        for count in range(1):    
            fitness1(
                drv.InOut(s1_freq), drv.InOut(s2_freq), drv.InOut(s3_freq), numpy.float64(eps),
                block = (1024,1,1), grid=(length/1024,1))

            destx = s1_freq.copy()
            desty = s2_freq.copy()
            destz = s3_freq.copy()   
                
            diffuse_display(
                drv.Out(destx), drv.Out(desty),drv.Out(destz), drv.In(s1_freq), drv.In(s2_freq),drv.In(s3_freq),
                numpy.float64(eps),numpy.int64(side), block = (1024,1,1), grid=(length/1024,1))
        
            s1_freq = destx.copy()
            s2_freq = desty.copy()
            s3_freq = destz.copy()
        
        rock.append(s1_freq[datax*datay])
        scissor.append(s2_freq[datax*datay])
        paper.append(s3_freq[datax*datay])        
        
        s1_rat = destx/(destx + desty + destz)
        s2_rat = desty/(destx + desty + destz)
        s3_rat = destz/(destx + desty + destz)       

        tuples = zip(s1_rat, s2_rat, s3_rat)  
        gridpic = numpy.reshape(tuples,(side,side,3))
        
        p.set_data(gridpic)
        plt.pause(0.00001)

except KeyboardInterrupt:
    pass


fig = plt.figure()
xx = range(len(rock))
pic = fig.add_subplot(111)
pic.plot(xx,rock,'r',xx,paper,'g',xx,scissor,'c')