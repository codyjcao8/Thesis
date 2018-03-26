import matplotlib.pyplot as plt

import matplotlib.colors
 
import pycuda.autoinit
import pycuda.driver as drv
import numpy

from pycuda.compiler import SourceModule

# kernal declaration
mod = SourceModule("""
__global__ void nichbail(double *ax, double *ay, double *bx, double *by)
{
  const int i = threadIdx.x+ blockIdx.x*blockDim.x;
  double mux= 0.7125;
  double muy= 0.0375;
  
  for(int k=0; k<100; k++){
      // nich bail
      bx[i]= 2.0*ax[i]*exp(-ay[i]);
      by[i]= ax[i]*(1.0-exp(-ay[i]));
      
      __syncthreads();
      
      // migrate    
      ax[i]= (1-mux)*bx[i] + (mux/2.0)*(bx[(i-1+2048)%2048]+bx[(i+1)%2048]);
      ay[i]= (1-muy)*by[i] + (muy/2.0)*(by[(i-1+2048)%2048]+by[(i+1)%2048]);
      }
}
""")

nichbail = mod.get_function("nichbail")

numpy.random.seed(125897)

u= 2.0*numpy.log(2.0)
v= numpy.log(2.0)

ax = u+.05*(2.0*numpy.random.rand(2048).astype(numpy.float64)-1.0)
ay = v+.05*(2.0*numpy.random.rand(2048).astype(numpy.float64)-1.0);

bx = numpy.zeros_like(ax)
by = numpy.zeros_like(ay)

plt.figure(57)
count= 0

while 1:
    nichbail(
            drv.InOut(ax), drv.InOut(ay), drv.InOut(bx), drv.InOut(by),
            block=(512,1,1), grid=(4,1))

    V=numpy.ones(2048)
    U=numpy.ones(2048)

    MAX=5 
    
    Z= ax;#+ay;
    for i in range(2048):
        V[i]= 100*numpy.log(1+Z[i])/MAX
        U[i]= 100*numpy.log(1+ay[i])/MAX

    plt.clf()  
    axx= plt.subplot(2,2,1)    
    axx.set_ylim([-20,6000])
    plt.plot(V,'.',color="purple",linewidth=3)
    axy= plt.subplot(2,2,2)          
    axy.set_ylim([-20,6000])
    plt.plot(U,'.',color="slategrey",linewidth=3)
    
    print '%.18le  %.18le ' % (ax[1],ay[1])
    
    for i in range(2048):
        if numpy.log(1+Z)[i]>MAX :
            V[i]= 100
        if numpy.log(1+U)[i]>MAX :
            U[i]= 100

    axx= plt.subplot(2,2,3)    
    axx.set_ylim([-1,101])
    plt.plot(V,'.',color="purple",linewidth=3)
    axy= plt.subplot(2,2,4)          
    axy.set_ylim([-1,101])
    plt.plot(U,'.',color="slategrey",linewidth=3)
    #count+=1
    #if count%100==0: print(count)            
    plt.pause(.0000001)