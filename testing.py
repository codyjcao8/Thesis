
import pycuda.autoinit
import pycuda.driver as drv
import numpy
import time
import matplotlib.pyplot as plt
from pycuda.compiler import SourceModule


start = time.time()

numpy.random.seed(156)


"""
a,b,c,d = 1,0,0,1
    - end picture is symmetrical

"""

mod = SourceModule("""
__global__ void fitness(double *dest, double *x)
{
  const int i = blockIdx.x*blockDim.x+threadIdx.x;
  const int a = 1;
  const int b = 0;
  const int c = 0;
  const int d = 1;
  dest[i] = x[i] + 0.10*x[i]*(((a-b)*x[i] + b) - ( (a+d-c-b)*x[i]*x[i] + (c+b-2*d)*x[i] + d));
}

__global__ void diffuse(double *dest, double *a)
{
  const int i = blockIdx.x*blockDim.x + threadIdx.x;
  const int ROW = i/64;
  const int COLUMN = i%64;
  const int above = (ROW + (COLUMN - 1)*64)%4096;
  const int below = (ROW + (COLUMN + 1)*64)%4096;
  const int right = (ROW + 1 + (COLUMN*64))%4096;
  const int left = (ROW - 1 + (COLUMN*64))%4096;
  dest[i] = .9*a[i] + .025*a[above]+ .025*a[below]+ .025*a[left]+ .025*a[right];
}

__global__ void test(int *a, int *b)
{
  const int i = (blockIdx.x*blockDim.x) + threadIdx.x;
  a[i] = blockIdx.x;
  b[i] = threadIdx.x;
}

__global__ void print_thread(int *a)
{
    const int i = threadIdx.x;
    __syncthreads();
    a[i] = i;
}
"""
)

a = numpy.zeros(10, dtype = numpy.float32)

print_thread = mod.get_function("print_thread")

print_thread(
    drv.InOut(a), block = (10,10,1),grid = (1,1)
)


print(a)

