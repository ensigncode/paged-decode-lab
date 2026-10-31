#include <cuda_runtime.h>
#include <cstdio>
__global__ void valid(float *p) { p[threadIdx.x+1] = float(threadIdx.x); }
int main(){float*p;cudaError_t e=cudaMalloc(&p,32*sizeof(float));if(e!=cudaSuccess)return 2;valid<<<1,32>>>(p);e=cudaGetLastError();if(e!=cudaSuccess){puts(cudaGetErrorString(e));return 3;}e=cudaDeviceSynchronize();cudaFree(p);return e==cudaSuccess?0:4;}
