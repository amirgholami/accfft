/*
 * File: kernelsf.cu
 * License: Please see LICENSE file.
 * AccFFT: Massively Parallel FFT Library
 * Created by Amir Gholami on 11/23/2015
 * Email: contact@accfft.org
 */
#include <stdio.h>

__device__ inline float testcase_gpu(float X,float Y,float Z){

  float sigma= 4;
  float pi=M_PI;
  float analytic;
  analytic=std::exp( -sigma * ( (X-pi)*(X-pi) + (Y-pi)*(Y-pi) + (Z-pi)*(Z-pi) ));
  if(analytic!=analytic) analytic=0;
  return analytic;
}
__global__ void initialize_gpu_kernel(float * a, int *n, int n2_, int* isize, int* istart ){
  unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
  unsigned int j = blockDim.y * blockIdx.y + threadIdx.y;
  unsigned int k = blockDim.z * blockIdx.z + threadIdx.z;

  if(i>=isize[0]) return;
  if(j>=isize[1]) return;
  if(k>=isize[2]) return;

  {
    float pi=M_PI;
    float X,Y,Z;
    long int ptr;

    X=2*pi/n[0]*(i+istart[0]);
    Y=2*pi/n[1]*(j+istart[1]);
    Z=2*pi/n[2]*k;

    ptr=i*isize[1]*n2_+j*n2_+k;
    a[ptr]=testcase_gpu(X,Y,Z);
  }
  return;

}// end initialize_gpu_kernel
void initialize_gpu(float *a,int*n, int * isize, int * istart) {

  int n2_=n[2]; // Outplace Transform requires no padding

  // corresponding GPU sizes
  int * n_gpu, *isize_gpu, *istart_gpu;
  cudaMalloc((void**) &n_gpu,3*sizeof(int));
  cudaMalloc((void**) &isize_gpu,3*sizeof(int));
  cudaMalloc((void**) &istart_gpu,3*sizeof(int));

  // Copy the sizes to GPU
  cudaMemcpy(n_gpu, n, 3*sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(isize_gpu, isize, 3*sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(istart_gpu, istart, 3*sizeof(int), cudaMemcpyHostToDevice);



  int blocksInX = std::ceil(n[0]/4.);
  int blocksInY = std::ceil(n[1]/4.);
  int blocksInZ = std::ceil(n2_/4.);

  dim3 Dg(blocksInX, blocksInY, blocksInZ);
  dim3 Db(4, 4, 4);
  initialize_gpu_kernel<<<Dg, Db>>>(a,n_gpu,n2_,isize_gpu,istart_gpu);
  cudaDeviceSynchronize();

  cudaFree(n_gpu);
  cudaFree(isize_gpu);
  cudaFree(istart_gpu);
  return;
} // end initialize_gpu
