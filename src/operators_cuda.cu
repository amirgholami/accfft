/*
 * File: operators_cuda.cu
 * License: Please see LICENSE file.
 * AccFFT: Massively Parallel FFT Library
 * Created by Amir Gholami on 11/30/2015
 * Email: contact@accfft.org
 */
#include <stdio.h>
#include <bitset>
//#include <accfft_common.h>
typedef double Complex[2];
typedef float Complexf[2];

/* Global Functions */
template <typename Tc>
__global__ void grad_mult_wave_numberx_cu(Tc* wA, Tc* A, int* N, int* osize, int * ostart, double scale ){
  unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
  unsigned int j = blockDim.y * blockIdx.y + threadIdx.y;
  unsigned int k = blockDim.z * blockIdx.z + threadIdx.z;

  if(i>=osize[0]) return;
  if(j>=osize[1]) return;
  if(k>=osize[2]) return;


  {
    long int X,wave;
    long int ptr;
    X=(i+ostart[0]);
    //Y=(j+ostart[1]);
    //Z=(k+ostart[2]);

    wave=X;

    if(X>N[0]/2)
      wave-=N[0];
    if(X==N[0]/2)
      wave=0; // Filter Nyquist

    ptr=(i*osize[1]+j)*osize[2]+k;
    wA[ptr][0] =-scale*wave*A[ptr][1];
    wA[ptr][1] = scale*wave*A[ptr][0];

  }

  return;
} // end grad_mult_wave_numberx_cu

template <typename Tc>
__global__ void grad_mult_wave_numbery_cu(Tc* wA, Tc* A, int* N, int* osize, int * ostart, double scale ){
  unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
  unsigned int j = blockDim.y * blockIdx.y + threadIdx.y;
  unsigned int k = blockDim.z * blockIdx.z + threadIdx.z;

  if(i>=osize[0]) return;
  if(j>=osize[1]) return;
  if(k>=osize[2]) return;


  {
    long int Y,wave;
    long int ptr;
    //X=(i+ostart[0]);
    Y=(j+ostart[1]);
    //Z=(k+ostart[2]);

    wave=Y;

    if(Y>N[1]/2)
      wave-=N[1];
    if(Y==N[1]/2)
      wave=0; // Filter Nyquist

    ptr=(i*osize[1]+j)*osize[2]+k;
    wA[ptr][0] =-scale*wave*A[ptr][1];
    wA[ptr][1] = scale*wave*A[ptr][0];

  }

  return;
} // end grad_mult_wave_numberx_cu

template <typename Tc>
__global__ void grad_mult_wave_numberz_cu(Tc* wA, Tc* A, int* N, int* osize, int * ostart, double scale ){
  unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
  unsigned int j = blockDim.y * blockIdx.y + threadIdx.y;
  unsigned int k = blockDim.z * blockIdx.z + threadIdx.z;

  if(i>=osize[0]) return;
  if(j>=osize[1]) return;
  if(k>=osize[2]) return;


  {
    long int Z,wave;
    long int ptr;
    //X=(i+ostart[0]);
    //Y=(j+ostart[1]);
    Z=(k+ostart[2]);

    wave=Z;

    if(Z>N[2]/2)
      wave-=N[2];
    if(Z==N[2]/2)
      wave=0; // Filter Nyquist

    ptr=(i*osize[1]+j)*osize[2]+k;
    wA[ptr][0] =-scale*wave*A[ptr][1];
    wA[ptr][1] = scale*wave*A[ptr][0];

  }

  return;
} // end grad_mult_wave_numberx_cu

template <typename Tc>
__global__ void laplace_mult_wave_number_cu(Tc* wA, Tc* A, int* N, int* osize, int * ostart, double scale ){
  unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
  unsigned int j = blockDim.y * blockIdx.y + threadIdx.y;
  unsigned int k = blockDim.z * blockIdx.z + threadIdx.z;

  if(i>=osize[0]) return;
  if(j>=osize[1]) return;
  if(k>=osize[2]) return;

  {
    long int X,Y,Z,wx,wy,wz,wave;
    long int ptr;
    X=(i+ostart[0]);
    Y=(j+ostart[1]);
    Z=(k+ostart[2]);

    wx=X;
    wy=Y;
    wz=Z;

    if(X>N[0]/2)
      wx-=N[0];
    if(X==N[0]/2)
      wx=0;

    if(Y>N[1]/2)
      wy-=N[1];
    if(Y==N[1]/2)
      wy=0;

    if(Z>N[2]/2)
      wz-=N[2];
    if(Z==N[2]/2)
      wz=0;

    wave=-wx*wx-wy*wy-wz*wz;

    ptr=(i*osize[1]+j)*osize[2]+k;
    wA[ptr][0] = scale*wave*A[ptr][0];
    wA[ptr][1] = scale*wave*A[ptr][1];
  }


  return;
} // end grad_mult_wave_numberx_cu

template <typename T>
__global__ void daxpy_cu(const long long int n,const T  alpha, T* x, T* y){
  // BLAS SAXPY( n, alpha, x, incx, y, incy )
  unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;

  if(i>=n) return;
  y[i]+=x[i]*alpha;
  return;
}


/* Host Functions */
template <typename Tc>
void grad_mult_wave_numberx_gpu_(Tc* wA, Tc* A, int* N, int * osize, int * ostart, std::bitset<3> xyz) {

  double scale=1;
  if(xyz[0]) scale*=N[0];
  if(xyz[1]) scale*=N[1];
  if(xyz[2]) scale*=N[2];
  scale=1./scale;

  // corresponding GPU sizes
  int * n_gpu, *osize_gpu, *ostart_gpu;
  cudaMalloc((void**) &n_gpu,3*sizeof(int));
  cudaMalloc((void**) &osize_gpu,3*sizeof(int));
  cudaMalloc((void**) &ostart_gpu,3*sizeof(int));

  // Copy the sizes to GPU
  cudaMemcpy(n_gpu, N, 3*sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(osize_gpu, osize, 3*sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(ostart_gpu, ostart, 3*sizeof(int), cudaMemcpyHostToDevice);



  int blocksInX = std::ceil(N[0]/4.);
  int blocksInY = std::ceil(N[1]/4.);
  int blocksInZ = std::ceil(N[2]/4.);

  dim3 Dg(blocksInX, blocksInY, blocksInZ);
  dim3 Db(4, 4, 4);
  grad_mult_wave_numberx_cu<<<Dg, Db>>>(wA, A,n_gpu,osize_gpu,ostart_gpu,scale);
  cudaDeviceSynchronize();

  cudaFree(n_gpu);
  cudaFree(osize_gpu);
  cudaFree(ostart_gpu);
  return;
} // end grad_mult_wave_numberx_gpu

template <typename Tc>
void grad_mult_wave_numbery_gpu_(Tc* wA, Tc* A, int* N, int * osize, int * ostart, std::bitset<3> xyz) {

  double scale=1;
  if(xyz[0]) scale*=N[0];
  if(xyz[1]) scale*=N[1];
  if(xyz[2]) scale*=N[2];
  scale=1./scale;

  // corresponding GPU sizes
  int * n_gpu, *osize_gpu, *ostart_gpu;
  cudaMalloc((void**) &n_gpu,3*sizeof(int));
  cudaMalloc((void**) &osize_gpu,3*sizeof(int));
  cudaMalloc((void**) &ostart_gpu,3*sizeof(int));

  // Copy the sizes to GPU
  cudaMemcpy(n_gpu, N, 3*sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(osize_gpu, osize, 3*sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(ostart_gpu, ostart, 3*sizeof(int), cudaMemcpyHostToDevice);



  int blocksInX = std::ceil(N[0]/4.);
  int blocksInY = std::ceil(N[1]/4.);
  int blocksInZ = std::ceil(N[2]/4.);

  dim3 Dg(blocksInX, blocksInY, blocksInZ);
  dim3 Db(4, 4, 4);
  grad_mult_wave_numbery_cu<<<Dg, Db>>>(wA, A,n_gpu,osize_gpu,ostart_gpu,scale);
  cudaDeviceSynchronize();

  cudaFree(n_gpu);
  cudaFree(osize_gpu);
  cudaFree(ostart_gpu);
  return;
} // end grad_mult_wave_numberx_gpu

template <typename Tc>
void grad_mult_wave_numberz_gpu_(Tc* wA, Tc* A, int* N, int * osize, int * ostart, std::bitset<3> xyz) {

  double scale=1;
  if(xyz[0]) scale*=N[0];
  if(xyz[1]) scale*=N[1];
  if(xyz[2]) scale*=N[2];
  scale=1./scale;

  // corresponding GPU sizes
  int * n_gpu, *osize_gpu, *ostart_gpu;
  cudaMalloc((void**) &n_gpu,3*sizeof(int));
  cudaMalloc((void**) &osize_gpu,3*sizeof(int));
  cudaMalloc((void**) &ostart_gpu,3*sizeof(int));

  // Copy the sizes to GPU
  cudaMemcpy(n_gpu, N, 3*sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(osize_gpu, osize, 3*sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(ostart_gpu, ostart, 3*sizeof(int), cudaMemcpyHostToDevice);



  int blocksInX = std::ceil(N[0]/4.);
  int blocksInY = std::ceil(N[1]/4.);
  int blocksInZ = std::ceil(N[2]/4.);

  dim3 Dg(blocksInX, blocksInY, blocksInZ);
  dim3 Db(4, 4, 4);
  grad_mult_wave_numberz_cu<<<Dg, Db>>>(wA, A,n_gpu,osize_gpu,ostart_gpu,scale);
  cudaDeviceSynchronize();

  cudaFree(n_gpu);
  cudaFree(osize_gpu);
  cudaFree(ostart_gpu);
  return;
} // end grad_mult_wave_numberx_gpu

template <typename Tc>
void laplace_mult_wave_number_gpu_(Tc* wA, Tc* A, int* N, int * osize, int * ostart) {

  double scale=1;
  scale*=N[0];
  scale*=N[1];
  scale*=N[2];
  scale=1./scale;

  // corresponding GPU sizes
  int * n_gpu, *osize_gpu, *ostart_gpu;
  cudaMalloc((void**) &n_gpu,3*sizeof(int));
  cudaMalloc((void**) &osize_gpu,3*sizeof(int));
  cudaMalloc((void**) &ostart_gpu,3*sizeof(int));

  // Copy the sizes to GPU
  cudaMemcpy(n_gpu, N, 3*sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(osize_gpu, osize, 3*sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(ostart_gpu, ostart, 3*sizeof(int), cudaMemcpyHostToDevice);



  int blocksInX = std::ceil(N[0]/4.);
  int blocksInY = std::ceil(N[1]/4.);
  int blocksInZ = std::ceil(N[2]/4.);

  dim3 Dg(blocksInX, blocksInY, blocksInZ);
  dim3 Db(4, 4, 4);
  laplace_mult_wave_number_cu<<<Dg, Db>>>(wA, A,n_gpu,osize_gpu,ostart_gpu,scale);
  cudaDeviceSynchronize();

  cudaFree(n_gpu);
  cudaFree(osize_gpu);
  cudaFree(ostart_gpu);
  return;
} // end grad_mult_wave_numberx_gpu
//template __global__ void grad_mult_wave_nunmberx_cu<Complex>(Complex* wA, Complex* A, int* N, int* osize, int * ostart, double scale );

template <typename T>
void daxpy_gpu_(const long long int n, const T alpha, T* x, T* y){
  // corresponding GPU sizes
  int blocksInX = std::ceil(n/128.);

  dim3 Dg(blocksInX, 1, 1);
  dim3 Db(128,1,1);
  daxpy_cu<<<Dg, Db>>>(n, alpha,x,y);
  cudaDeviceSynchronize();
  return;

}



extern "C"{
/* Double Precision */
void grad_mult_wave_numberx_gpu_c(Complex* wA, Complex* A, int*n, int * osize, int * ostart, std::bitset<3> xyz){
  grad_mult_wave_numberx_gpu_(wA, A, n, osize,ostart,  xyz);
  return;
}
void grad_mult_wave_numbery_gpu_c(Complex* wA, Complex* A, int*n, int * osize, int * ostart, std::bitset<3> xyz){
  grad_mult_wave_numbery_gpu_(wA, A, n, osize,ostart,  xyz);
  return;
}
void grad_mult_wave_numberz_gpu_c(Complex* wA, Complex* A, int*n, int * osize, int * ostart, std::bitset<3> xyz){
  grad_mult_wave_numberz_gpu_(wA, A, n, osize,ostart,  xyz);
  return;
}
void laplace_mult_wave_number_gpu_c(Complex* wA, Complex* A, int*n, int * osize, int * ostart){
  laplace_mult_wave_number_gpu_(wA, A, n, osize,ostart);
  return;
}
void daxpy_gpu_c(const long long int n, const double alpha, double *x, double* y){
  daxpy_gpu_(n,alpha,x,y);
  return;
}

/* Single Precision */
void grad_mult_wave_numberx_gpu_cf(Complexf* wA, Complexf* A, int*n, int * osize, int * ostart, std::bitset<3> xyz){
  grad_mult_wave_numberx_gpu_(wA, A, n, osize,ostart,  xyz);
  return;
}
void grad_mult_wave_numbery_gpu_cf(Complexf* wA, Complexf* A, int*n, int * osize, int * ostart, std::bitset<3> xyz){
  grad_mult_wave_numbery_gpu_(wA, A, n, osize,ostart,  xyz);
  return;
}
void grad_mult_wave_numberz_gpu_cf(Complexf* wA, Complexf* A, int*n, int * osize, int * ostart, std::bitset<3> xyz){
  grad_mult_wave_numberz_gpu_(wA, A, n, osize,ostart,  xyz);
  return;
}
void laplace_mult_wave_number_gpu_cf(Complexf* wA, Complexf* A, int*n, int * osize, int * ostart){
  laplace_mult_wave_number_gpu_(wA, A, n, osize,ostart);
  return;
}
void daxpy_gpu_cf(const long long int n, const float alpha, float* x, float* y){
  daxpy_gpu_(n,alpha,x,y);
  return;
}

}
