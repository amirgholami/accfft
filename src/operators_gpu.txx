/*
 *  Copyright (c) 2014-2015, Amir Gholami, George Biros
 *  All rights reserved.
 *  This file is part of AccFFT library.
 *
 *  AccFFT is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 2 of the License, or
 *  (at your option) any later version.
 *
 *  AccFFT is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with AccFFT.  If not, see <http://www.gnu.org/licenses/>.
 *
*/

#ifndef OPERATORS_GPU_TXX
#define OPERATORS_GPU_TXX
template <typename Tc>
void grad_mult_wave_numberx_gpu(Tc* wA, Tc* A, int* N, int * osize, int * ostart, std::bitset<3> xyz);
template <typename Tc>
void grad_mult_wave_numbery_gpu(Tc* wA, Tc* A, int* N, int * osize, int * ostart, std::bitset<3> xyz);
template <typename Tc>
void grad_mult_wave_numberz_gpu(Tc* wA, Tc* A, int* N, int * osize, int * ostart, std::bitset<3> xyz);
template <typename Tc>
void laplace_mult_wave_number_gpu(Tc* wA, Tc* A, int* N, int * osize, int * ostart);
template <typename T>
void daxpy_gpu(const long long int n, const T alpha, T* x, T* y);

extern "C"{
/* Double Precision */
void grad_mult_wave_numberx_gpu_c(Complex* wA, Complex* A, int*n, int * osize, int * ostart, std::bitset<3> xyz);
void grad_mult_wave_numbery_gpu_c(Complex* wA, Complex* A, int*n, int * osize, int * ostart, std::bitset<3> xyz);
void grad_mult_wave_numberz_gpu_c(Complex* wA, Complex* A, int*n, int * osize, int * ostart, std::bitset<3> xyz);
void laplace_mult_wave_number_gpu_c(Complex* wA, Complex* A, int*n, int * osize, int * ostart);
void daxpy_gpu_c(const long long int n, const double alpha, double *x, double* y);
/* Single Precision */
void grad_mult_wave_numberx_gpu_cf(Complexf* wA, Complexf* A, int*n, int * osize, int * ostart, std::bitset<3> xyz);
void grad_mult_wave_numbery_gpu_cf(Complexf* wA, Complexf* A, int*n, int * osize, int * ostart, std::bitset<3> xyz);
void grad_mult_wave_numberz_gpu_cf(Complexf* wA, Complexf* A, int*n, int * osize, int * ostart, std::bitset<3> xyz);
void laplace_mult_wave_number_gpu_cf(Complexf* wA, Complexf* A, int*n, int * osize, int * ostart);
void daxpy_gpu_cf(const long long int n, const float alpha, float *x, float* y);
}

/* Double Precision Instantiation */
template <> void grad_mult_wave_numberx_gpu<Complex>(Complex* wA, Complex* A, int* N, int * osize, int * ostart, std::bitset<3> xyz){
  grad_mult_wave_numberx_gpu_c(wA, A, N, osize,ostart,xyz);
}
template <> void grad_mult_wave_numbery_gpu<Complex>(Complex* wA, Complex* A, int* N, int * osize, int * ostart, std::bitset<3> xyz){
  grad_mult_wave_numbery_gpu_c(wA, A, N, osize,ostart,xyz);
}
template <> void grad_mult_wave_numberz_gpu<Complex>(Complex* wA, Complex* A, int* N, int * osize, int * ostart, std::bitset<3> xyz){
  grad_mult_wave_numberz_gpu_c(wA, A, N, osize,ostart,xyz);
}
template <> void laplace_mult_wave_number_gpu<Complex>(Complex* wA, Complex* A, int* N, int * osize, int * ostart){
  laplace_mult_wave_number_gpu_c(wA, A, N, osize,ostart);
}
template <> void daxpy_gpu<double>(const long long int n, const double alpha, double* x, double* y){
  daxpy_gpu_c(n,alpha,x,y);
}

/* Single Precision Instantiation */
template <> void grad_mult_wave_numberx_gpu<Complexf>(Complexf* wA, Complexf* A, int* N, int * osize, int * ostart, std::bitset<3> xyz){
  grad_mult_wave_numberx_gpu_cf(wA, A, N, osize,ostart,xyz);
}
template <> void grad_mult_wave_numbery_gpu<Complexf>(Complexf* wA, Complexf* A, int* N, int * osize, int * ostart, std::bitset<3> xyz){
  grad_mult_wave_numbery_gpu_cf(wA, A, N, osize,ostart,xyz);
}
template <> void grad_mult_wave_numberz_gpu<Complexf>(Complexf* wA, Complexf* A, int* N, int * osize, int * ostart, std::bitset<3> xyz){
  grad_mult_wave_numberz_gpu_cf(wA, A, N, osize,ostart,xyz);
}
template <> void laplace_mult_wave_number_gpu<Complexf>(Complexf* wA, Complexf* A, int* N, int * osize, int * ostart){
  laplace_mult_wave_number_gpu_cf(wA, A, N, osize,ostart);
}
template <> void daxpy_gpu<float>(const long long int n, const float alpha, float* x, float* y){
  daxpy_gpu_cf(n,alpha,x,y);
}



template <typename T, typename Tp>
void accfft_grad_gpu_t(T* A_x, T* A_y, T*A_z, T* A,Tp* plan, std::bitset<3> *pXYZ, double* timer){
  typedef T Tc[2];
	int procid;
  MPI_Comm c_comm=plan->c_comm;
  MPI_Comm_rank(c_comm,&procid);
  if(!plan->r2c_plan_baked){
    PCOUT<<"Error in accfft_grad! plan is not correctly made."<<std::endl;
    return;
  }

  double * timings;
  if(timer==NULL){
    timings=new double[5];
    memset(timings,0,sizeof(double)*5);
  }
  else{
    timings=timer;
  }

  std::bitset<3> XYZ;
  if(pXYZ!=NULL){
    XYZ=*pXYZ;
  }
  else{
    XYZ[0]=1; XYZ[1]=1; XYZ[2]=1;
  }

	double self_exec_time= - MPI_Wtime();
  int *N=plan->N;

  int isize[3],osize[3],istart[3],ostart[3];
  long long int alloc_max;
  /* Get the local pencil size and the allocation size */
  alloc_max=accfft_local_size_dft_r2c_gpu_t<T>(N,isize,istart,osize,ostart,c_comm);
  //PCOUT<<"istart[0]= "<<istart[0]<<" istart[1]= "<<istart[1]<<" istart[2]="<<istart[2]<<std::endl;
  //PCOUT<<"ostart[0]= "<<ostart[0]<<" ostart[1]= "<<ostart[1]<<" ostart[2]="<<ostart[2]<<std::endl;

  Tc* A_hat;//=(Tc*) accfft_alloc(alloc_max);
  Tc* tmp  ;//=(Tc*) accfft_alloc(alloc_max);
  cudaMalloc((void**) &A_hat, alloc_max);
  cudaMalloc((void**) &tmp, alloc_max);
  std::bitset<3> scale_xyz(0);
  scale_xyz[0]=1;
  scale_xyz[1]=1;
  scale_xyz[2]=1;

	MPI_Barrier(c_comm);

	/* Forward transform */
  accfft_execute_r2c_gpu_t<T,Tc>(plan,A,A_hat,timings);

	/* Multiply x Wave Numbers */
	if(XYZ[0]){
		grad_mult_wave_numberx_gpu<Tc>(tmp,A_hat, N,osize,ostart,scale_xyz);
		MPI_Barrier(c_comm);

		/* Backward transform */
    accfft_execute_c2r_gpu_t<Tc,T>(plan,tmp,A_x,timings);
	}
	/* Multiply y Wave Numbers */
	if(XYZ[1]){
		grad_mult_wave_numbery_gpu<Tc>(tmp,A_hat, N,osize,ostart,scale_xyz);
		/* Backward transform */
    accfft_execute_c2r_gpu_t<Tc,T>(plan,tmp,A_y,timings);
	}

	/* Multiply z Wave Numbers */
	if(XYZ[2]){
		grad_mult_wave_numberz_gpu<Tc>(tmp,A_hat, N,osize,ostart,scale_xyz);
		/* Backward transform */
    accfft_execute_c2r_gpu_t<Tc,T>(plan,tmp,A_z,timings);
	}

	cudaFree(A_hat);
	cudaFree(tmp);

	self_exec_time+= MPI_Wtime();

  if(timer==NULL){
    delete [] timings;
  }
	return;
}

template <typename T,typename Tp>
void accfft_laplace_gpu_t(T* LA, T* A, Tp* plan, double* timer){
  typedef T Tc[2];
	int procid;
  MPI_Comm c_comm=plan->c_comm;
  MPI_Comm_rank(c_comm,&procid);
  if(!plan->r2c_plan_baked){
    PCOUT<<"Error in accfft_grad! plan is not correctly made."<<std::endl;
    return;
  }

  double * timings;
  if(timer==NULL){
    timings=new double[5];
    memset(timings,0,sizeof(double)*5);
  }
  else{
    timings=timer;
  }

	double self_exec_time= - MPI_Wtime();
  int *N=plan->N;

  int isize[3],osize[3],istart[3],ostart[3];
  long long int alloc_max;
  /* Get the local pencil size and the allocation size */
  alloc_max=accfft_local_size_dft_r2c_gpu_t<T>(N,isize,istart,osize,ostart,c_comm);

  Tc* A_hat;//=(Tc*) accfft_alloc(alloc_max);
  Tc* tmp  ;//=(Tc*) accfft_alloc(alloc_max);
  cudaMalloc((void**) &A_hat, alloc_max);
  cudaMalloc((void**) &tmp, alloc_max);
	MPI_Barrier(c_comm);

	/* Forward transform */
  accfft_execute_r2c_gpu_t<T,Tc>(plan,A,A_hat,timings);

  /* Multiply x Wave Numbers */
  laplace_mult_wave_number_gpu<Tc>(tmp,A_hat, N,osize,ostart);
  MPI_Barrier(c_comm);

  /* Backward transform */
  accfft_execute_c2r_gpu_t<Tc,T>(plan,tmp,LA,timings);

  cudaFree(A_hat);
  cudaFree(tmp);

  self_exec_time+= MPI_Wtime();

  if(timer==NULL){
    delete [] timings;
  }
  return;
}

template <typename T, typename Tp>
void accfft_divergence_gpu_t(T* div_A, T* A_x, T* A_y, T* A_z, Tp* plan, double* timer){
  typedef T Tc[2];
	int procid;
  MPI_Comm c_comm=plan->c_comm;
  MPI_Comm_rank(c_comm,&procid);

  if(!plan->r2c_plan_baked){
    PCOUT<<"Error in accfft_grad! plan is not correctly made."<<std::endl;
    return;
  }
  double * timings;
  if(timer==NULL){
    timings=new double[5];
    memset(timings,0,sizeof(double)*5);
  }
  else{
    timings=timer;
  }

	double self_exec_time= - MPI_Wtime();
  int *N=plan->N;

  int isize[3],osize[3],istart[3],ostart[3];
  long long int alloc_max;
  /* Get the local pencil size and the allocation size */
  alloc_max=accfft_local_size_dft_r2c_gpu_t<T>(N,isize,istart,osize,ostart,c_comm);
  //PCOUT<<"istart[0]= "<<istart[0]<<" istart[1]= "<<istart[1]<<" istart[2]="<<istart[2]<<std::endl;
  //PCOUT<<"ostart[0]= "<<ostart[0]<<" ostart[1]= "<<ostart[1]<<" ostart[2]="<<ostart[2]<<std::endl;

  Tc* A_hat;//=(Tc*) accfft_alloc(alloc_max);
  Tc* tmp  ;//=(Tc*) accfft_alloc(alloc_max);
  T* tmp2  ;//=(Tc*) accfft_alloc(alloc_max);
  cudaMalloc((void**) &A_hat, alloc_max);
  cudaMalloc((void**) &tmp, alloc_max);
  cudaMalloc((void**) &tmp2, alloc_max);
  std::bitset<3> scale_xyz(0);
  std::bitset<3> xyz=(0);
  scale_xyz[0]=1;
  scale_xyz[1]=1;
  scale_xyz[2]=1;

	MPI_Barrier(c_comm);

  /* Forward transform in x direction*/
  xyz[0]=1;
  xyz[1]=0;
  xyz[2]=1;
  accfft_execute_r2c_gpu_t<T,Tc>(plan,A_x,A_hat,timings,xyz);
  /* Multiply x Wave Numbers */
  grad_mult_wave_numberx_gpu<Tc>(tmp,A_hat, N,osize,ostart,xyz);
  MPI_Barrier(c_comm);
  /* Backward transform */
  accfft_execute_c2r_gpu_t<Tc,T>(plan,tmp,tmp2,timings,xyz);

  //memcpy(div_A,tmp2,isize[0]*isize[1]*isize[2]*sizeof(T));
  cudaMemcpy(div_A,tmp2,isize[0]*isize[1]*isize[2]*sizeof(T),cudaMemcpyDeviceToDevice);

  /* Forward transform in y direction*/
  xyz[0]=0;
  xyz[1]=1;
  xyz[2]=1;
  accfft_execute_r2c_gpu_t<T,Tc>(plan,A_y,A_hat,timings,xyz);
  /* Multiply y Wave Numbers */
  grad_mult_wave_numbery_gpu<Tc>(tmp,A_hat, N,osize,ostart,xyz);
  MPI_Barrier(c_comm);
  /* Backward transform */
  accfft_execute_c2r_gpu_t<Tc,T>(plan,tmp,tmp2,timings,xyz);

  //for (int i=0;i<isize[0]*isize[1]*isize[2];++i)
  //  div_A[i]+=tmp2[i];
  const long long int n=isize[0]*isize[1]*isize[2];
  double alpha=1.0;
  daxpy_gpu<T>(n,alpha,tmp2,div_A);


  /* Forward transform in z direction*/
  xyz[0]=0;
  xyz[1]=0;
  xyz[2]=1;
  accfft_execute_r2c_gpu_t<T,Tc>(plan,A_z,A_hat,timings,xyz);
  /* Multiply z Wave Numbers */
  grad_mult_wave_numberz_gpu<Tc>(tmp,A_hat, N,osize,ostart,xyz);
  MPI_Barrier(c_comm);
  /* Backward transform */
  accfft_execute_c2r_gpu_t<Tc,T>(plan,tmp,tmp2,timings,xyz);

  //for (int i=0;i<isize[0]*isize[1]*isize[2];++i)
  //  div_A[i]+=tmp2[i];
  daxpy_gpu<T>(n,alpha,tmp2,div_A);

	cudaFree(A_hat);
	cudaFree(tmp);
	cudaFree(tmp2);

	self_exec_time+= MPI_Wtime();

  if(timer==NULL){
    delete [] timings;
  }
	return;
}




#endif
