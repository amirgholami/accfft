/**
 * @file
 * CPU functions of AccFFT operators
 */
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
#include <mpi.h>
#include <fftw3.h>
#include <omp.h>
#include <iostream>
#include <cmath>
#include <math.h>
#include "transpose.h"
#include <string.h>
#include "accfft.h"
#include "accfft_common.h"
#define VERBOSE 0
#define PCOUT if(procid==0) std::cout
static void grad_mult_wave_nunmberx(Complex* wA, Complex* A, int* N,MPI_Comm c_comm,std::bitset<3> xyz );


static void grad_mult_wave_nunmbery(Complex* wA, Complex* A, int* N,MPI_Comm c_comm ,std::bitset<3> xyz){

	int procid;
	MPI_Comm_rank(c_comm,&procid);

	double scale=1;
  if(xyz[0]) scale*=N[0];
  if(xyz[1]) scale*=N[1];
  if(xyz[2]) scale*=N[2];
  //PCOUT<<scale<<std::endl;
  scale=1./scale;

  int istart[3], isize[3], osize[3],ostart[3];
  accfft_local_size_dft_r2c(N,isize,istart,osize,ostart,c_comm);

#pragma omp parallel
  {
    double X,Y,Z,wave;
    long int ptr;
#pragma omp for
    for (int i=0; i<osize[0]; i++){
      for (int j=0; j<osize[1]; j++){
        for (int k=0; k<osize[2]; k++){
          X=(i+ostart[0]);
          Y=(j+ostart[1]);
          Z=(k+ostart[2]);

          wave=Y;

          if(Y>N[1]/2)
            wave-=N[1];

          ptr=(i*osize[1]+j)*osize[2]+k;
          wA[ptr][0] =-scale*wave*A[ptr][1];
          wA[ptr][1] = scale*wave*A[ptr][0];
        }
      }
    }
  }

  return;
}

static void grad_mult_wave_nunmberz(Complex* wA, Complex* A, int* N,MPI_Comm c_comm,std::bitset<3> xyz ){

	int procid;
	MPI_Comm_rank(c_comm,&procid);
	double scale=1;
  if(xyz[0]) scale*=N[0];
  if(xyz[1]) scale*=N[1];
  if(xyz[2]) scale*=N[2];
  //PCOUT<<scale<<std::endl;
  scale=1./scale;

  int istart[3], isize[3], osize[3],ostart[3];
  accfft_local_size_dft_r2c(N,isize,istart,osize,ostart,c_comm);
  //PCOUT<<osize[0]<<'\t'<<osize[1]<<'\t'<<osize[2]<<std::endl;

#pragma omp parallel
  {
    double X,Y,Z,wave;
    long int ptr;
#pragma omp for
    for (int i=0; i<osize[0]; i++){
      for (int j=0; j<osize[1]; j++){
        for (int k=0; k<osize[2]; k++){
          X=(i+ostart[0]);
          Y=(j+ostart[1]);
          Z=(k+ostart[2]);

          wave=Z;

          if(Z>N[2]/2)
            wave-=N[2];

          ptr=(i*osize[1]+j)*osize[2]+k;
          wA[ptr][0] =-scale*wave*A[ptr][1];
          wA[ptr][1] = scale*wave*A[ptr][0];
        }
      }
    }
  }

  return;
}

static void grad_mult_wave_nunmber_laplace(Complex* wA, Complex* A, int* N,MPI_Comm c_comm ){

	int procid;
	MPI_Comm_rank(c_comm,&procid);
	const double scale=1./(N[0]*N[1]*N[2]);

  int istart[3], isize[3], osize[3],ostart[3];
  accfft_local_size_dft_r2c(N,isize,istart,osize,ostart,c_comm);

#pragma omp parallel
  {
    double X,Y,Z,wx,wy,wz,wave;
    long int ptr;
#pragma omp for
    for (int i=0; i<osize[0]; i++){
      for (int j=0; j<osize[1]; j++){
        for (int k=0; k<osize[2]; k++){
          X=(i+ostart[0]);
          Y=(j+ostart[1]);
          Z=(k+ostart[2]);

          wx=X;
          wy=Y;
          wz=Z;

          if(X>N[0]/2)
            wx-=N[0];
          if(Y>N[1]/2)
            wy-=N[1];
          if(Z>N[2]/2)
            wz-=N[2];

          wave=-wx*wx-wy*wy-wz*wz;

          ptr=(i*osize[1]+j)*osize[2]+k;
          wA[ptr][0] = scale*wave*A[ptr][0];
          wA[ptr][1] = scale*wave*A[ptr][1];
        }
      }
    }
  }

  return;
}

void accfft_grad(double * A_x, double *A_y, double *A_z,double *A,accfft_plan *plan, std::bitset<3> XYZ, double* timer){
	int procid;
  MPI_Comm c_comm=plan->c_comm;
  MPI_Comm_rank(c_comm,&procid);

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
  alloc_max=accfft_local_size_dft_r2c(N,isize,istart,osize,ostart,c_comm);

  Complex * A_hat=(Complex*) accfft_alloc(alloc_max);
  Complex * tmp  =(Complex*) accfft_alloc(alloc_max);
  std::bitset<3> scale_xyz={0};
  scale_xyz[0]=1;
  scale_xyz[1]=1;
  scale_xyz[2]=1;

	MPI_Barrier(c_comm);

	/* Forward transform */
  accfft_execute_r2c(plan,A,A_hat,timings);

	/* Multiply x Wave Numbers */
	if(XYZ[0]){
		grad_mult_wave_nunmberx(tmp,A_hat, N,c_comm,scale_xyz);
		MPI_Barrier(c_comm);

		/* Backward transform */
    accfft_execute_c2r(plan,tmp,A_x,timings);
	}
	/* Multiply y Wave Numbers */
	if(XYZ[1]){
		grad_mult_wave_nunmbery(tmp,A_hat, N,c_comm,scale_xyz);
		/* Backward transform */
    accfft_execute_c2r(plan,tmp,A_y,timings);
	}

	/* Multiply z Wave Numbers */
	if(XYZ[2]){
		grad_mult_wave_nunmberz(tmp,A_hat, N,c_comm,scale_xyz);
		/* Backward transform */
    accfft_execute_c2r(plan,tmp,A_z,timings);
	}

	accfft_free(A_hat);
	accfft_free(tmp);

	self_exec_time+= MPI_Wtime();

	timings[0]=self_exec_time;

  if(timer==NULL){
    delete [] timings;
  }
	return;
}

void accfft_laplace(double * LA,double *A,accfft_plan *plan, double* timer){
	int procid;
  MPI_Comm c_comm=plan->c_comm;
  MPI_Comm_rank(c_comm,&procid);

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
  alloc_max=accfft_local_size_dft_r2c(N,isize,istart,osize,ostart,c_comm);

  Complex * A_hat=(Complex*) accfft_alloc(alloc_max);
  Complex * tmp  =(Complex*) accfft_alloc(alloc_max);

	MPI_Barrier(c_comm);

	/* Forward transform */
  accfft_execute_r2c(plan,A,A_hat,timings);

  /* Multiply x Wave Numbers */
  grad_mult_wave_nunmber_laplace(tmp,A_hat, N,c_comm);
  MPI_Barrier(c_comm);

  /* Backward transform */
  accfft_execute_c2r(plan,tmp,LA,timings);

  accfft_free(A_hat);
  accfft_free(tmp);

  self_exec_time+= MPI_Wtime();

  timings[0]=self_exec_time;

  if(timer==NULL){
    delete [] timings;
  }
  return;
}

static void grad_mult_wave_nunmberx(Complex* wA, Complex* A, int* N,MPI_Comm c_comm,std::bitset<3> xyz ){

	int procid;
	MPI_Comm_rank(c_comm,&procid);
	double scale=1;
  if(xyz[0]) scale*=N[0];
  if(xyz[1]) scale*=N[1];
  if(xyz[2]) scale*=N[2];
  //PCOUT<<"xyz= "<<xyz<<" scale= "<< scale<<std::endl;
  scale=1./scale;

  int istart[3], isize[3], osize[3],ostart[3];
  accfft_local_size_dft_r2c(N,isize,istart,osize,ostart,c_comm);

#pragma omp parallel
  {
    double X,Y,Z,wave;
    long int ptr;
#pragma omp for
    for (int i=0; i<osize[0]; i++){
      for (int j=0; j<osize[1]; j++){
        for (int k=0; k<osize[2]; k++){
          X=(i+ostart[0]);
          Y=(j+ostart[1]);
          Z=(k+ostart[2]);

          wave=X;

          if(X>N[0]/2)
            wave-=N[0];

          ptr=(i*osize[1]+j)*osize[2]+k;
          wA[ptr][0] =-scale*wave*A[ptr][1];
          wA[ptr][1] = scale*wave*A[ptr][0];
        }
      }
    }
  }

  return;
}

void accfft_divergence(double* div_A, double * A_x, double *A_y, double *A_z,accfft_plan *plan, double* timer){
	int procid;
  MPI_Comm c_comm=plan->c_comm;
  MPI_Comm_rank(c_comm,&procid);

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
  alloc_max=accfft_local_size_dft_r2c(N,isize,istart,osize,ostart,c_comm);

  Complex * A_hat=(Complex*) accfft_alloc(alloc_max);
  Complex * tmp  =(Complex*) accfft_alloc(alloc_max);
  double  * tmp2 =(double*)  accfft_alloc(alloc_max);
  std::bitset<3> xyz={0};

  MPI_Barrier(c_comm);


  /* Forward transform in x direction*/
  xyz[0]=1;
  xyz[1]=0;
  xyz[2]=1;
  accfft_execute_r2c(plan,A_x,A_hat,timings,xyz);
  /* Multiply x Wave Numbers */
  grad_mult_wave_nunmberx(tmp,A_hat, N,c_comm,xyz);
  MPI_Barrier(c_comm);
  /* Backward transform */
  accfft_execute_c2r(plan,tmp,tmp2,timings,xyz);

  memcpy(div_A,tmp2,isize[0]*isize[1]*isize[2]*sizeof(double));

  /* Forward transform in y direction*/
  xyz[0]=1;
  xyz[1]=1;
  xyz[2]=1;
  accfft_execute_r2c(plan,A_y,A_hat,timings,xyz);
  /* Multiply y Wave Numbers */
  grad_mult_wave_nunmbery(tmp,A_hat, N,c_comm,xyz);
  MPI_Barrier(c_comm);
  /* Backward transform */
  accfft_execute_c2r(plan,tmp,tmp2,timings,xyz);

  for (int i=0;i<isize[0]*isize[1]*isize[2];++i)
    div_A[i]+=tmp2[i];


  /* Forward transform in z direction*/
  xyz[0]=1;
  xyz[1]=1;
  xyz[2]=1;
  accfft_execute_r2c(plan,A_z,A_hat,timings,xyz);
  /* Multiply z Wave Numbers */
  grad_mult_wave_nunmberz(tmp,A_hat, N,c_comm,xyz);
  MPI_Barrier(c_comm);
  /* Backward transform */
  accfft_execute_c2r(plan,tmp,tmp2,timings,xyz);

  for (int i=0;i<isize[0]*isize[1]*isize[2];++i)
    div_A[i]+=tmp2[i];

  accfft_free(A_hat);
  accfft_free(tmp);
  accfft_free(tmp2);

  self_exec_time+= MPI_Wtime();

  timings[0]=self_exec_time;

  if(timer==NULL){
    delete [] timings;
  }
  return;
}



