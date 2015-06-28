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

#ifndef ACCFFT_GPU_H
#define ACCFFT_GPU_H
#include <mpi.h>
#include <omp.h>
#include <iostream>
#include <cmath>
#include <math.h>
#include "transpose_cuda.h"
#include <string.h>
#include "cuda.h"
#include <cufft.h>
#include "accfft_common.h"

inline cudaError_t checkCuda_accfft(cudaError_t result)
{
#if defined(DEBUG) || defined(_DEBUG)
  if (result != cudaSuccess) {
    fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
    assert(result == cudaSuccess);
  }
#endif
  return result;
}
inline cufftResult checkCuda_accfft(cufftResult result)
{
#if defined(DEBUG) || defined(_DEBUG)
  if (result != CUFFT_SUCCESS) {
    fprintf(stderr, "CUDA Runtime Error: %s\n", result);
    assert(result == CUFFT_SUCCESS);
  }
#endif
  return result;
}

struct accfft_plan_gpu{
  int N[3];
  int alloc_max;
  Mem_Mgr_gpu * Mem_mgr;
  T_Plan_gpu  * T_plan_1;
  T_Plan_gpu  * T_plan_2;
  T_Plan_gpu  * T_plan_2i;
  T_Plan_gpu  * T_plan_1i;
  cufftHandle fplan_0, iplan_0,fplan_1,iplan_1, fplan_2, iplan_2;
  int coord[2],np[2],periods[2];
  MPI_Comm c_comm,row_comm,col_comm;

  int osize_0[3],  ostart_0[3];
  int osize_1[3],  ostart_1[3];
  int osize_2[3],  ostart_2[3];
  int osize_1i[3], ostart_1i[3];
  int osize_2i[3], ostart_2i[3];

  double * data;
  double * data_out;
  Complex * data_c;
  Complex * data_out_c;
  int procid;
  bool inplace;
};

int dfft_get_local_size_gpu(int N0, int N1, int N2, int * isize, int * istart,MPI_Comm c_comm );
int accfft_local_size_dft_r2c_gpu( int * n,int * isize, int * istart, int * osize, int *ostart,MPI_Comm c_comm, bool inplace=0);

accfft_plan_gpu*  accfft_plan_dft_3d_r2c_gpu(int * n, double * data_d, double * data_out_d, MPI_Comm c_comm,unsigned flags=ACCFFT_MEASURE);

void accfft_execute_gpu(accfft_plan_gpu* plan, int direction,double * data_d=NULL, double * data_out_d=NULL, double * timer=NULL);

int accfft_local_size_dft_c2c_gpu( int * n,int * isize, int * istart, int * osize, int *ostart,MPI_Comm c_comm);

accfft_plan_gpu*  accfft_plan_dft_3d_c2c_gpu(int * n, Complex * data_d, Complex * data_out_d, MPI_Comm c_comm,unsigned flags=ACCFFT_MEASURE);

void accfft_execute_c2c_gpu(accfft_plan_gpu* plan, int direction,Complex * data_d=NULL, Complex * data_out_d=NULL, double * timer=NULL);
void accfft_destroy_plan(accfft_plan_gpu * plan);
void accfft_destroy_plan_gpu(accfft_plan_gpu * plan);
void accfft_execute_r2c_gpu(accfft_plan_gpu* plan, double * data=NULL,Complex * data_out=NULL, double * timer=NULL);
void accfft_execute_c2r_gpu(accfft_plan_gpu* plan, Complex * data=NULL,double * data_out=NULL, double * timer=NULL);
void accfft_cleanup_gpu();
#endif
