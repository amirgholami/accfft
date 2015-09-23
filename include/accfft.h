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

#ifndef ACCFFT_H
#define ACCFFT_H
#include <mpi.h>
#include <fftw3.h>
#include <omp.h>
#include <iostream>
#include <cmath>
#include <math.h>
#include "transpose.h"
#include <string.h>
#include <string>
#include <bitset>
#include "accfft_common.h"

struct accfft_plan{
  int N[3];
  int alloc_max;
  Mem_Mgr * Mem_mgr;
  T_Plan  * T_plan_1;
  T_Plan  * T_plan_2;
  T_Plan  * T_plan_2i;
  T_Plan  * T_plan_1i;
  fftw_plan fplan_0, iplan_0,fplan_1,iplan_1, fplan_2, iplan_2;
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

int accfft_init(int nthreads);
int dfft_get_local_size(int N0, int N1, int N2, int * isize, int * istart,MPI_Comm c_comm );
int accfft_local_size_dft_r2c( int * n,int * isize, int * istart, int * osize, int *ostart,MPI_Comm c_comm);

accfft_plan*  accfft_plan_dft_3d_r2c(int * n, double * data, double * data_out, MPI_Comm c_comm,unsigned flags=ACCFFT_MEASURE);

int accfft_local_size_dft_c2c( int * n,int * isize, int * istart, int * osize, int *ostart,MPI_Comm c_comm);
accfft_plan*  accfft_plan_dft_3d_c2c(int * n, Complex * data, Complex * data_out, MPI_Comm c_comm,unsigned flags=ACCFFT_MEASURE);

void accfft_execute_r2c(accfft_plan* plan, double * data=NULL,Complex * data_out=NULL, double * timer=NULL,std::bitset<3> xyz=1);
void accfft_execute_c2r(accfft_plan* plan, Complex * data=NULL,double * data_out=NULL, double * timer=NULL,std::bitset<3> xyz=1);
void accfft_execute(accfft_plan* plan, int direction, double * data=NULL,double * data_out=NULL,double * timer=NULL,std::bitset<3> xyz=1);
void accfft_execute_c2c(accfft_plan* plan, int direction, Complex * data=NULL, Complex * data_out=NULL,double * timer=NULL);
void accfft_destroy_plan(accfft_plan * plan);
void accfft_cleanup();
#endif
