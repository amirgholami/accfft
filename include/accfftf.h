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

#ifndef ACCFFTF_H
#define ACCFFTF_H
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

struct accfft_planf{
  int N[3];
  int alloc_max;
   Mem_Mgr<float> * Mem_mgr;
   T_Plan <float>* T_plan_1;
   T_Plan <float>* T_plan_2;
   T_Plan <float>* T_plan_2i;
   T_Plan <float>* T_plan_1i;
  fftwf_plan fplan_0, iplan_0,fplan_1,iplan_1, fplan_2, iplan_2;
  int coord[2],np[2],periods[2];
  MPI_Comm c_comm,row_comm,col_comm;

  int osize_0[3],  ostart_0[3];
  int osize_1[3],  ostart_1[3];
  int osize_2[3],  ostart_2[3];
  int osize_1i[3], ostart_1i[3];
  int osize_2i[3], ostart_2i[3];

  int isize[3],istart[3];
  int osize[3],ostart[3];


  float * data;
  float * data_out;
  Complexf * data_c;
  Complexf * data_out_c;
  int procid;
  bool oneD;
  bool inplace;
  bool r2c_plan_baked;
  bool c2c_plan_baked;

  accfft_planf(){
    r2c_plan_baked=0;
    c2c_plan_baked=0;
    data=NULL;
    data_out=NULL;
    T_plan_1=NULL;
    T_plan_1i=NULL;
    T_plan_2=NULL;
    T_plan_2i=NULL;
    Mem_mgr=NULL;
  };


};

int accfft_init(int nthreads);
int dfft_get_local_size(int N0, int N1, int N2, int * isize, int * istart,MPI_Comm c_comm );


int accfft_local_size_dft_c2cf( int * n,int * isize, int * istart, int * osize, int *ostart,MPI_Comm c_comm);
accfft_planf*  accfft_plan_dft_3d_c2cf(int * n, Complexf * data, Complexf * data_out, MPI_Comm c_comm,unsigned flags=ACCFFT_MEASURE);

void accfft_executef(accfft_planf* plan, int direction, float * data=NULL, float* data_out=NULL,double * timer=NULL,std::bitset<3> xyz=111);
void accfft_execute_c2cf(accfft_planf* plan, int direction, Complexf * data=NULL, Complexf * data_out=NULL,double * timer=NULL,std::bitset<3> xyz=111);
void accfft_destroy_plan(accfft_planf * plan);
void accfft_cleanup();


int accfft_local_size_dft_r2cf( int * n,int * isize, int * istart, int * osize, int *ostart,MPI_Comm c_comm);

accfft_planf*  accfft_plan_dft_3d_r2cf(int * n, float * data, float * data_out, MPI_Comm c_comm,unsigned flags=ACCFFT_MEASURE);
void accfft_execute_r2cf(accfft_planf* plan, float * data=NULL,Complexf * data_out=NULL, double * timer=NULL,std::bitset<3> xyz=111);
void accfft_execute_c2rf(accfft_planf* plan, Complexf * data=NULL,float * data_out=NULL, double * timer=NULL,std::bitset<3> xyz=111);

template <typename T,typename Tc>
void accfft_execute_r2c_t(accfft_planf* plan, T* data,Tc* data_out, double * timer=NULL,std::bitset<3> XYZ=111);

template <typename Tc, typename T>
void accfft_execute_c2r_t(accfft_planf* plan, Tc* data, T* data_out, double * timer=NULL,std::bitset<3> XYZ=111);

template <typename T>
int accfft_local_size_dft_r2c_t( int * n,int * isize, int * istart, int * osize, int *ostart,MPI_Comm c_comm);


#endif
