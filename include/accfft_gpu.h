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
#include <cstddef>
#include <bitset>
#include "cuda.h"
#include <cufft.h>
#include "accfft_common.h"

template <typename real>
class AccFFT_gpu {
        typedef real cplx[2];

	int alloc_max;
	T_Plan_gpu<real> * T_plan_1;
	T_Plan_gpu<real> * T_plan_2;
	T_Plan_gpu<real> * T_plan_2i;
	T_Plan_gpu<real> * T_plan_1i;
	T_Plan_gpu<real> * T_plan_y;
	T_Plan_gpu<real> * T_plan_yi;
	T_Plan_gpu<real> * T_plan_x;
	T_Plan_gpu<real> * T_plan_xi;
	cufftHandle fplan_0, iplan_0, fplan_1, iplan_1, fplan_2, iplan_2;
  cufftHandle fplan_y, fplan_x, iplan_y, iplan_x;
	int coord[2], np[2], periods[2];
	MPI_Comm row_comm, col_comm;

	real * data;
	cplx * data_out;
	cplx * data_c;
	cplx * data_out_c;
	int procid;
	bool inplace;
	bool oneD;

        cufftType cu_r2c, cu_c2r, cu_c2c; // CUFFT_R2C || CUFFT_D2Z etc.

    /* Continuation of initialization sequence after setting type-specific
     * variables:
     */
    void init_c2c(int *n, cplx *data_d,
        cplx * data_out_d, MPI_Comm ic_comm, unsigned flags);
    void init_r2c(int *n, real *data_d,
        cplx* data_out_d, MPI_Comm ic_comm, unsigned flags);

  public:
		/* FIXME: these are required by operators.cpp for now */
    int N[3];
    Mem_Mgr_gpu<real> *Mem_mgr;
    MPI_Comm c_comm;
    
    int osize_0[3], ostart_0[3];
    int osize_1[3], ostart_1[3];
    int osize_2[3], ostart_2[3];
    int osize_1i[3], ostart_1i[3];
    int osize_2i[3], ostart_2i[3];
    int osize_y[3], osize_yi[3], ostart_y[3];
    int osize_x[3], osize_xi[3], ostart_x[3];

    int isize[3], istart[3];
    int osize[3], ostart[3];

    bool r2c_plan_baked;
    bool c2c_plan_baked;
    /* FIXME: operators should be implemented with better defined scope */

    AccFFT_gpu(int *n, real *data_d,
        cplx* data_out_d, MPI_Comm ic_comm, unsigned flags = ACCFFT_MEASURE);
    AccFFT_gpu(int *n, cplx *data_d,
        cplx* data_out_d, MPI_Comm ic_comm, unsigned flags = ACCFFT_MEASURE);

    ~AccFFT_gpu();

    void execute(int direction, real *data_d = NULL, real *out_d = NULL,
        double * timer = NULL, std::bitset<3> xyz = 111);
    void execute_r2c(real *data = NULL, cplx *out = NULL,
        double *timer = NULL, std::bitset<3> xyz = 111);
    void execute_c2r(cplx *data = NULL, real *out = NULL,
        double *timer = NULL, std::bitset<3> xyz = 111);
    void execute_c2c(int direction, cplx *data_d = NULL, cplx *out_d = NULL,
        double * timer = NULL, std::bitset<3> xyz = 111);
    void execute_z(int dir, real *data, real *data_out, double *timer = NULL);
    void execute_y(int dir, real *data, real *data_out, double *timer = NULL);
    void execute_x(int dir, real *data, real *data_out, double *timer = NULL);

    void execute_r2c_z(real *data, cplx *data_out, double *timer = NULL);
    void execute_c2r_z(cplx *data, real *data_out, double *timer = NULL);
    void execute_r2c_y(real *data, cplx *data_out, double *timer = NULL);
    void execute_c2r_y(cplx *data, real *data_out, double *timer = NULL);
    void execute_r2c_x(real *data, cplx *data_out, double *timer = NULL);
    void execute_c2r_x(cplx *data, real *data_out, double *timer = NULL);
};

void accfft_cleanup_gpu();

typedef class AccFFT_gpu<float> AccFFTs_gpu;
typedef class AccFFT_gpu<double> AccFFTd_gpu;

template class AccFFT_gpu<float>;
template class AccFFT_gpu<double>;

/*template <typename T>
    int dfft_get_local_size_gpu(int N0, int N1, int N2,
        int *isize, int *istart, MPI_Comm c_comm);*/
template <typename T>
    int accfft_local_size_dft_c2c_gpu(const int *n, int *isize, int *istart,
        int *osize, int *ostart, MPI_Comm c_comm);
template <typename T>
    int accfft_local_size_dft_r2c_gpu(const int *n, int *isize, int *istart,
        int *osize, int *ostart, MPI_Comm c_comm);

template <typename real> class cu_type;
template<> class cu_type<float> {
  public:
    typedef cufftReal real;
    typedef cufftComplex cplx;
};
template<> class cu_type<double> {
  public:
    typedef cufftDoubleReal real;
    typedef cufftDoubleComplex cplx;
};
#endif

#ifndef ACCFFT_CHECKCUDA_H
#define ACCFFT_CHECKCUDA_H
inline cudaError_t checkCuda_accfft(cudaError_t result) {
#if defined(DEBUG) || defined(_DEBUG)
	if (result != cudaSuccess) {
		fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
		assert(result == cudaSuccess);
	}
#endif
	return result;
}
inline cufftResult checkCuda_accfft(cufftResult result) {
#if defined(DEBUG) || defined(_DEBUG)
	if (result != CUFFT_SUCCESS) {
		fprintf(stderr, "CUDA Runtime Error: %s\n", result);
		assert(result == CUFFT_SUCCESS);
	}
#endif
	return result;
}
#endif
