/**
 * @file
 * Single precision GPU functions of AccFFT
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

#include "accfft_gpu.h"
#include <mpi.h>
#include <omp.h>
#include <iostream>
#include <cmath>
#include <math.h>
#include "transpose_cuda.h"
#include <cuda_runtime_api.h>
#include <string.h>
#include <cuda.h>
#include <cufft.h>
#include "accfft_common.h"
#include "dtypes.h"
#define VERBOSE 0

/**
 * Cleanup all GPU resources
 */
void accfft_cleanup_gpu() {
	// empty for now
}

/**
 * Get the local sizes of the distributed global data for a GPU  R2C transform
 * @param n Integer array of size 3, corresponding to the global data size
 * @param isize The size of the data that is locally distributed to the calling process
 * @param istart The starting index of the data that locally resides on the calling process
 * @param osize The output size of the data that locally resides on the calling process,
 * after the R2C transform is finished
 * @param ostart The output starting index of the data that locally resides on the calling process,
 * after the R2C transform is finished
 * @param c_comm Cartesian communicator returned by \ref accfft_create_comm
 * @return
 */
template <typename T>
int accfft_local_size_dft_r2c_gpu(const int *n, int *isize, int *istart,
		int *osize, int *ostart, MPI_Comm c_comm) {
  return accfft_local_size_dft_r2c<T>(n, isize, istart, osize, ostart, c_comm);
}
#define R2C_SIZE(real) \
  template int accfft_local_size_dft_r2c_gpu<real>( \
          const int *n, int *isize, int *istart, \
          int *osize, int *ostart, MPI_Comm c_comm);
TPL_DECL(R2C_SIZE)

/**
 * Creates a 3D  R2C parallel FFT plan. If data_out point to the same location as the input
 * data, then an inplace plan will be created. Otherwise the plan would be outplace.
 * @param n Integer array of size 3, corresponding to the global data size
 * @param data Input data in spatial domain
 * @param data_out Output data in frequency domain
 * @param c_comm Cartesian communicator returned by \ref accfft_create_comm
 * @param flags AccFFT flags, See \ref flags for more details.
 * @return
 */

template<>
AccFFT_gpu<float> :: AccFFT_gpu
(int *n, float *data_d, Complexf *data_out_d,
		MPI_Comm c_comm, unsigned flags) {
  cu_r2c = CUFFT_R2C;
  cu_c2r = CUFFT_C2R;
  cu_c2c = CUFFT_C2C;
  init_r2c(n, data_d, data_out_d, c_comm, flags);
}
template<>
AccFFT_gpu<float> :: AccFFT_gpu
(int *n, Complexf *data_d, Complexf *data_out_d,
		MPI_Comm c_comm, unsigned flags) {
  cu_r2c = CUFFT_R2C;
  cu_c2r = CUFFT_C2R;
  cu_c2c = CUFFT_C2C;
  init_c2c(n, data_d, data_out_d, c_comm, flags);
}
template<>
AccFFT_gpu<double> :: AccFFT_gpu
(int *n, double *data_d, Complex *data_out_d,
		MPI_Comm c_comm, unsigned flags) {
  cu_r2c = CUFFT_D2Z;
  cu_c2r = CUFFT_Z2D;
  cu_c2c = CUFFT_Z2Z;
  init_r2c(n, data_d, data_out_d, c_comm, flags);
}
template<>
AccFFT_gpu<double> :: AccFFT_gpu
(int *n, Complex *data_d, Complex *data_out_d,
		MPI_Comm c_comm, unsigned flags) {
  cu_r2c = CUFFT_D2Z;
  cu_c2r = CUFFT_Z2D;
  cu_c2c = CUFFT_Z2Z;
  init_c2c(n, data_d, data_out_d, c_comm, flags);
}

static inline cufftResult exec_c2r(cufftHandle plan,
        cufftComplex *data, cufftReal *out) {
    return cufftExecC2R(plan, data, out);
}
static inline cufftResult exec_c2r(cufftHandle plan,
        cufftDoubleComplex *data, cufftDoubleReal *out) {
    return cufftExecZ2D(plan, data, out);
}
static inline cufftResult exec_r2c(cufftHandle plan,
        cufftReal *data, cufftComplex *out) {
    return cufftExecR2C(plan, data, out);
}
static inline cufftResult exec_r2c(cufftHandle plan,
        cufftDoubleReal *data, cufftDoubleComplex *out) {
    return cufftExecD2Z(plan, data, out);
}
static inline cufftResult exec_c2c(cufftHandle plan,
        cufftComplex *data, cufftComplex *out, int dir) {
    return cufftExecC2C(plan, data, out, dir);
}
static inline cufftResult exec_c2c(cufftHandle plan,
        cufftDoubleComplex *data, cufftDoubleComplex *out, int dir) {
    return cufftExecZ2Z(plan, data, out, dir);
}

GMETHOD(void, init_r2c)(int *n, real *data_d,
		cplx *data_out_d, MPI_Comm ic_comm, unsigned flags) {
  r2c_plan_baked = 0;
  c2c_plan_baked = 0;
  data = NULL;
  data_out = NULL;
  T_plan_1 = NULL;
  T_plan_1i = NULL;
  T_plan_2 = NULL;
  T_plan_2i = NULL;
  T_plan_y = NULL;
  T_plan_yi = NULL;
  T_plan_x = NULL;
  T_plan_xi = NULL;
  Mem_mgr = NULL;
  fplan_y = -1;
  fplan_x = -1;
  iplan_y = -1;
  iplan_x = -1;

  c_comm = ic_comm;
	MPI_Comm_rank(c_comm, &procid);
	MPI_Cart_get(c_comm, 2, np, periods, coord);
	MPI_Comm_split(c_comm, coord[0], coord[1], &row_comm);
	MPI_Comm_split(c_comm, coord[1], coord[0], &col_comm);
	N[0] = n[0];
	N[1] = n[1];
	N[2] = n[2];
	data = data_d;
	data_out = data_out_d;

	if (np[1] == 1)
		oneD = true;
	else
		oneD = false;

	if((real *)data_out_d == data_d) {
		inplace = true;
	} else {
		inplace = false;
	}

	int n_tuples_i, n_tuples_o;
	//inplace==true ? n_tuples=(n[2]/2+1)*2: n_tuples=n[2]*2;
	n_tuples_i = inplace == true ? (n[2] / 2 + 1) * 2 : n[2];
	n_tuples_o = (n[2] / 2 + 1) * 2;

	//int isize[3],osize[3],istart[3],ostart[3];
	alloc_max = accfft_local_size_dft_r2c_gpu<real>(n, isize, istart,
			osize, ostart, c_comm);

	dfft_get_local_size<real>(n[0], n[1], n_tuples_o / 2, osize_0, ostart_0, c_comm);
	dfft_get_local_size<real>(n[0], n_tuples_o / 2, n[1], osize_1, ostart_1,
			c_comm);
	dfft_get_local_size<real>(n[1], n_tuples_o / 2, n[0], osize_2, ostart_2,
			c_comm);

	std::swap(osize_1[1], osize_1[2]);
	std::swap(ostart_1[1], ostart_1[2]);

	std::swap(ostart_2[1], ostart_2[2]);
	std::swap(ostart_2[0], ostart_2[1]);
	std::swap(osize_2[1], osize_2[2]);
	std::swap(osize_2[0], osize_2[1]);

  // osize_y is the configuration after y transpose
	dfft_get_local_size<real>(n[0], n[2], n[1], osize_y, ostart_y, c_comm);
	std::swap(osize_y[1], osize_y[2]);
	std::swap(ostart_y[1], ostart_y[2]);
  osize_yi[0] = osize_y[0];
  osize_yi[1] = osize_y[1] / 2 + 1;
  osize_yi[2] = osize_y[2];

  // for x only fft. The strategy is to divide (N1/P1 x N2) by P0 completely.
  // So we treat the last size to be just 1.
	dfft_get_local_size<real>(isize[1] * n[2], n[2], n[0], osize_x, ostart_x, c_comm);
  osize_x[1] = osize_x[0];
  osize_x[0] = osize_x[2];
  osize_x[2] = 1;

	dfft_get_local_size<real>(isize[1] * n[2], n[2], n[0] / 2 + 1, osize_xi, ostart_x, c_comm);
  osize_xi[1] = osize_xi[0];
  osize_xi[0] = osize_xi[2];
  osize_xi[2] = 1;
  ostart_x[0] = 0;
  ostart_x[1] = -1<<8; // starts have no meaning in this approach
  ostart_x[2] = -1<<8;

	for (int i = 0; i < 3; i++) {
		osize_1i[i] = osize_1[i];
		osize_2i[i] = osize_2[i];
		ostart_1i[i] = ostart_1[i];
		ostart_2i[i] = ostart_2[i];
	}

	// fplan_0
	int NX = n[0], NY = n[1], NZ = n[2];
	cufftResult_t cufft_error;
	{
		int f_inembed[1] = { n_tuples_i };
		int f_onembed[1] = { n_tuples_o / 2 };
		int idist = (n_tuples_i);
		int odist = n_tuples_o / 2;
		int istride = 1;
		int ostride = 1;
		int batch = osize_0[0] * osize_0[1];  //NX;

		if (batch != 0) {
			cufft_error = cufftPlanMany(&fplan_0, 1, &n[2], f_inembed,
					istride, idist, // *inembed, istride, idist
					f_onembed, ostride, odist, // *onembed, ostride, odist
					cu_r2c, batch);
			if (cufft_error != CUFFT_SUCCESS) {
				fprintf(stderr, "CUFFT error: fplan_0 creation failed %d \n",
						cufft_error);
				throw std::runtime_error("CUFFT error");
			}
			//cufftSetCompatibilityMode(fplan,CUFFT_COMPATIBILITY_FFTW_PADDING); if (cudaGetLastError() != cudaSuccess){fprintf(stderr, "Cuda error:Failed at fplan cuda compatibility\n"); return;}
		}
		if (batch != 0) {
			cufft_error = cufftPlanMany(&iplan_0, 1, &n[2], f_onembed,
					ostride, odist, // *onembed, ostride, odist
					f_inembed, istride, idist, // *inembed, istride, idist
					cu_c2r, batch);
			if (cufft_error != CUFFT_SUCCESS) {
				fprintf(stderr, "CUFFT error: iplan_0 creation failed %d \n",
						cufft_error);
				throw std::runtime_error("CUFFT error");
			}
			//cufftSetCompatibilityMode(fplan,CUFFT_COMPATIBILITY_FFTW_PADDING); if (cudaGetLastError() != cudaSuccess){fprintf(stderr, "Cuda error:Failed at fplan cuda compatibility\n"); return;}
		}
	}
	// fplan_1
	{
		int f_inembed[1] = { NY };
		int f_onembed[1] = { NY };
		int idist = 1;
		int odist = 1;
		int istride = osize_1[2];
		int ostride = osize_1[2];
		int batch = osize_1[2];

		if (batch != 0) {
			cufft_error = cufftPlanMany(&fplan_1, 1, &n[1], f_inembed,
					istride, idist, // *inembed, istride, idist
					f_onembed, ostride, odist, // *onembed, ostride, odist
					cu_c2c, batch);
			if (cufft_error != CUFFT_SUCCESS) {
				fprintf(stderr, "CUFFT error: fplan_1 creation failed %d \n",
						cufft_error);
				throw std::runtime_error("CUFFT error");
			}
			//cufftSetCompatibilityMode(fplan,CUFFT_COMPATIBILITY_FFTW_PADDING); if (cudaGetLastError() != cudaSuccess){fprintf(stderr, "Cuda error:Failed at fplan cuda compatibility\n"); return;}
		}
	}
	// fplan_2
	{
		int f_inembed[1] = { NX };
		int f_onembed[1] = { NX };
		int idist = 1;
		int odist = 1;
		int istride = osize_2[1] * osize_2[2];
		int ostride = osize_2[1] * osize_2[2];
		int batch = osize_2[1] * osize_2[2];

		if (batch != 0) {
			cufft_error = cufftPlanMany(&fplan_2, 1, &n[0], f_inembed,
					istride, idist, // *inembed, istride, idist
					f_onembed, ostride, odist, // *onembed, ostride, odist
					cu_c2c, batch);
			if (cufft_error != CUFFT_SUCCESS) {
				fprintf(stderr, "CUFFT error: fplan_2 creation failed %d \n",
						cufft_error);
				throw std::runtime_error("CUFFT error");
			}
			//cufftSetCompatibilityMode(fplan,CUFFT_COMPATIBILITY_FFTW_PADDING); if (cudaGetLastError() != cudaSuccess){fprintf(stderr, "Cuda error:Failed at fplan cuda compatibility\n"); return;}
		}
	}
	// fplan_y
	{
		int f_inembed[1] = { NY };
		int f_onembed[1] = { NY / 2 + 1 };
		int idist = 1;
		int odist = 1;
		int istride = osize_y[2];
		int ostride = osize_y[2];
		int batch = osize_y[2];

		if (batch != 0) {
			cufft_error = cufftPlanMany(&fplan_y, 1, &n[1], f_inembed,
					istride, idist, // *inembed, istride, idist
					f_onembed, ostride, odist, // *onembed, ostride, odist
					cu_r2c, batch);
			if (cufft_error != CUFFT_SUCCESS) {
				fprintf(stderr, "CUFFT error: fplan_y creation failed %d \n",
						cufft_error);
				throw std::runtime_error("CUFFT error");
			}
			//cufftSetCompatibilityMode(fplan,CUFFT_COMPATIBILITY_FFTW_PADDING); if (cudaGetLastError() != cudaSuccess){fprintf(stderr, "Cuda error:Failed at fplan cuda compatibility\n"); return;}
		}
		if (batch != 0) {
			cufft_error = cufftPlanMany(&iplan_y, 1, &n[1], f_onembed,
					ostride, odist, // *inembed, istride, idist
					f_inembed, istride, idist, // *onembed, ostride, odist
					cu_c2r, batch);
			if (cufft_error != CUFFT_SUCCESS) {
				fprintf(stderr, "CUFFT error: iplan_y creation failed %d \n",
						cufft_error);
				throw std::runtime_error("CUFFT error");
			}
			//cufftSetCompatibilityMode(fplan,CUFFT_COMPATIBILITY_FFTW_PADDING); if (cudaGetLastError() != cudaSuccess){fprintf(stderr, "Cuda error:Failed at fplan cuda compatibility\n"); return;}
		}
	}
  // fplan_x
	{
		int f_inembed[1] = { n[0] };
		int f_onembed[1] = { n[0] / 2 + 1 };
		int idist = 1;
		int odist = 1;
		int istride = osize_x[1] * osize_x[2];
		int ostride = osize_xi[1] * osize_xi[2];
		int batch = osize_x[1] * osize_x[2];

		if (batch != 0) {
			cufft_error = cufftPlanMany(&fplan_x, 1, &n[0], f_inembed,
					istride, idist, // *inembed, istride, idist
					f_onembed, ostride, odist, // *onembed, ostride, odist
					cu_r2c, batch);
			if (cufft_error != CUFFT_SUCCESS) {
				fprintf(stderr, "CUFFT error: fplan_x creation failed %d \n",
						cufft_error);
				throw std::runtime_error("CUFFT error");
			}
			//cufftSetCompatibilityMode(fplan,CUFFT_COMPATIBILITY_FFTW_PADDING); if (cudaGetLastError() != cudaSuccess){fprintf(stderr, "Cuda error:Failed at fplan cuda compatibility\n"); return;}
		}
		if (batch != 0) {
			cufft_error = cufftPlanMany(&iplan_x, 1, &n[0], f_onembed,
					ostride, odist, // *onembed, ostride, odist
					f_inembed, istride, idist, // *inembed, istride, idist
					cu_c2r, batch);
			if (cufft_error != CUFFT_SUCCESS) {
				fprintf(stderr, "CUFFT error: iplan_x creation failed %d \n",
						cufft_error);
				throw std::runtime_error("CUFFT error");
			}
			//cufftSetCompatibilityMode(fplan,CUFFT_COMPATIBILITY_FFTW_PADDING); if (cudaGetLastError() != cudaSuccess){fprintf(stderr, "Cuda error:Failed at fplan cuda compatibility\n"); return;}
		}
	}

	// 1D Decomposition
	if (oneD) {
		int N0 = n[0], N1 = n[1], N2 = n[2];

		Mem_mgr = new Mem_Mgr_gpu<real>(N0, N1, n_tuples_o, c_comm, 1, alloc_max);
		T_plan_2 = new T_Plan_gpu<real>(N0, N1, n_tuples_o, Mem_mgr, c_comm);
		T_plan_2i = new T_Plan_gpu<real>(N1, N0, n_tuples_o, Mem_mgr, c_comm);
		T_plan_1 = NULL;
		T_plan_1i = NULL;

		alloc_max = alloc_max;
		T_plan_2->alloc_local = alloc_max;
		T_plan_2i->alloc_local = alloc_max;

		if (flags == ACCFFT_MEASURE) {
			T_plan_2->which_fast_method_gpu(T_plan_2, (real *)data_out_d);
		} else {
			T_plan_2->method = 2;
			T_plan_2->kway = 2;
		}
		checkCuda_accfft(cudaDeviceSynchronize());

		T_plan_2i->method = -T_plan_2->method;
		T_plan_2i->kway = T_plan_2->kway;
		T_plan_2i->kway_async = T_plan_2->kway_async;
	}

	// 2D Decomposition
	if (!oneD) {
		// the reaseon for n_tuples/2 is to avoid splitting of imag and real parts of complex numbers
		Mem_mgr = new Mem_Mgr_gpu<real>(n[1], n_tuples_o / 2, 2,
				row_comm, osize_0[0], alloc_max);
		T_plan_1 = new T_Plan_gpu<real>(n[1], n_tuples_o / 2, 2,
				Mem_mgr, row_comm, osize_0[0]);
		T_plan_2 = new T_Plan_gpu<real>(n[0], n[1], osize_2[2] * 2,
				Mem_mgr, col_comm);
		T_plan_2i = new T_Plan_gpu<real>(n[1], n[0], osize_2i[2] * 2,
				Mem_mgr, col_comm);
		T_plan_1i = new T_Plan_gpu<real>(n_tuples_o / 2, n[1], 2,
				Mem_mgr, row_comm, osize_1i[0]);
    // to transpose N0/P0 x N1/P1 x N2 -> N0/P0 x N1 x N2/P1
		T_plan_y = new T_Plan_gpu<real>(n[1], n[2], 1,
				Mem_mgr, row_comm, isize[0]);
    // to transpose N0/P0 x N1 x N2/P1 -> N0/P0 x N1/P1 x N2
		T_plan_yi = new T_Plan_gpu<real>(n[2], n[1], 1,
				Mem_mgr, row_comm, isize[0]);


		T_plan_1->alloc_local = alloc_max;
		T_plan_2->alloc_local = alloc_max;
		T_plan_2i->alloc_local = alloc_max;
		T_plan_1i->alloc_local = alloc_max;
		T_plan_y->alloc_local = alloc_max;
		T_plan_yi->alloc_local = alloc_max;

		if (flags == ACCFFT_MEASURE) {
			if (coord[0] == 0) {
				T_plan_1->which_fast_method_gpu(T_plan_1,
						(real *)data_out_d, osize_0[0]);
			}
		} else {
			T_plan_1->method = 2;
			T_plan_1->kway = 2;
		}

		MPI_Bcast(&T_plan_1->method, 1, MPI_INT, 0, c_comm);
		MPI_Bcast(&T_plan_1->kway, 1, MPI_INT, 0, c_comm);
		MPI_Bcast(&T_plan_1->kway_async, 1, par::Mpi_datatype<bool>::value(), 0, c_comm);

		checkCuda_accfft(cudaDeviceSynchronize());
		T_plan_2->method = T_plan_1->method;
		T_plan_2i->method = -T_plan_1->method;
		T_plan_1i->method = -T_plan_1->method;
		T_plan_y->method = T_plan_1->method;
		T_plan_yi->method = -T_plan_1->method;

		T_plan_2->kway = T_plan_1->kway;
		T_plan_2i->kway = T_plan_1->kway;
		T_plan_1i->kway = T_plan_1->kway;
		T_plan_y->kway = T_plan_1->kway;
		T_plan_yi->kway = T_plan_1->kway;

		T_plan_2->kway_async = T_plan_1->kway_async;
		T_plan_2i->kway_async = T_plan_1->kway_async;
		T_plan_1i->kway_async = T_plan_1->kway_async;
		T_plan_y->kway_async = T_plan_1->kway_async;
		T_plan_yi->kway_async = T_plan_1->kway_async;

		iplan_1 = -1;
		iplan_2 = -1;

	} // end 2d r2c

  // T_plan_x has to be created for both oneD and !oneD
  // Note that the T method is set via T_plan_2 for oneD case
  // to transpose N0/P0 x N1/P1 x N2 -> N0 x (N1/P1 x N2)/P0
	T_plan_x = new T_Plan_gpu<real>(n[0], isize[1] * isize[2], 1,
			Mem_mgr, col_comm, 1);
  // to transpose N0 x (N1/P1 x N2)/P0 -> N0/P0 x N1/P1 x N2
	T_plan_xi = new T_Plan_gpu<real>(isize[1] * isize[2], n[0], 1,
			Mem_mgr, col_comm, 1);
	T_plan_x->alloc_local = alloc_max;
	T_plan_xi->alloc_local = alloc_max;
	T_plan_x->method = T_plan_2->method;
	T_plan_xi->method = -T_plan_2->method;
	T_plan_x->kway = T_plan_2->kway;
	T_plan_xi->kway = T_plan_2->kway;
	T_plan_x->kway_async = T_plan_2->kway_async;
	T_plan_xi->kway_async = T_plan_2->kway_async;
	r2c_plan_baked = true;
} //end accfft_plan_dft_3d_r2c_gpu

GMETHOD(void, execute)(int direction, real *data_d,
		real *data_out_d, double *timer, std::bitset<3> xyz) {
	if (data_d == NULL)
		data_d = data;
	if (data_out_d == NULL)
		data_out_d = (real *)data_out;

	int * coord = coord;
	double fft_time = 0;
	double timings[5] = { 0 };

	cudaEvent_t memcpy_startEvent, memcpy_stopEvent;
	cudaEvent_t fft_startEvent, fft_stopEvent;
	checkCuda_accfft(cudaEventCreate(&memcpy_startEvent));
	checkCuda_accfft(cudaEventCreate(&memcpy_stopEvent));
	checkCuda_accfft(cudaEventCreate(&fft_startEvent));
	checkCuda_accfft(cudaEventCreate(&fft_stopEvent));
	int NY = N[1];
	float dummy_time = 0;

	if (direction == -1) {
		/**************************************************************/
		/*******************  N0/P0 x N1/P1 x N2 **********************/
		/**************************************************************/
		// FFT in Z direction
		if (xyz[2]) {
			checkCuda_accfft(cudaEventRecord(fft_startEvent, 0));
			checkCuda_accfft(
					exec_r2c(fplan_0,
					  reinterpret_cast<typename cu_type<real>::real *>(data_d),
					  reinterpret_cast<typename cu_type<real>::cplx *>(data_out_d)));
			checkCuda_accfft(cudaEventRecord(fft_stopEvent, 0));
			checkCuda_accfft(cudaEventSynchronize(fft_stopEvent)); // wait until fft is executed
			checkCuda_accfft(
					cudaEventElapsedTime(&dummy_time, fft_startEvent,
							fft_stopEvent));
			fft_time += dummy_time / 1000;
		} else
			data_out_d = data_d;

		// Perform N0/P0 transpose

		if (!oneD) {
			T_plan_1->execute_gpu(T_plan_1, data_out_d, timings, 2,
					osize_0[0], coord[0]);
		}
		/**************************************************************/
		/*******************  N0/P0 x N1 x N2/P1 **********************/
		/**************************************************************/
		if (xyz[1]) {
			checkCuda_accfft(cudaEventRecord(fft_startEvent, 0));
			for (int i = 0; i < osize_1[0]; ++i) {
				checkCuda_accfft(
            exec_c2c(fplan_1,
								reinterpret_cast<typename cu_type<real>::cplx *>(
                        &data_out_d[2 * i * osize_1[1] * osize_1[2]]),
								reinterpret_cast<typename cu_type<real>::cplx *>(
                        &data_out_d[2 * i * osize_1[1] * osize_1[2]]),
                CUFFT_FORWARD));
			}
			checkCuda_accfft(cudaEventRecord(fft_stopEvent, 0));
			checkCuda_accfft(cudaEventSynchronize(fft_stopEvent)); // wait until fft is executed
			checkCuda_accfft(
					cudaEventElapsedTime(&dummy_time, fft_startEvent,
							fft_stopEvent));
			fft_time += dummy_time / 1000;
		}

		if (oneD) {
			T_plan_2->execute_gpu(T_plan_2, data_out_d, timings, 2);
		} else {
			T_plan_2->execute_gpu(T_plan_2, data_out_d, timings, 2,
					1, coord[1]);
		}
		/**************************************************************/
		/*******************  N0 x N1/P0 x N2/P1 **********************/
		/**************************************************************/
		if (xyz[0]) {
			checkCuda_accfft(cudaEventRecord(fft_startEvent, 0));
			checkCuda_accfft(
					exec_c2c(fplan_2, reinterpret_cast<typename cu_type<real>::cplx *>(data_out_d),
							reinterpret_cast<typename cu_type<real>::cplx *>(data_out_d), CUFFT_FORWARD));
			checkCuda_accfft(cudaEventRecord(fft_stopEvent, 0));
			checkCuda_accfft(cudaEventSynchronize(fft_stopEvent)); // wait until fft is executed
			checkCuda_accfft(
					cudaEventElapsedTime(&dummy_time, fft_startEvent,
							fft_stopEvent));
			fft_time += dummy_time / 1000;
		}
	} else if (direction == 1) {
		if (xyz[0]) {
			checkCuda_accfft(cudaEventRecord(fft_startEvent, 0));
			checkCuda_accfft(
					exec_c2c(fplan_2, reinterpret_cast<typename cu_type<real>::cplx *>(data_d),
							reinterpret_cast<typename cu_type<real>::cplx *>(data_d), CUFFT_INVERSE));
			checkCuda_accfft(cudaEventRecord(fft_stopEvent, 0));
			checkCuda_accfft(cudaEventSynchronize(fft_stopEvent)); // wait until fft is executed
			checkCuda_accfft(
					cudaEventElapsedTime(&dummy_time, fft_startEvent,
							fft_stopEvent));
			fft_time += dummy_time / 1000;
		}

		if (oneD) {
			T_plan_2i->execute_gpu(T_plan_2i, (real*) data_d,
					timings, 1);
		} else {
			T_plan_2i->execute_gpu(T_plan_2i, (real*) data_d,
					timings, 1, 1, coord[1]);
		}
		/**************************************************************/
		/*******************  N0/P0 x N1 x N2/P1 **********************/
		/**************************************************************/
		if (xyz[1]) {
			checkCuda_accfft(cudaEventRecord(fft_startEvent, 0));
			for (int i = 0; i < osize_1i[0]; ++i) {
				checkCuda_accfft(
						exec_c2c(fplan_1,
								reinterpret_cast<typename cu_type<real>::cplx *>(&data_d[2 * i * NY * osize_1i[2]]),
								reinterpret_cast<typename cu_type<real>::cplx *>(&data_d[2 * i * NY * osize_1i[2]]),
								CUFFT_INVERSE));
			}
			checkCuda_accfft(cudaEventRecord(fft_stopEvent, 0));
			checkCuda_accfft(cudaEventSynchronize(fft_stopEvent)); // wait until fft is executed
			checkCuda_accfft(
					cudaEventElapsedTime(&dummy_time, fft_startEvent,
							fft_stopEvent));
			fft_time += dummy_time / 1000;
		}

		if (!oneD) {
			T_plan_1i->execute_gpu(T_plan_1i, (real*) data_d,
					timings, 1, osize_1i[0], coord[0]);
		}
		/**************************************************************/
		/*******************  N0/P0 x N1/P1 x N2 **********************/
		/**************************************************************/

		// IFFT in Z direction
		if (xyz[2]) {
			checkCuda_accfft(cudaEventRecord(fft_startEvent, 0));
			checkCuda_accfft(
					exec_c2r(iplan_0, reinterpret_cast<typename cu_type<real>::cplx *>(data_d),
							reinterpret_cast<typename cu_type<real>::real *>(data_out_d)));
			checkCuda_accfft(cudaEventRecord(fft_stopEvent, 0));
			checkCuda_accfft(cudaEventSynchronize(fft_stopEvent)); // wait until fft is executed
			checkCuda_accfft(
					cudaEventElapsedTime(&dummy_time, fft_startEvent,
							fft_stopEvent));
			fft_time += dummy_time / 1000;
		} else
			data_out_d = data_d;

	}

	checkCuda_accfft(cudaEventDestroy(memcpy_startEvent));
	checkCuda_accfft(cudaEventDestroy(memcpy_stopEvent));
	checkCuda_accfft(cudaEventDestroy(fft_startEvent));
	checkCuda_accfft(cudaEventDestroy(fft_stopEvent));
	timings[4] += fft_time;
	if (timer == NULL) {
		//delete [] timings;
	} else {
		timer[0] += timings[0];
		timer[1] += timings[1];
		timer[2] += timings[2];
		timer[3] += timings[3];
		timer[4] += timings[4];
	}

	return;
}

/**
 * Execute  R2C plan. This function is blocking and only returns after the transform is completed.
 * @note For inplace transforms, data_out should point to the same memory address as data, AND
 * the plan must have been created as inplace.
 * @param plan FFT plan created by \ref accfft_plan_dft_3d_r2cf.
 * @param data Input data in spatial domain.
 * @param data_out Output data in frequency domain.
 * @param timer See \ref timer for more details.
 * @param XYZ a bit set field that determines which directions FFT should be executed
 */
GMETHOD(void, execute_r2c)(real *data, cplx *data_out,
						double *timer, std::bitset<3> xyz) {
	if (r2c_plan_baked) {
		this->execute(-1, data, (real*) data_out, timer, xyz);
	} else {
		if (procid == 0)
			std::cout
					<< "Error. r2c plan has not been made correctly. Please first create the plan before calling execute functions."
					<< std::endl;
	}

	return;
}

/**
 * Execute  C2R plan. This function is blocking and only returns after the transform is completed.
 * @note For inplace transforms, data_out should point to the same memory address as data, AND
 * the plan must have been created as inplace.
 * @param plan FFT plan created by \ref accfft_plan_dft_3d_r2cf.
 * @param data Input data in frequency domain.
 * @param data_out Output data in frequency domain.
 * @param timer See \ref timer for more details.
 * @param XYZ a bit set field that determines which directions FFT should be executed
 */
GMETHOD(void, execute_c2r)(cplx * data,
		real *data_out, double *timer, std::bitset<3> xyz) {
	if (r2c_plan_baked) {
		this->execute(1, (real*) data, data_out, timer, xyz);
	} else {
		if (procid == 0)
			std::cout
					<< "Error. r2c plan has not been made correctly. Please first create the plan before calling execute functions."
					<< std::endl;
	}

	return;
}

/**
 * Get the local sizes of the distributed global data for a GPU  C2C transform
 * @param n Integer array of size 3, corresponding to the global data size
 * @param isize The size of the data that is locally distributed to the calling process
 * @param istart The starting index of the data that locally resides on the calling process
 * @param osize The output size of the data that locally resides on the calling process,
 * after the C2C transform is finished
 * @param ostart The output starting index of the data that locally resides on the calling process,
 * after the R2C transform is finished
 * @param c_comm Cartesian communicator returned by \ref accfft_create_comm
 * @return
 */
template <typename T>
int accfft_local_size_dft_c2c_gpu(const int *n, int *isize, int *istart, 
                             int *osize, int *ostart, MPI_Comm c_comm) {
  return accfft_local_size_dft_c2c<T>(n, isize, istart, osize, ostart, c_comm);
}
#define C2C_SIZE(real) \
  template int accfft_local_size_dft_c2c_gpu<real>( \
          const int *n, int *isize, int *istart, \
          int *osize, int *ostart, MPI_Comm c_comm);
TPL_DECL(C2C_SIZE)

/**
 * Creates a 3D  C2C parallel FFT plan. If data_out point to the same location as the input
 * data, then an inplace plan will be created. Otherwise the plan would be outplace.
 * @param n Integer array of size 3, corresponding to the global data size
 * @param data Input data in spatial domain
 * @param data_out Output data in frequency domain
 * @param c_comm Cartesian communicator returned by \ref accfft_create_comm
 * @param flags AccFFT flags, See \ref flags for more details.
 * @return
 */
GMETHOD(void, init_c2c)(int *n, cplx *data_d,
		cplx *data_out_d, MPI_Comm ic_comm, unsigned flags) {
  r2c_plan_baked = 0;
  c2c_plan_baked = 0;
  data = NULL;
  data_out = NULL;
  T_plan_1 = NULL;
  T_plan_1i = NULL;
  T_plan_2 = NULL;
  T_plan_2i = NULL;
  T_plan_y = NULL;
  T_plan_yi = NULL;
  T_plan_x = NULL;
  T_plan_xi = NULL;
  Mem_mgr = NULL;
  fplan_y = -1;
  fplan_x = -1;
  iplan_y = -1;
  iplan_x = -1;

  c_comm = ic_comm;
	MPI_Comm_rank(c_comm, &procid);
	MPI_Cart_get(c_comm, 2, np, periods, coord);
	MPI_Comm_split(c_comm, coord[0], coord[1], &row_comm);
	MPI_Comm_split(c_comm, coord[1], coord[0], &col_comm);
	N[0] = n[0];
	N[1] = n[1];
	N[2] = n[2];
	int NX = n[0], NY = n[1], NZ = n[2];
	cufftResult_t cufft_error;

	data_c = data_d;
	data_out_c = data_out_d;
	if (data_out_d == data_d) {
		inplace = true;
	} else {
		inplace = false;
	}

	if (np[1] == 1)
		oneD = true;
	else
		oneD = false;

	int alloc_local;
	int n_tuples = n[2] * 2;

	//int isize[3],osize[3],istart[3],ostart[3];
	alloc_max = accfft_local_size_dft_c2c_gpu<real>(n, isize, istart,
			osize, ostart, c_comm);

	dfft_get_local_size<real>(n[0], n[1], n[2], osize_0, ostart_0, c_comm);
	dfft_get_local_size<real>(n[0], n[2], n[1], osize_1, ostart_1, c_comm);
	dfft_get_local_size<real>(n[1], n[2], n[0], osize_2, ostart_2, c_comm);

	std::swap(osize_1[1], osize_1[2]);
	std::swap(ostart_1[1], ostart_1[2]);

	std::swap(ostart_2[1], ostart_2[2]);
	std::swap(ostart_2[0], ostart_2[1]);
	std::swap(osize_2[1], osize_2[2]);
	std::swap(osize_2[0], osize_2[1]);

	for (int i = 0; i < 3; i++) {
		osize_1i[i] = osize_1[i];
		osize_2i[i] = osize_2[i];
		ostart_1i[i] = ostart_1[i];
		ostart_2i[i] = ostart_2[i];
	}

	// fplan_0
	{
		int f_inembed[1] = { NZ };
		int f_onembed[1] = { NZ };
		int idist = (NZ);
		int odist = (NZ);
		int istride = 1;
		int ostride = 1;
		int batch = osize_0[0] * osize_0[1];  //NX;

		if (batch != 0) {
			cufft_error = cufftPlanMany(&fplan_0, 1, &n[2], f_inembed,
					istride, idist, // *inembed, istride, idist
					f_onembed, ostride, odist, // *onembed, ostride, odist
					cu_c2c, batch);
			if (cufft_error != CUFFT_SUCCESS) {
				fprintf(stderr, "CUFFT error: fplan_0 creation failed %d \n",
						cufft_error);
				throw std::runtime_error("CUFFT error");
			}
			//cufftSetCompatibilityMode(fplan,CUFFT_COMPATIBILITY_FFTW_PADDING); if (cudaGetLastError() != cudaSuccess){fprintf(stderr, "Cuda error:Failed at fplan cuda compatibility\n"); return;}
		}
	}
	// fplan_1
	{
		int f_inembed[1] = { NY };
		int f_onembed[1] = { NY };
		int idist = 1;
		int odist = 1;
		int istride = osize_1[2];
		int ostride = osize_1[2];
		int batch = osize_1[2];

		if (batch != 0) {
			cufft_error = cufftPlanMany(&fplan_1, 1, &n[1], f_inembed,
					istride, idist, // *inembed, istride, idist
					f_onembed, ostride, odist, // *onembed, ostride, odist
					cu_c2c, batch);
			if (cufft_error != CUFFT_SUCCESS) {
				fprintf(stderr, "CUFFT error: fplan_1 creation failed %d \n",
						cufft_error);
				throw std::runtime_error("CUFFT error");
			}
			//cufftSetCompatibilityMode(fplan,CUFFT_COMPATIBILITY_FFTW_PADDING); if (cudaGetLastError() != cudaSuccess){fprintf(stderr, "Cuda error:Failed at fplan cuda compatibility\n"); return;}
		}
	}
	// fplan_2
	{
		int f_inembed[1] = { NX };
		int f_onembed[1] = { NX };
		int idist = 1;
		int odist = 1;
		int istride = osize_2[1] * osize_2[2];
		int ostride = osize_2[1] * osize_2[2];
		int batch = osize_2[1] * osize_2[2];
		;

		if (batch != 0) {
			cufft_error = cufftPlanMany(&fplan_2, 1, &n[0], f_inembed,
					istride, idist, // *inembed, istride, idist
					f_onembed, ostride, odist, // *onembed, ostride, odist
					cu_c2c, batch);
			if (cufft_error != CUFFT_SUCCESS) {
				fprintf(stderr, "CUFFT error: fplan_2 creation failed %d \n",
						cufft_error);
				throw std::runtime_error("CUFFT error");
			}
			//cufftSetCompatibilityMode(fplan,CUFFT_COMPATIBILITY_FFTW_PADDING); if (cudaGetLastError() != cudaSuccess){fprintf(stderr, "Cuda error:Failed at fplan cuda compatibility\n"); return;}
		}
	}

	// 1D Decomposition
	if (oneD) {
		int NX = n[0], NY = n[1], NZ = n[2];

		Mem_mgr = new Mem_Mgr_gpu<real>(NX, NY, (NZ) * 2, c_comm, 1, alloc_max);
		T_plan_2 = new T_Plan_gpu<real>(NX, NY, (NZ) * 2, Mem_mgr,
				c_comm);
		T_plan_2i = new T_Plan_gpu<real>(NY, NX, NZ * 2, Mem_mgr,
				c_comm);

		T_plan_2->alloc_local = alloc_max;
		T_plan_2i->alloc_local = alloc_max;
		T_plan_1 = NULL;
		T_plan_1i = NULL;

		if (flags == ACCFFT_MEASURE) {
			T_plan_2->which_fast_method_gpu(T_plan_2,
					(real*) data_out_d);
		} else {
			T_plan_2->method = 2;
			T_plan_2->kway = 2;
		}
		checkCuda_accfft(cudaDeviceSynchronize());

		T_plan_2i->method = -T_plan_2->method;
		T_plan_2i->kway = T_plan_2->kway;
		T_plan_2i->kway_async = T_plan_2->kway_async;

	} // end 1d c2c

	// 2D Decomposition
	if (!oneD) {
		// the reaseon for n_tuples/2 is to avoid splitting of imag and real parts of complex numbers
		Mem_mgr = new Mem_Mgr_gpu<real>(n[1], n[2], 2, row_comm,
				osize_0[0], alloc_max);
		T_plan_1 = new T_Plan_gpu<real>(n[1], n[2], 2, Mem_mgr,
				row_comm, osize_0[0]);
		T_plan_2 = new T_Plan_gpu<real>(n[0], n[1], 2 * osize_2[2],
				Mem_mgr, col_comm);
		T_plan_2i = new T_Plan_gpu<real>(n[1], n[0], 2 * osize_2i[2],
				Mem_mgr, col_comm);
		T_plan_1i = new T_Plan_gpu<real>(n[2], n[1], 2, Mem_mgr,
				row_comm, osize_1i[0]);

		T_plan_1->alloc_local = alloc_max;
		T_plan_2->alloc_local = alloc_max;
		T_plan_2i->alloc_local = alloc_max;
		T_plan_1i->alloc_local = alloc_max;

		iplan_0 = -1;
		iplan_1 = -1;
		iplan_2 = -1;

		int coord[2], np[2], periods[2];
		MPI_Cart_get(c_comm, 2, np, periods, coord);

		if (flags == ACCFFT_MEASURE) {
			if (coord[0] == 0) {
				T_plan_1->which_fast_method_gpu(T_plan_1,
						(real*) data_out_d, osize_0[0]);
			}
		} else {
			T_plan_1->method = 2;
			T_plan_1->kway = 2;
		}

		MPI_Bcast(&T_plan_1->method, 1, MPI_INT, 0, c_comm);
		MPI_Bcast(&T_plan_1->kway, 1, MPI_INT, 0, c_comm);
		MPI_Bcast(&T_plan_1->kway_async, 1, par::Mpi_datatype<bool>::value(), 0, c_comm);
		checkCuda_accfft(cudaDeviceSynchronize());

		T_plan_2->method = T_plan_1->method;
		T_plan_2i->method = -T_plan_1->method;
		T_plan_1i->method = -T_plan_1->method;

		T_plan_2->kway = T_plan_1->kway;
		T_plan_2i->kway = T_plan_1->kway;
		T_plan_1i->kway = T_plan_1->kway;

		T_plan_2->kway_async = T_plan_1->kway_async;
		T_plan_2i->kway_async = T_plan_1->kway_async;
		T_plan_1i->kway_async = T_plan_1->kway_async;

	} // end 2d c2c

	c2c_plan_baked = true;
}

/**
 * Execute  C2C plan. This function is blocking and only returns after the transform is completed.
 * @note For inplace transforms, data_out should point to the same memory address as data, AND
 * the plan must have been created as inplace.
 * @param plan FFT plan created by \ref accfft_plan_dft_3d_r2cf.
 * @param data Input data in frequency domain.
 * @param data_out Output data in frequency domain.
 * @param timer See \ref timer for more details.
 * @param XYZ a bit set field that determines which directions FFT should be executed
 */
GMETHOD(void, execute_c2c)(int direction,
		cplx *data_d, cplx *data_out_d, double *timer, std::bitset<3> xyz) {

	if (!c2c_plan_baked) {
		if (procid == 0)
			std::cout
					<< "Error. r2c plan has not been made correctly. Please first create the plan before calling execute functions."
					<< std::endl;
		return;
	}

	if (data_d == NULL)
		data_d = data_c;
	if (data_out_d == NULL)
		data_out_d = data_out_c;
	int * coord = coord;
	int procid = procid;
	double fft_time = 0;
	double timings[5] = { 0 };

	int NY = N[1];
	cudaEvent_t memcpy_startEvent, memcpy_stopEvent;
	cudaEvent_t fft_startEvent, fft_stopEvent;
	checkCuda_accfft(cudaEventCreate(&memcpy_startEvent));
	checkCuda_accfft(cudaEventCreate(&memcpy_stopEvent));
	checkCuda_accfft(cudaEventCreate(&fft_startEvent));
	checkCuda_accfft(cudaEventCreate(&fft_stopEvent));
	cufftResult_t cufft_error;
	float dummy_time = 0;

	if (direction == -1) {
		/**************************************************************/
		/*******************  N0/P0 x N1/P1 x N2 **********************/
		/**************************************************************/
		// FFT in Z direction
		if (xyz[2]) {
			checkCuda_accfft(cudaEventRecord(fft_startEvent, 0));
			checkCuda_accfft(
					exec_c2c(fplan_0, reinterpret_cast<typename cu_type<real>::cplx *>(data_d),
							reinterpret_cast<typename cu_type<real>::cplx *>(data_out_d), CUFFT_FORWARD));
			checkCuda_accfft(cudaEventRecord(fft_stopEvent, 0));
			checkCuda_accfft(cudaEventSynchronize(fft_stopEvent)); // wait until fft is executed
			checkCuda_accfft(
					cudaEventElapsedTime(&dummy_time, fft_startEvent,
							fft_stopEvent));
			fft_time += dummy_time / 1000;
		} else
			data_out_d = data_d;

		if (!oneD) {
			T_plan_1->execute_gpu(T_plan_1, (real*) data_out_d,
					timings, 2, osize_0[0], coord[0]);
		}
		checkCuda_accfft(cudaDeviceSynchronize());
		/**************************************************************/
		/*******************  N0/P0 x N1 x N2/P1 **********************/
		/**************************************************************/
		if (xyz[1]) {
			checkCuda_accfft(cudaEventRecord(fft_startEvent, 0));
			for (int i = 0; i < osize_1[0]; ++i) {
				checkCuda_accfft(
						exec_c2c(fplan_1,
								reinterpret_cast<typename cu_type<real>::cplx *>(
                    &data_out_d[i * osize_1[1] * osize_1[2]]),
								reinterpret_cast<typename cu_type<real>::cplx *>(
                    &data_out_d[i * osize_1[1] * osize_1[2]]),
                CUFFT_FORWARD));
			}
			checkCuda_accfft(cudaEventRecord(fft_stopEvent, 0));
			checkCuda_accfft(cudaEventSynchronize(fft_stopEvent)); // wait until fft is executed
			checkCuda_accfft(
					cudaEventElapsedTime(&dummy_time, fft_startEvent,
							fft_stopEvent));
			fft_time += dummy_time / 1000;
		}


		if (oneD) {
			T_plan_2->execute_gpu(T_plan_2, (real*) data_out_d,
					timings, 2);
		} else {
			T_plan_2->execute_gpu(T_plan_2, (real*) data_out_d,
					timings, 2, 1, coord[1]);
		}
    //checkCuda_accfft(cudaDeviceSynchronize()); // DOUBLE ONLY?
		/**************************************************************/
		/*******************  N0 x N1/P0 x N2/P1 **********************/
		/**************************************************************/
		if (xyz[0]) {
			checkCuda_accfft(cudaEventRecord(fft_startEvent, 0));
			checkCuda_accfft(
					exec_c2c(fplan_2, reinterpret_cast<typename cu_type<real>::cplx *>(data_out_d),
							reinterpret_cast<typename cu_type<real>::cplx *>(data_out_d), CUFFT_FORWARD));
			checkCuda_accfft(cudaEventRecord(fft_stopEvent, 0));
			checkCuda_accfft(cudaEventSynchronize(fft_stopEvent)); // wait until fft is executed
			checkCuda_accfft(
					cudaEventElapsedTime(&dummy_time, fft_startEvent,
							fft_stopEvent));
			fft_time += dummy_time / 1000;
		}

	} else if (direction == 1) {
		if (xyz[0]) {
			checkCuda_accfft(cudaEventRecord(fft_startEvent, 0));
			checkCuda_accfft(
					exec_c2c(fplan_2, reinterpret_cast<typename cu_type<real>::cplx *>(data_d),
							reinterpret_cast<typename cu_type<real>::cplx *>(data_d), CUFFT_INVERSE));
			checkCuda_accfft(cudaDeviceSynchronize());
			checkCuda_accfft(cudaEventRecord(fft_stopEvent, 0));
			checkCuda_accfft(cudaEventSynchronize(fft_stopEvent)); // wait until fft is executed
			checkCuda_accfft(
					cudaEventElapsedTime(&dummy_time, fft_startEvent,
							fft_stopEvent));
			fft_time += dummy_time / 1000;
		}

		if (oneD) {
			T_plan_2i->execute_gpu(T_plan_2i, (real*) data_d,
					timings, 1);
		} else {
			T_plan_2i->execute_gpu(T_plan_2i, (real*) data_d,
					timings, 1, 1, coord[1]);
		}
    //checkCuda_accfft(cudaDeviceSynchronize()); DOUBLE ONLY?
		/**************************************************************/
		/*******************  N0/P0 x N1 x N2/P1 **********************/
		/**************************************************************/
		if (xyz[1]) {
			checkCuda_accfft(cudaEventRecord(fft_startEvent, 0));
			for (int i = 0; i < osize_1i[0]; ++i) {
				checkCuda_accfft(
						exec_c2c(fplan_1,
								reinterpret_cast<typename cu_type<real>::cplx *>(&data_d[i * NY * osize_1i[2]]),
								reinterpret_cast<typename cu_type<real>::cplx *>(&data_d[i * NY * osize_1i[2]]),
								CUFFT_INVERSE));
			}
			checkCuda_accfft(cudaEventRecord(fft_stopEvent, 0));
			checkCuda_accfft(cudaEventSynchronize(fft_stopEvent)); // wait until fft is executed
			checkCuda_accfft(
					cudaEventElapsedTime(&dummy_time, fft_startEvent,
							fft_stopEvent));
			fft_time += dummy_time / 1000;
		}

		if (!oneD) {
			T_plan_1i->execute_gpu(T_plan_1i, (real*) data_d,
					timings, 1, osize_1i[0], coord[0]);
		}
		/**************************************************************/
		/*******************  N0/P0 x N1/P1 x N2 **********************/
		/**************************************************************/

		if (xyz[2]) {
			checkCuda_accfft(cudaEventRecord(fft_startEvent, 0));
			checkCuda_accfft(
					exec_c2c(fplan_0, reinterpret_cast<typename cu_type<real>::cplx *>(data_d),
							reinterpret_cast<typename cu_type<real>::cplx *>(data_out_d), CUFFT_INVERSE));
			checkCuda_accfft(cudaEventRecord(fft_stopEvent, 0));
			checkCuda_accfft(cudaEventSynchronize(fft_stopEvent)); // wait until fft is executed
			checkCuda_accfft(
					cudaEventElapsedTime(&dummy_time, fft_startEvent,
							fft_stopEvent));
			fft_time += dummy_time / 1000;
		} else
			data_out_d = data_d;

	}

	checkCuda_accfft(cudaEventDestroy(memcpy_startEvent));
	checkCuda_accfft(cudaEventDestroy(memcpy_stopEvent));
	checkCuda_accfft(cudaEventDestroy(fft_startEvent));
	checkCuda_accfft(cudaEventDestroy(fft_stopEvent));
	timings[4] += fft_time;
	if (timer == NULL) {
		//delete [] timings;
	} else {
		timer[0] += timings[0];
		timer[1] += timings[1];
		timer[2] += timings[2];
		timer[3] += timings[3];
		timer[4] += timings[4];
	}

	return;
}

/**
 * Destroy  AccFFT GPU plan.
 * @param plan Input plan to be destroyed.
 */
GMETHOD1(~AccFFT_gpu)() {
	if (T_plan_1 != NULL)
		delete (T_plan_1);
	if (T_plan_1i != NULL)
		delete (T_plan_1i);
	if (T_plan_2 != NULL)
		delete (T_plan_2);
	if (T_plan_2i != NULL)
		delete (T_plan_2i);
	if (Mem_mgr != NULL)
		delete (Mem_mgr);
	if (fplan_x != -1)
		cufftDestroy(fplan_x);
	if (fplan_y != -1)
		cufftDestroy(fplan_y);
	if (iplan_x != -1)
		cufftDestroy(iplan_x);
	if (iplan_y != -1)
		cufftDestroy(iplan_y);
	if (T_plan_y != NULL)
		delete (T_plan_y);
	if (T_plan_yi != NULL)
		delete (T_plan_yi);

	if (fplan_0 != -1)
		cufftDestroy(fplan_0);
	if (fplan_1 != -1)
		cufftDestroy(fplan_1);
	if (fplan_2 != -1)
		cufftDestroy(fplan_2);

	if (iplan_0 != -1)
		cufftDestroy(iplan_0);
	//if(iplan_1!=-1)cufftDestroy(iplan_1);
	//if(iplan_2!=-1)cufftDestroy(iplan_2);

	MPI_Comm_free(&row_comm);
	MPI_Comm_free(&col_comm);
	return;
}

// templates for execution only in z direction
GMETHOD(void, execute_z)(int direction,
            real *data_d, real *data_out_d, double *timer) {
	if (data_d == NULL)
		data_d = data;
	if (data_out_d == NULL)
		data_out_d = (real *)data_out;

	int * coord = coord;
	int procid = procid;
	double fft_time = 0;
	double timings[5] = { 0 };

	cudaEvent_t fft_startEvent, fft_stopEvent;
	checkCuda_accfft(cudaEventCreate(&fft_startEvent));
	checkCuda_accfft(cudaEventCreate(&fft_stopEvent));
	int NY = N[1];
	float dummy_time = 0;

  if (direction == -1) {
    /**************************************************************/
    /*******************  N0/P0 x N1/P1 x N2 **********************/
    /**************************************************************/
    // FFT in Z direction
    checkCuda_accfft(cudaEventRecord(fft_startEvent, 0));
    checkCuda_accfft(
        exec_r2c(fplan_0, reinterpret_cast<typename cu_type<real>::real *>(data_d),
          reinterpret_cast<typename cu_type<real>::cplx *>(data_out_d)));
    checkCuda_accfft(cudaEventRecord(fft_stopEvent, 0));
    checkCuda_accfft(cudaEventSynchronize(fft_stopEvent)); // wait until fft is executed
    checkCuda_accfft(
        cudaEventElapsedTime(&dummy_time, fft_startEvent,
          fft_stopEvent));
    fft_time += dummy_time / 1000;
  } else if (direction == 1) {
    /**************************************************************/
    /*******************  N0/P0 x N1/P1 x N2 **********************/
    /**************************************************************/

    // IFFT in Z direction
    checkCuda_accfft(cudaEventRecord(fft_startEvent, 0));
    checkCuda_accfft(
        exec_c2r(iplan_0, reinterpret_cast<typename cu_type<real>::cplx *>(data_d),
          reinterpret_cast<typename cu_type<real>::real *>(data_out_d)));
    checkCuda_accfft(cudaEventRecord(fft_stopEvent, 0));
    checkCuda_accfft(cudaEventSynchronize(fft_stopEvent)); // wait until fft is executed
    checkCuda_accfft(
        cudaEventElapsedTime(&dummy_time, fft_startEvent,
          fft_stopEvent));
    fft_time += dummy_time / 1000;

  }
	checkCuda_accfft(cudaEventDestroy(fft_startEvent));
	checkCuda_accfft(cudaEventDestroy(fft_stopEvent));
  timings[4] += fft_time;
  if (timer == NULL) {
    //delete [] timings;
  } else {
    timer[0] += timings[0];
    timer[1] += timings[1];
    timer[2] += timings[2];
    timer[3] += timings[3];
    timer[4] += timings[4];
  }
  //checkCuda_accfft(cudaDeviceSynchronize()); // DOUBLE ONLY?

  return;
}

GMETHOD(void, execute_r2c_z)(
        real *data, cplx *data_out, double *timer) {
	if (r2c_plan_baked) {
		this->execute_z(-1, data, (real*) data_out, timer);
	} else {
		if (procid == 0)
			std::cout
					<< "Error. r2c plan has not been made correctly. Please first create the plan before calling execute functions."
					<< std::endl;
	}
  return;
}

GMETHOD(void, execute_c2r_z)(
                cplx *data, real *data_out, double *timer) {
	if (r2c_plan_baked) {
		this->execute_z(1, (real*) data, data_out, timer);
	} else {
		if (procid == 0)
			std::cout
					<< "Error. r2c plan has not been made correctly. Please first create the plan before calling execute functions."
					<< std::endl;
	}
  return;
}

// templates for execution only in y direction
GMETHOD(void, execute_y)(int direction,
                real *data_d, real *data_out_d, double *timer) {
	if (data_d == NULL)
		data_d = data;
	if (data_out_d == NULL)
		data_out_d = (real *)data_out;

	int * coord = coord;
	int procid = procid;
	double fft_time = 0;
	double timings[5] = { 0 };

	cudaEvent_t memcpy_startEvent, memcpy_stopEvent;
	cudaEvent_t fft_startEvent, fft_stopEvent;
	checkCuda_accfft(cudaEventCreate(&memcpy_startEvent));
	checkCuda_accfft(cudaEventCreate(&memcpy_stopEvent));
	checkCuda_accfft(cudaEventCreate(&fft_startEvent));
	checkCuda_accfft(cudaEventCreate(&fft_stopEvent));
	int NY = N[1];
	float dummy_time = 0;
  int64_t alloc_max = alloc_max;

  int64_t N_local = isize[0] * isize[1] * isize[2];
  real* cwork_d = Mem_mgr->buffer_d3;

	if (direction == -1) {
		/**************************************************************/
		/*******************  N0/P0 x N1/P1 x N2 **********************/
		/**************************************************************/
    timings[0] += -MPI_Wtime();
    cudaMemcpy(cwork_d, data_d, N_local * sizeof(real), cudaMemcpyDeviceToDevice);
    timings[0] += +MPI_Wtime();
		if (!oneD) {
			T_plan_y->execute_gpu(T_plan_y, cwork_d, timings, 2,
					osize_y[0], coord[0]);
		}
		/**************************************************************/
		/*******************  N0/P0 x N1 x N2/P1 **********************/
		/**************************************************************/
			checkCuda_accfft(cudaEventRecord(fft_startEvent, 0));
    for (int i = 0; i < osize_y[0]; ++i) {
      checkCuda_accfft(
          exec_r2c(fplan_y,
            reinterpret_cast<typename cu_type<real>::real *>(
                &cwork_d[i * osize_y[1] * osize_y[2]]),
            reinterpret_cast<typename cu_type<real>::cplx *>(
                &data_out_d[2 * i * osize_yi[1] * osize_y[2]]))
          );
    }
			checkCuda_accfft(cudaEventRecord(fft_stopEvent, 0));
			checkCuda_accfft(cudaEventSynchronize(fft_stopEvent)); // wait until fft is executed
			checkCuda_accfft(
					cudaEventElapsedTime(&dummy_time, fft_startEvent,
							fft_stopEvent));
			fft_time += dummy_time / 1000;
	} else if (direction == 1) {
		/**************************************************************/
		/*******************  N0/P0 x N1 x N2/P1 **********************/
		/**************************************************************/
			checkCuda_accfft(cudaEventRecord(fft_startEvent, 0));
    for (int i = 0; i < osize_yi[0]; ++i) {
      checkCuda_accfft(
          exec_c2r(iplan_y,
            reinterpret_cast<typename cu_type<real>::cplx *>(
                &data_d[2 * i * osize_yi[1] * osize_yi[2]]),
            reinterpret_cast<typename cu_type<real>::real *>(
                &cwork_d[i * osize_y[1] * osize_yi[2]]))
            );
    }
			checkCuda_accfft(cudaEventRecord(fft_stopEvent, 0));
			checkCuda_accfft(cudaEventSynchronize(fft_stopEvent)); // wait until fft is executed
			checkCuda_accfft(
					cudaEventElapsedTime(&dummy_time, fft_startEvent,
							fft_stopEvent));
			fft_time += dummy_time / 1000;

		if (!oneD) {
			T_plan_yi->execute_gpu(T_plan_yi, (real*) cwork_d,
					timings, 1, osize_yi[0], coord[0]);
		}
    timings[0] += -MPI_Wtime();
    cudaMemcpy(data_out_d, cwork_d, N_local * sizeof(real), cudaMemcpyDeviceToDevice);
    timings[0] += +MPI_Wtime();
		/**************************************************************/
		/*******************  N0/P0 x N1/P1 x N2 **********************/
		/**************************************************************/
	}

	checkCuda_accfft(cudaEventDestroy(memcpy_startEvent));
	checkCuda_accfft(cudaEventDestroy(memcpy_stopEvent));
	checkCuda_accfft(cudaEventDestroy(fft_startEvent));
	checkCuda_accfft(cudaEventDestroy(fft_stopEvent));
	timings[4] += fft_time;
	if (timer == NULL) {
		//delete [] timings;
	} else {
		timer[0] += timings[0];
		timer[1] += timings[1];
		timer[2] += timings[2];
		timer[3] += timings[3];
		timer[4] += timings[4];
	}

	return;
}

GMETHOD(void, execute_r2c_y)(
        real *data, cplx *data_out, double *timer) {
	if (r2c_plan_baked) {
		this->execute_y(-1, data, (real*) data_out, timer);
	} else {
		if (procid == 0)
			std::cout
					<< "Error. r2c plan has not been made correctly. Please first create the plan before calling execute functions."
					<< std::endl;
	}
  return;
}

GMETHOD(void, execute_c2r_y)(
                cplx *data, real *data_out, double *timer) {
	if (r2c_plan_baked) {
		this->execute_y(1, (real*) data, data_out, timer);
	} else {
		if (procid == 0)
			std::cout
					<< "Error. r2c plan has not been made correctly. Please first create the plan before calling execute functions."
					<< std::endl;
	}
  return;
}

// templates for execution only in x direction
GMETHOD(void, execute_x)(int direction,
            real *data_d, real *data_out_d, double *timer) {
	if (data_d == NULL)
		data_d = data;
	if (data_out_d == NULL)
		data_out_d = (real *)data_out;

	int procid = procid;
	double fft_time = 0;
	double timings[5] = { 0 };

	cudaEvent_t memcpy_startEvent, memcpy_stopEvent;
	cudaEvent_t fft_startEvent, fft_stopEvent;
	checkCuda_accfft(cudaEventCreate(&memcpy_startEvent));
	checkCuda_accfft(cudaEventCreate(&memcpy_stopEvent));
	checkCuda_accfft(cudaEventCreate(&fft_startEvent));
	checkCuda_accfft(cudaEventCreate(&fft_stopEvent));
	int NY = N[1];
	float dummy_time = 0;
  int64_t alloc_max = alloc_max;

  int64_t N_local = isize[0] * isize[1] * isize[2];
  real* cwork_d = Mem_mgr->buffer_d3;

	if (direction == -1) {
		/**************************************************************/
		/*******************  N0/P0 x N1/P1 x N2 **********************/
		/**************************************************************/
    timings[0] += -MPI_Wtime();
    cudaMemcpy(cwork_d, data_d, N_local * sizeof(real), cudaMemcpyDeviceToDevice);
    timings[0] += +MPI_Wtime();
		T_plan_x->execute_gpu(T_plan_x, cwork_d, timings, 2);
		/**************************************************************/
		/*******************  N0/P0 x N1 x N2/P1 **********************/
		/**************************************************************/
			checkCuda_accfft(cudaEventRecord(fft_startEvent, 0));
    checkCuda_accfft(
        exec_r2c(fplan_x, reinterpret_cast<typename cu_type<real>::real *>(cwork_d),
          reinterpret_cast<typename cu_type<real>::cplx *>(data_out_d)));
			checkCuda_accfft(cudaEventRecord(fft_stopEvent, 0));
			checkCuda_accfft(cudaEventSynchronize(fft_stopEvent)); // wait until fft is executed
			checkCuda_accfft(
					cudaEventElapsedTime(&dummy_time, fft_startEvent,
							fft_stopEvent));
			fft_time += dummy_time / 1000;

	} else if (direction == 1) {
		/**************************************************************/
		/*******************  N0/P0 x N1 x N2/P1 **********************/
		/**************************************************************/
			checkCuda_accfft(cudaEventRecord(fft_startEvent, 0));
    checkCuda_accfft(
        exec_c2r(iplan_x, reinterpret_cast<typename cu_type<real>::cplx *>(data_d),
          reinterpret_cast<typename cu_type<real>::real *>(data_d)));
			checkCuda_accfft(cudaEventRecord(fft_stopEvent, 0));
			checkCuda_accfft(cudaEventSynchronize(fft_stopEvent)); // wait until fft is executed
			checkCuda_accfft(
					cudaEventElapsedTime(&dummy_time, fft_startEvent,
							fft_stopEvent));
			fft_time += dummy_time / 1000;

		T_plan_xi->execute_gpu(T_plan_xi, data_d, timings, 1);
    timings[0] += -MPI_Wtime();
    cudaMemcpy(data_out_d, data_d, N_local * sizeof(real), cudaMemcpyDeviceToDevice);
    timings[0] += +MPI_Wtime();
		/**************************************************************/
		/*******************  N0/P0 x N1/P1 x N2 **********************/
		/**************************************************************/
	}

	checkCuda_accfft(cudaEventDestroy(memcpy_startEvent));
	checkCuda_accfft(cudaEventDestroy(memcpy_stopEvent));
	checkCuda_accfft(cudaEventDestroy(fft_startEvent));
	checkCuda_accfft(cudaEventDestroy(fft_stopEvent));
	timings[4] += fft_time;
	if (timer == NULL) {
		//delete [] timings;
	} else {
		timer[0] += timings[0];
		timer[1] += timings[1];
		timer[2] += timings[2];
		timer[3] += timings[3];
		timer[4] += timings[4];
	}
        //checkCuda_accfft(cudaDeviceSynchronize()); // DOUBLE ONLY?

	return;
}

GMETHOD(void, execute_r2c_x)(
        real *data, cplx *data_out, double *timer) {
	if (r2c_plan_baked) {
		this->execute_x(-1, data, (real*) data_out, timer);
	} else {
		if (procid == 0)
			std::cout
					<< "Error. r2c plan has not been made correctly. Please first create the plan before calling execute functions."
					<< std::endl;
	}
  return;
}

GMETHOD(void, execute_c2r_x)(
                cplx *data, real *data_out, double *timer) {
	if (r2c_plan_baked) {
		this->execute_x(1, (real*) data, data_out, timer);
	} else {
		if (procid == 0)
			std::cout
					<< "Error. r2c plan has not been made correctly. Please first create the plan before calling execute functions."
					<< std::endl;
	}
  return;
}

