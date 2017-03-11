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

#include "accfft_gpuf.h"
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
#define VERBOSE 0

/**
 * Cleanup all CPU resources
 */
void accfft_cleanup_gpuf() {
	// empty for now
}

/**
 * Get the local sizes of the distributed global data for a GPU single precision R2C transform
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
int accfft_local_size_dft_r2c_gpuf(int * n, int * isize, int * istart, int * osize,
		int *ostart, MPI_Comm c_comm) {
  return accfft_local_size_dft_r2c_t<float>(n, isize, istart, osize, ostart, c_comm);
}

/**
 * Creates a 3D single precision R2C parallel FFT plan. If data_out point to the same location as the input
 * data, then an inplace plan will be created. Otherwise the plan would be outplace.
 * @param n Integer array of size 3, corresponding to the global data size
 * @param data Input data in spatial domain
 * @param data_out Output data in frequency domain
 * @param c_comm Cartesian communicator returned by \ref accfft_create_comm
 * @param flags AccFFT flags, See \ref flags for more details.
 * @return
 */
accfft_plan_gpuf* accfft_plan_dft_3d_r2c_gpuf(int * n, float * data_d,
		float * data_out_d, MPI_Comm c_comm, unsigned flags) {
	accfft_plan_gpuf *plan = new accfft_plan_gpuf;
	int procid;
	MPI_Comm_rank(c_comm, &procid);
	plan->procid = procid;
	MPI_Cart_get(c_comm, 2, plan->np, plan->periods, plan->coord);
	plan->c_comm = c_comm;
	int *coord = plan->coord;
	MPI_Comm_split(c_comm, coord[0], coord[1], &plan->row_comm);
	MPI_Comm_split(c_comm, coord[1], coord[0], &plan->col_comm);
	plan->N[0] = n[0];
	plan->N[1] = n[1];
	plan->N[2] = n[2];
	plan->data = data_d;
	plan->data_out = data_out_d;

	if (plan->np[1] == 1)
		plan->oneD = true;
	else
		plan->oneD = false;

	if (data_out_d == data_d) {
		plan->inplace = true;
	} else {
		plan->inplace = false;
	}

	int *osize_0 = plan->osize_0, *ostart_0 = plan->ostart_0;
	int *osize_1 = plan->osize_1, *ostart_1 = plan->ostart_1;
	int *osize_2 = plan->osize_2, *ostart_2 = plan->ostart_2;
	int *osize_1i = plan->osize_1i, *ostart_1i = plan->ostart_1i;
	int *osize_2i = plan->osize_2i, *ostart_2i = plan->ostart_2i;
  int* osize_y = plan->osize_y, *ostart_y = plan->ostart_y;
  int* osize_yi = plan->osize_yi;
  int* osize_x = plan->osize_x, *ostart_x = plan->ostart_x;
  int* osize_xi = plan->osize_xi;
  int* isize = plan->isize;

	int alloc_max = 0;
	int n_tuples_i, n_tuples_o;
	//plan->inplace==true ? n_tuples=(n[2]/2+1)*2: n_tuples=n[2]*2;
	plan->inplace == true ? n_tuples_i = (n[2] / 2 + 1) * 2 : n_tuples_i = n[2];
	n_tuples_o = (n[2] / 2 + 1) * 2;

	//int isize[3],osize[3],istart[3],ostart[3];
	alloc_max = accfft_local_size_dft_r2c_gpuf(n, plan->isize, plan->istart,
			plan->osize, plan->ostart, c_comm);
	plan->alloc_max = alloc_max;

	dfft_get_local_size_t<float>(n[0], n[1], n_tuples_o / 2, osize_0, ostart_0, c_comm);
	dfft_get_local_size_t<float>(n[0], n_tuples_o / 2, n[1], osize_1, ostart_1,
			c_comm);
	dfft_get_local_size_t<float>(n[1], n_tuples_o / 2, n[0], osize_2, ostart_2,
			c_comm);

	std::swap(osize_1[1], osize_1[2]);
	std::swap(ostart_1[1], ostart_1[2]);

	std::swap(ostart_2[1], ostart_2[2]);
	std::swap(ostart_2[0], ostart_2[1]);
	std::swap(osize_2[1], osize_2[2]);
	std::swap(osize_2[0], osize_2[1]);

  // osize_y is the configuration after y transpose
	dfft_get_local_size_t<float>(n[0], n[2], n[1], osize_y, ostart_y, c_comm);
	std::swap(osize_y[1], osize_y[2]);
	std::swap(ostart_y[1], ostart_y[2]);
  osize_yi[0] = osize_y[0];
  osize_yi[1] = osize_y[1] / 2 + 1;
  osize_yi[2] = osize_y[2];

  // for x only fft. The strategy is to divide (N1/P1 x N2) by P0 completely.
  // So we treat the last size to be just 1.
	dfft_get_local_size_t<float>(isize[1] * n[2], n[2], n[0], osize_x, ostart_x, c_comm);
  osize_x[1] = osize_x[0];
  osize_x[0] = osize_x[2];
  osize_x[2] = 1;

	dfft_get_local_size_t<float>(isize[1] * n[2], n[2], n[0] / 2 + 1, osize_xi, ostart_x, c_comm);
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
			cufft_error = cufftPlanMany(&plan->fplan_0, 1, &n[2], f_inembed,
					istride, idist, // *inembed, istride, idist
					f_onembed, ostride, odist, // *onembed, ostride, odist
					CUFFT_R2C, batch);
			if (cufft_error != CUFFT_SUCCESS) {
				fprintf(stderr, "CUFFT error: fplan_0 creation failed %d \n",
						cufft_error);
				return NULL;
			}
			//cufftSetCompatibilityMode(fplan,CUFFT_COMPATIBILITY_FFTW_PADDING); if (cudaGetLastError() != cudaSuccess){fprintf(stderr, "Cuda error:Failed at fplan cuda compatibility\n"); return;}
		}
		if (batch != 0) {
			cufft_error = cufftPlanMany(&plan->iplan_0, 1, &n[2], f_onembed,
					ostride, odist, // *onembed, ostride, odist
					f_inembed, istride, idist, // *inembed, istride, idist
					CUFFT_C2R, batch);
			if (cufft_error != CUFFT_SUCCESS) {
				fprintf(stderr, "CUFFT error: iplan_0 creation failed %d \n",
						cufft_error);
				return NULL;
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
			cufft_error = cufftPlanMany(&plan->fplan_y, 1, &n[1], f_inembed,
					istride, idist, // *inembed, istride, idist
					f_onembed, ostride, odist, // *onembed, ostride, odist
					CUFFT_R2C, batch);
			if (cufft_error != CUFFT_SUCCESS) {
				fprintf(stderr, "CUFFT error: fplan_y creation failed %d \n",
						cufft_error);
				return NULL;
			}
			//cufftSetCompatibilityMode(fplan,CUFFT_COMPATIBILITY_FFTW_PADDING); if (cudaGetLastError() != cudaSuccess){fprintf(stderr, "Cuda error:Failed at fplan cuda compatibility\n"); return;}
		}
		if (batch != 0) {
			cufft_error = cufftPlanMany(&plan->iplan_y, 1, &n[1], f_onembed,
					ostride, odist, // *inembed, istride, idist
					f_inembed, istride, idist, // *onembed, ostride, odist
					CUFFT_C2R, batch);
			if (cufft_error != CUFFT_SUCCESS) {
				fprintf(stderr, "CUFFT error: iplan_y creation failed %d \n",
						cufft_error);
				return NULL;
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
			cufft_error = cufftPlanMany(&plan->fplan_1, 1, &n[1], f_inembed,
					istride, idist, // *inembed, istride, idist
					f_onembed, ostride, odist, // *onembed, ostride, odist
					CUFFT_C2C, batch);
			if (cufft_error != CUFFT_SUCCESS) {
				fprintf(stderr, "CUFFT error: fplan_1 creation failed %d \n",
						cufft_error);
				return NULL;
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
			cufft_error = cufftPlanMany(&plan->fplan_2, 1, &n[0], f_inembed,
					istride, idist, // *inembed, istride, idist
					f_onembed, ostride, odist, // *onembed, ostride, odist
					CUFFT_C2C, batch);
			if (cufft_error != CUFFT_SUCCESS) {
				fprintf(stderr, "CUFFT error: fplan_2 creation failed %d \n",
						cufft_error);
				return NULL;
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
			cufft_error = cufftPlanMany(&plan->fplan_x, 1, &n[0], f_inembed,
					istride, idist, // *inembed, istride, idist
					f_onembed, ostride, odist, // *onembed, ostride, odist
					CUFFT_R2C, batch);
			if (cufft_error != CUFFT_SUCCESS) {
				fprintf(stderr, "CUFFT error: fplan_x creation failed %d \n",
						cufft_error);
				return NULL;
			}
			//cufftSetCompatibilityMode(fplan,CUFFT_COMPATIBILITY_FFTW_PADDING); if (cudaGetLastError() != cudaSuccess){fprintf(stderr, "Cuda error:Failed at fplan cuda compatibility\n"); return;}
		}
		if (batch != 0) {
			cufft_error = cufftPlanMany(&plan->iplan_x, 1, &n[0], f_onembed,
					ostride, odist, // *onembed, ostride, odist
					f_inembed, istride, idist, // *inembed, istride, idist
					CUFFT_C2R, batch);
			if (cufft_error != CUFFT_SUCCESS) {
				fprintf(stderr, "CUFFT error: iplan_x creation failed %d \n",
						cufft_error);
				return NULL;
			}
			//cufftSetCompatibilityMode(fplan,CUFFT_COMPATIBILITY_FFTW_PADDING); if (cudaGetLastError() != cudaSuccess){fprintf(stderr, "Cuda error:Failed at fplan cuda compatibility\n"); return;}
		}
	}

	// 1D Decomposition
	if (plan->oneD) {
		int N0 = n[0], N1 = n[1], N2 = n[2];

		plan->Mem_mgr = new Mem_Mgr_gpu<float>(N0, N1, n_tuples_o, c_comm, 1, plan->alloc_max);
		plan->T_plan_2 = new T_Plan_gpu<float>(N0, N1, n_tuples_o,
				plan->Mem_mgr, c_comm);
		plan->T_plan_2i = new T_Plan_gpu<float>(N1, N0, n_tuples_o,
				plan->Mem_mgr, c_comm);
		plan->T_plan_1 = NULL;
		plan->T_plan_1i = NULL;

		plan->alloc_max = alloc_max;
		plan->T_plan_2->alloc_local = alloc_max;
		plan->T_plan_2i->alloc_local = alloc_max;

		if (flags == ACCFFT_MEASURE) {
			plan->T_plan_2->which_fast_method_gpu(plan->T_plan_2, data_out_d);
		} else {
			plan->T_plan_2->method = 2;
			plan->T_plan_2->kway = 2;
		}
		checkCuda_accfft(cudaDeviceSynchronize());
		MPI_Barrier(plan->c_comm);

		plan->T_plan_2i->method = -plan->T_plan_2->method;
		plan->T_plan_2i->kway = plan->T_plan_2->kway;
		plan->T_plan_2i->kway_async = plan->T_plan_2->kway_async;

	}

	// 2D Decomposition
	if (!plan->oneD) {
		// the reaseon for n_tuples/2 is to avoid splitting of imag and real parts of complex numbers
		plan->Mem_mgr = new Mem_Mgr_gpu<float>(n[1], n_tuples_o / 2, 2,
				plan->row_comm, osize_0[0], alloc_max);
		plan->T_plan_1 = new T_Plan_gpu<float>(n[1], n_tuples_o / 2, 2,
				plan->Mem_mgr, plan->row_comm, osize_0[0]);
		plan->T_plan_2 = new T_Plan_gpu<float>(n[0], n[1], osize_2[2] * 2,
				plan->Mem_mgr, plan->col_comm);
		plan->T_plan_2i = new T_Plan_gpu<float>(n[1], n[0], osize_2i[2] * 2,
				plan->Mem_mgr, plan->col_comm);
		plan->T_plan_1i = new T_Plan_gpu<float>(n_tuples_o / 2, n[1], 2,
				plan->Mem_mgr, plan->row_comm, osize_1i[0]);
    // to transpose N0/P0 x N1/P1 x N2 -> N0/P0 x N1 x N2/P1
		plan->T_plan_y = new T_Plan_gpu<float>(n[1], n[2], 1,
				plan->Mem_mgr, plan->row_comm, plan->isize[0]);
    // to transpose N0/P0 x N1 x N2/P1 -> N0/P0 x N1/P1 x N2
		plan->T_plan_yi = new T_Plan_gpu<float>(n[2], n[1], 1,
				plan->Mem_mgr, plan->row_comm, plan->isize[0]);


		plan->T_plan_1->alloc_local = plan->alloc_max;
		plan->T_plan_2->alloc_local = plan->alloc_max;
		plan->T_plan_2i->alloc_local = plan->alloc_max;
		plan->T_plan_1i->alloc_local = plan->alloc_max;
		plan->T_plan_y->alloc_local = plan->alloc_max;
		plan->T_plan_yi->alloc_local = plan->alloc_max;

		if (flags == ACCFFT_MEASURE) {
			if (coord[0] == 0) {
				plan->T_plan_1->which_fast_method_gpu(plan->T_plan_1,
						data_out_d, osize_0[0]);
			}
		} else {
			plan->T_plan_1->method = 2;
			plan->T_plan_1->kway = 2;
		}

		MPI_Bcast(&plan->T_plan_1->method, 1, MPI_INT, 0, c_comm);
		MPI_Bcast(&plan->T_plan_1->kway, 1, MPI_INT, 0, c_comm);
		MPI_Bcast(&plan->T_plan_1->kway_async, 1, MPI::BOOL, 0, c_comm);

		checkCuda_accfft(cudaDeviceSynchronize());
		MPI_Barrier(plan->c_comm);
		plan->T_plan_1->method = plan->T_plan_1->method;
		plan->T_plan_2->method = plan->T_plan_1->method;
		plan->T_plan_2i->method = -plan->T_plan_1->method;
		plan->T_plan_1i->method = -plan->T_plan_1->method;
		plan->T_plan_y->method = plan->T_plan_1->method;
		plan->T_plan_yi->method = -plan->T_plan_1->method;

		plan->T_plan_1->kway = plan->T_plan_1->kway;
		plan->T_plan_2->kway = plan->T_plan_1->kway;
		plan->T_plan_2i->kway = plan->T_plan_1->kway;
		plan->T_plan_1i->kway = plan->T_plan_1->kway;
		plan->T_plan_y->kway = plan->T_plan_1->kway;
		plan->T_plan_yi->kway = plan->T_plan_1->kway;

		plan->T_plan_1->kway_async = plan->T_plan_1->kway_async;
		plan->T_plan_2->kway_async = plan->T_plan_1->kway_async;
		plan->T_plan_2i->kway_async = plan->T_plan_1->kway_async;
		plan->T_plan_1i->kway_async = plan->T_plan_1->kway_async;
		plan->T_plan_y->kway_async = plan->T_plan_1->kway_async;
		plan->T_plan_yi->kway_async = plan->T_plan_1->kway_async;

		plan->iplan_1 = -1;
		plan->iplan_2 = -1;

	} // end 2d r2c

  // T_plan_x has to be created for both oneD and !oneD
  // Note that the T method is set via T_plan_2 for oneD case
  // to transpose N0/P0 x N1/P1 x N2 -> N0 x (N1/P1 x N2)/P0
	plan->T_plan_x = new T_Plan_gpu<float>(n[0], isize[1] * isize[2], 1,
			plan->Mem_mgr, plan->col_comm, 1);
  // to transpose N0 x (N1/P1 x N2)/P0 -> N0/P0 x N1/P1 x N2
	plan->T_plan_xi = new T_Plan_gpu<float>(isize[1] * isize[2], n[0], 1,
			plan->Mem_mgr, plan->col_comm, 1);
	plan->T_plan_x->alloc_local = plan->alloc_max;
	plan->T_plan_xi->alloc_local = plan->alloc_max;
	plan->T_plan_x->method = plan->T_plan_2->method;
	plan->T_plan_xi->method = -plan->T_plan_2->method;
	plan->T_plan_x->kway = plan->T_plan_2->kway;
	plan->T_plan_xi->kway = plan->T_plan_2->kway;
	plan->T_plan_x->kway_async = plan->T_plan_2->kway_async;
	plan->T_plan_xi->kway_async = plan->T_plan_2->kway_async;
	plan->r2c_plan_baked = true;
	return plan;

} //end accfft_plan_dft_3d_r2c_gpuf

void accfft_execute_gpuf(accfft_plan_gpuf* plan, int direction, float * data_d,
		float * data_out_d, double * timer, std::bitset<3> xyz) {

	if (data_d == NULL)
		data_d = plan->data;
	if (data_out_d == NULL)
		data_out_d = plan->data_out;

	int * coords = plan->coord;
	int procid = plan->procid;
	double fft_time = 0;
	double timings[5] = { 0 };

	cudaEvent_t memcpy_startEvent, memcpy_stopEvent;
	cudaEvent_t fft_startEvent, fft_stopEvent;
	checkCuda_accfft(cudaEventCreate(&memcpy_startEvent));
	checkCuda_accfft(cudaEventCreate(&memcpy_stopEvent));
	checkCuda_accfft(cudaEventCreate(&fft_startEvent));
	checkCuda_accfft(cudaEventCreate(&fft_stopEvent));
	int NY = plan->N[1];
	float dummy_time = 0;

	int *osize_0 = plan->osize_0; // *ostart_0 =plan->ostart_0;
	int *osize_1 = plan->osize_1; // *ostart_1 =plan->ostart_1;
	//int *osize_2 =plan->osize_2, *ostart_2 =plan->ostart_2;
	int *osize_1i = plan->osize_1i;  //*ostart_1i=plan->ostart_1i;
	//int *osize_2i=plan->osize_2i,*ostart_2i=plan->ostart_2i;

	if (direction == -1) {
		/**************************************************************/
		/*******************  N0/P0 x N1/P1 x N2 **********************/
		/**************************************************************/
		// FFT in Z direction
		if (xyz[2]) {
			checkCuda_accfft(cudaEventRecord(fft_startEvent, 0));
			checkCuda_accfft(
					cufftExecR2C(plan->fplan_0, (cufftReal*) data_d,
							(cufftComplex*) data_out_d));
			checkCuda_accfft(cudaEventRecord(fft_stopEvent, 0));
			checkCuda_accfft(cudaEventSynchronize(fft_stopEvent)); // wait until fft is executed
			checkCuda_accfft(
					cudaEventElapsedTime(&dummy_time, fft_startEvent,
							fft_stopEvent));
			fft_time += dummy_time / 1000;
		} else
			data_out_d = data_d;

		// Perform N0/P0 transpose

		if (!plan->oneD) {
			plan->T_plan_1->execute_gpu(plan->T_plan_1, data_out_d, timings, 2,
					osize_0[0], coords[0]);
		}
		/**************************************************************/
		/*******************  N0/P0 x N1 x N2/P1 **********************/
		/**************************************************************/
		if (xyz[1]) {
			checkCuda_accfft(cudaEventRecord(fft_startEvent, 0));
			for (int i = 0; i < osize_1[0]; ++i) {
				checkCuda_accfft(
						cufftExecC2C(plan->fplan_1,
								(cufftComplex*) &data_out_d[2 * i * osize_1[1]
										* osize_1[2]],
								(cufftComplex*) &data_out_d[2 * i * osize_1[1]
										* osize_1[2]], CUFFT_FORWARD));
			}
			checkCuda_accfft(cudaEventRecord(fft_stopEvent, 0));
			checkCuda_accfft(cudaEventSynchronize(fft_stopEvent)); // wait until fft is executed
			checkCuda_accfft(
					cudaEventElapsedTime(&dummy_time, fft_startEvent,
							fft_stopEvent));
			fft_time += dummy_time / 1000;
			MPI_Barrier(plan->c_comm);
		}

		if (plan->oneD) {
			plan->T_plan_2->execute_gpu(plan->T_plan_2, data_out_d, timings, 2);
		} else {
			plan->T_plan_2->execute_gpu(plan->T_plan_2, data_out_d, timings, 2,
					1, coords[1]);
		}
		MPI_Barrier(plan->c_comm);
		/**************************************************************/
		/*******************  N0 x N1/P0 x N2/P1 **********************/
		/**************************************************************/
		if (xyz[0]) {
			checkCuda_accfft(cudaEventRecord(fft_startEvent, 0));
			checkCuda_accfft(
					cufftExecC2C(plan->fplan_2, (cufftComplex*) data_out_d,
							(cufftComplex*) data_out_d, CUFFT_FORWARD));
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
					cufftExecC2C(plan->fplan_2, (cufftComplex*) data_d,
							(cufftComplex*) data_d, CUFFT_INVERSE));
			checkCuda_accfft(cudaEventRecord(fft_stopEvent, 0));
			checkCuda_accfft(cudaEventSynchronize(fft_stopEvent)); // wait until fft is executed
			checkCuda_accfft(
					cudaEventElapsedTime(&dummy_time, fft_startEvent,
							fft_stopEvent));
			fft_time += dummy_time / 1000;
			MPI_Barrier(plan->c_comm);
		}

		if (plan->oneD) {
			plan->T_plan_2i->execute_gpu(plan->T_plan_2i, (float*) data_d,
					timings, 1);
		} else {
			plan->T_plan_2i->execute_gpu(plan->T_plan_2i, (float*) data_d,
					timings, 1, 1, coords[1]);
		}
		MPI_Barrier(plan->c_comm);
		/**************************************************************/
		/*******************  N0/P0 x N1 x N2/P1 **********************/
		/**************************************************************/
		if (xyz[1]) {
			checkCuda_accfft(cudaEventRecord(fft_startEvent, 0));
			for (int i = 0; i < osize_1i[0]; ++i) {
				checkCuda_accfft(
						cufftExecC2C(plan->fplan_1,
								(cufftComplex*) &data_d[2 * i * NY * osize_1i[2]],
								(cufftComplex*) &data_d[2 * i * NY * osize_1i[2]],
								CUFFT_INVERSE));
			}
			checkCuda_accfft(cudaEventRecord(fft_stopEvent, 0));
			checkCuda_accfft(cudaEventSynchronize(fft_stopEvent)); // wait until fft is executed
			checkCuda_accfft(
					cudaEventElapsedTime(&dummy_time, fft_startEvent,
							fft_stopEvent));
			fft_time += dummy_time / 1000;
			MPI_Barrier(plan->c_comm);
		}

		if (!plan->oneD) {
			plan->T_plan_1i->execute_gpu(plan->T_plan_1i, (float*) data_d,
					timings, 1, osize_1i[0], coords[0]);
		}
		MPI_Barrier(plan->c_comm);
		/**************************************************************/
		/*******************  N0/P0 x N1/P1 x N2 **********************/
		/**************************************************************/

		// IFFT in Z direction
		if (xyz[2]) {
			checkCuda_accfft(cudaEventRecord(fft_startEvent, 0));
			checkCuda_accfft(
					cufftExecC2R(plan->iplan_0, (cufftComplex*) data_d,
							(cufftReal*) data_out_d));
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
	MPI_Barrier(plan->c_comm);

	return;
}

/**
 * Execute single precision R2C plan. This function is blocking and only returns after the transform is completed.
 * @note For inplace transforms, data_out should point to the same memory address as data, AND
 * the plan must have been created as inplace.
 * @param plan FFT plan created by \ref accfft_plan_dft_3d_r2cf.
 * @param data Input data in spatial domain.
 * @param data_out Output data in frequency domain.
 * @param timer See \ref timer for more details.
 * @param XYZ a bit set field that determines which directions FFT should be executed
 */
void accfft_execute_r2c_gpuf(accfft_plan_gpuf* plan, float * data,
		Complexf * data_out, double * timer, std::bitset<3> xyz) {
	if (plan->r2c_plan_baked) {
		accfft_execute_gpuf(plan, -1, data, (float*) data_out, timer, xyz);
	} else {
		if (plan->procid == 0)
			std::cout
					<< "Error. r2c plan has not been made correctly. Please first create the plan before calling execute functions."
					<< std::endl;
	}

	return;
}

/**
 * Execute single precision C2R plan. This function is blocking and only returns after the transform is completed.
 * @note For inplace transforms, data_out should point to the same memory address as data, AND
 * the plan must have been created as inplace.
 * @param plan FFT plan created by \ref accfft_plan_dft_3d_r2cf.
 * @param data Input data in frequency domain.
 * @param data_out Output data in frequency domain.
 * @param timer See \ref timer for more details.
 * @param XYZ a bit set field that determines which directions FFT should be executed
 */
void accfft_execute_c2r_gpuf(accfft_plan_gpuf* plan, Complexf * data,
		float * data_out, double* timer, std::bitset<3> xyz) {
	if (plan->r2c_plan_baked) {
		accfft_execute_gpuf(plan, 1, (float*) data, data_out, timer, xyz);
	} else {
		if (plan->procid == 0)
			std::cout
					<< "Error. r2c plan has not been made correctly. Please first create the plan before calling execute functions."
					<< std::endl;
	}

	return;
}

/**
 * Get the local sizes of the distributed global data for a GPU single precision C2C transform
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
int accfft_local_size_dft_c2c_gpuf(int * n, int * isize, int * istart, int * osize,
		int *ostart, MPI_Comm c_comm) {
  return accfft_local_size_dft_c2c_t<float>(n, isize, istart, osize, ostart, c_comm);
}

/**
 * Creates a 3D single precision C2C parallel FFT plan. If data_out point to the same location as the input
 * data, then an inplace plan will be created. Otherwise the plan would be outplace.
 * @param n Integer array of size 3, corresponding to the global data size
 * @param data Input data in spatial domain
 * @param data_out Output data in frequency domain
 * @param c_comm Cartesian communicator returned by \ref accfft_create_comm
 * @param flags AccFFT flags, See \ref flags for more details.
 * @return
 */
accfft_plan_gpuf* accfft_plan_dft_3d_c2c_gpuf(int * n, Complexf * data_d,
		Complexf * data_out_d, MPI_Comm c_comm, unsigned flags) {
	accfft_plan_gpuf *plan = new accfft_plan_gpuf;
	int nprocs, procid;
	MPI_Comm_rank(c_comm, &procid);
	plan->procid = procid;
	MPI_Cart_get(c_comm, 2, plan->np, plan->periods, plan->coord);
	plan->c_comm = c_comm;
	int *coord = plan->coord;
	MPI_Comm_split(c_comm, coord[0], coord[1], &plan->row_comm);
	MPI_Comm_split(c_comm, coord[1], coord[0], &plan->col_comm);
	plan->N[0] = n[0];
	plan->N[1] = n[1];
	plan->N[2] = n[2];
	int NX = n[0], NY = n[1], NZ = n[2];
	cufftResult_t cufft_error;

	plan->data_c = data_d;
	plan->data_out_c = data_out_d;
	if (data_out_d == data_d) {
		plan->inplace = true;
	} else {
		plan->inplace = false;
	}

	if (plan->np[1] == 1)
		plan->oneD = true;
	else
		plan->oneD = false;

	int *osize_0 = plan->osize_0, *ostart_0 = plan->ostart_0;
	int *osize_1 = plan->osize_1, *ostart_1 = plan->ostart_1;
	int *osize_2 = plan->osize_2, *ostart_2 = plan->ostart_2;
	int *osize_1i = plan->osize_1i, *ostart_1i = plan->ostart_1i;
	int *osize_2i = plan->osize_2i, *ostart_2i = plan->ostart_2i;

	int alloc_local;
	int alloc_max = 0, n_tuples = n[2] * 2;

	//int isize[3],osize[3],istart[3],ostart[3];
	alloc_max = accfft_local_size_dft_c2c_gpuf(n, plan->isize, plan->istart,
			plan->osize, plan->ostart, c_comm);
	plan->alloc_max = alloc_max;

	dfft_get_local_size_t<float>(n[0], n[1], n[2], osize_0, ostart_0, c_comm);
	dfft_get_local_size_t<float>(n[0], n[2], n[1], osize_1, ostart_1, c_comm);
	dfft_get_local_size_t<float>(n[1], n[2], n[0], osize_2, ostart_2, c_comm);

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
			cufft_error = cufftPlanMany(&plan->fplan_0, 1, &n[2], f_inembed,
					istride, idist, // *inembed, istride, idist
					f_onembed, ostride, odist, // *onembed, ostride, odist
					CUFFT_C2C, batch);
			if (cufft_error != CUFFT_SUCCESS) {
				fprintf(stderr, "CUFFT error: fplan_0 creation failed %d \n",
						cufft_error);
				return NULL;
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
			cufft_error = cufftPlanMany(&plan->fplan_1, 1, &n[1], f_inembed,
					istride, idist, // *inembed, istride, idist
					f_onembed, ostride, odist, // *onembed, ostride, odist
					CUFFT_C2C, batch);
			if (cufft_error != CUFFT_SUCCESS) {
				fprintf(stderr, "CUFFT error: fplan_1 creation failed %d \n",
						cufft_error);
				return NULL;
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
			cufft_error = cufftPlanMany(&plan->fplan_2, 1, &n[0], f_inembed,
					istride, idist, // *inembed, istride, idist
					f_onembed, ostride, odist, // *onembed, ostride, odist
					CUFFT_C2C, batch);
			if (cufft_error != CUFFT_SUCCESS) {
				fprintf(stderr, "CUFFT error: fplan_2 creation failed %d \n",
						cufft_error);
				return NULL;
			}
			//cufftSetCompatibilityMode(fplan,CUFFT_COMPATIBILITY_FFTW_PADDING); if (cudaGetLastError() != cudaSuccess){fprintf(stderr, "Cuda error:Failed at fplan cuda compatibility\n"); return;}
		}
	}

	// 1D Decomposition
	if (plan->oneD) {
		int NX = n[0], NY = n[1], NZ = n[2];

		plan->alloc_max = alloc_max;

		plan->Mem_mgr = new Mem_Mgr_gpu<float>(NX, NY, (NZ) * 2, c_comm, 1, plan->alloc_max);
		plan->T_plan_2 = new T_Plan_gpu<float>(NX, NY, (NZ) * 2, plan->Mem_mgr,
				c_comm);
		plan->T_plan_2i = new T_Plan_gpu<float>(NY, NX, NZ * 2, plan->Mem_mgr,
				c_comm);

		plan->T_plan_2->alloc_local = alloc_max;
		plan->T_plan_2i->alloc_local = alloc_max;
		plan->T_plan_1 = NULL;
		plan->T_plan_1i = NULL;

		if (flags == ACCFFT_MEASURE) {
			plan->T_plan_2->which_fast_method_gpu(plan->T_plan_2,
					(float*) data_out_d);
		} else {
			plan->T_plan_2->method = 2;
			plan->T_plan_2->kway = 2;
		}
		checkCuda_accfft(cudaDeviceSynchronize());
		MPI_Barrier(plan->c_comm);

		plan->T_plan_2i->method = -plan->T_plan_2->method;
		plan->T_plan_2i->kway = plan->T_plan_2->kway;
		plan->T_plan_2i->kway_async = plan->T_plan_2->kway_async;

	} // end 1d c2c

	// 2D Decomposition
	if (!plan->oneD) {
		// the reaseon for n_tuples/2 is to avoid splitting of imag and real parts of complex numbers
		plan->Mem_mgr = new Mem_Mgr_gpu<float>(n[1], n[2], 2, plan->row_comm,
				osize_0[0], alloc_max);
		plan->T_plan_1 = new T_Plan_gpu<float>(n[1], n[2], 2, plan->Mem_mgr,
				plan->row_comm, osize_0[0]);
		plan->T_plan_2 = new T_Plan_gpu<float>(n[0], n[1], 2 * osize_2[2],
				plan->Mem_mgr, plan->col_comm);
		plan->T_plan_2i = new T_Plan_gpu<float>(n[1], n[0], 2 * osize_2i[2],
				plan->Mem_mgr, plan->col_comm);
		plan->T_plan_1i = new T_Plan_gpu<float>(n[2], n[1], 2, plan->Mem_mgr,
				plan->row_comm, osize_1i[0]);

		plan->T_plan_1->alloc_local = plan->alloc_max;
		plan->T_plan_2->alloc_local = plan->alloc_max;
		plan->T_plan_2i->alloc_local = plan->alloc_max;
		plan->T_plan_1i->alloc_local = plan->alloc_max;

		plan->iplan_0 = -1;
		plan->iplan_1 = -1;
		plan->iplan_2 = -1;

		int coords[2], np[2], periods[2];
		MPI_Cart_get(c_comm, 2, np, periods, coords);

		if (flags == ACCFFT_MEASURE) {
			if (coords[0] == 0) {
				plan->T_plan_1->which_fast_method_gpu(plan->T_plan_1,
						(float*) data_out_d, osize_0[0]);
			}
		} else {
			plan->T_plan_1->method = 2;
			plan->T_plan_1->kway = 2;
		}

		MPI_Bcast(&plan->T_plan_1->method, 1, MPI_INT, 0, c_comm);
		MPI_Bcast(&plan->T_plan_1->kway, 1, MPI_INT, 0, c_comm);
		MPI_Bcast(&plan->T_plan_1->kway_async, 1, MPI::BOOL, 0, c_comm);
		checkCuda_accfft(cudaDeviceSynchronize());
		MPI_Barrier(plan->c_comm);

		plan->T_plan_1->method = plan->T_plan_1->method;
		plan->T_plan_2->method = plan->T_plan_1->method;
		plan->T_plan_2i->method = -plan->T_plan_1->method;
		plan->T_plan_1i->method = -plan->T_plan_1->method;

		plan->T_plan_1->kway = plan->T_plan_1->kway;
		plan->T_plan_2->kway = plan->T_plan_1->kway;
		plan->T_plan_2i->kway = plan->T_plan_1->kway;
		plan->T_plan_1i->kway = plan->T_plan_1->kway;

		plan->T_plan_1->kway_async = plan->T_plan_1->kway_async;
		plan->T_plan_2->kway_async = plan->T_plan_1->kway_async;
		plan->T_plan_2i->kway_async = plan->T_plan_1->kway_async;
		plan->T_plan_1i->kway_async = plan->T_plan_1->kway_async;

	} // end 2d c2c

	plan->c2c_plan_baked = true;
	return plan;
}

/**
 * Execute single precision C2C plan. This function is blocking and only returns after the transform is completed.
 * @note For inplace transforms, data_out should point to the same memory address as data, AND
 * the plan must have been created as inplace.
 * @param plan FFT plan created by \ref accfft_plan_dft_3d_r2cf.
 * @param data Input data in frequency domain.
 * @param data_out Output data in frequency domain.
 * @param timer See \ref timer for more details.
 * @param XYZ a bit set field that determines which directions FFT should be executed
 */
void accfft_execute_c2c_gpuf(accfft_plan_gpuf* plan, int direction,
		Complexf * data_d, Complexf * data_out_d, double * timer,
		std::bitset<3> xyz) {

	if (!plan->c2c_plan_baked) {
		if (plan->procid == 0)
			std::cout
					<< "Error. r2c plan has not been made correctly. Please first create the plan before calling execute functions."
					<< std::endl;
		return;
	}

	if (data_d == NULL)
		data_d = plan->data_c;
	if (data_out_d == NULL)
		data_out_d = plan->data_out_c;
	int * coords = plan->coord;
	int procid = plan->procid;
	double fft_time = 0;
	double timings[5] = { 0 };

	int NY = plan->N[1];
	cudaEvent_t memcpy_startEvent, memcpy_stopEvent;
	cudaEvent_t fft_startEvent, fft_stopEvent;
	checkCuda_accfft(cudaEventCreate(&memcpy_startEvent));
	checkCuda_accfft(cudaEventCreate(&memcpy_stopEvent));
	checkCuda_accfft(cudaEventCreate(&fft_startEvent));
	checkCuda_accfft(cudaEventCreate(&fft_stopEvent));
	cufftResult_t cufft_error;
	float dummy_time = 0;

	int *osize_0 = plan->osize_0, *ostart_0 = plan->ostart_0;
	int *osize_1 = plan->osize_1, *ostart_1 = plan->ostart_1;
	int *osize_2 = plan->osize_2, *ostart_2 = plan->ostart_2;
	int *osize_1i = plan->osize_1i, *ostart_1i = plan->ostart_1i;
	int *osize_2i = plan->osize_2i, *ostart_2i = plan->ostart_2i;

	if (direction == -1) {
		/**************************************************************/
		/*******************  N0/P0 x N1/P1 x N2 **********************/
		/**************************************************************/
		// FFT in Z direction
		if (xyz[2]) {
			checkCuda_accfft(cudaEventRecord(fft_startEvent, 0));
			checkCuda_accfft(
					cufftExecC2C(plan->fplan_0, (cufftComplex*) data_d,
							(cufftComplex*) data_out_d, CUFFT_FORWARD));
			checkCuda_accfft(cudaEventRecord(fft_stopEvent, 0));
			checkCuda_accfft(cudaEventSynchronize(fft_stopEvent)); // wait until fft is executed
			checkCuda_accfft(
					cudaEventElapsedTime(&dummy_time, fft_startEvent,
							fft_stopEvent));
			fft_time += dummy_time / 1000;
		} else
			data_out_d = data_d;

		if (!plan->oneD) {
			plan->T_plan_1->execute_gpu(plan->T_plan_1, (float*) data_out_d,
					timings, 2, osize_0[0], coords[0]);
		}
		checkCuda_accfft(cudaDeviceSynchronize());
		MPI_Barrier(plan->c_comm);
		/**************************************************************/
		/*******************  N0/P0 x N1 x N2/P1 **********************/
		/**************************************************************/
		if (xyz[1]) {
			checkCuda_accfft(cudaEventRecord(fft_startEvent, 0));
			for (int i = 0; i < osize_1[0]; ++i) {
				checkCuda_accfft(
						cufftExecC2C(plan->fplan_1,
								(cufftComplex*) &data_out_d[i * osize_1[1]
										* osize_1[2]],
								(cufftComplex*) &data_out_d[i * osize_1[1]
										* osize_1[2]], CUFFT_FORWARD));
			}
			checkCuda_accfft(cudaEventRecord(fft_stopEvent, 0));
			checkCuda_accfft(cudaEventSynchronize(fft_stopEvent)); // wait until fft is executed
			checkCuda_accfft(
					cudaEventElapsedTime(&dummy_time, fft_startEvent,
							fft_stopEvent));
			fft_time += dummy_time / 1000;
		}

		MPI_Barrier(plan->c_comm);

		if (plan->oneD) {
			plan->T_plan_2->execute_gpu(plan->T_plan_2, (float*) data_out_d,
					timings, 2);
		} else {
			plan->T_plan_2->execute_gpu(plan->T_plan_2, (float*) data_out_d,
					timings, 2, 1, coords[1]);
		}
		MPI_Barrier(plan->c_comm);
		/**************************************************************/
		/*******************  N0 x N1/P0 x N2/P1 **********************/
		/**************************************************************/
		if (xyz[0]) {
			checkCuda_accfft(cudaEventRecord(fft_startEvent, 0));
			checkCuda_accfft(
					cufftExecC2C(plan->fplan_2, (cufftComplex*) data_out_d,
							(cufftComplex*) data_out_d, CUFFT_FORWARD));
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
					cufftExecC2C(plan->fplan_2, (cufftComplex*) data_d,
							(cufftComplex*) data_d, CUFFT_INVERSE));
			checkCuda_accfft(cudaDeviceSynchronize());
			checkCuda_accfft(cudaEventRecord(fft_stopEvent, 0));
			checkCuda_accfft(cudaEventSynchronize(fft_stopEvent)); // wait until fft is executed
			checkCuda_accfft(
					cudaEventElapsedTime(&dummy_time, fft_startEvent,
							fft_stopEvent));
			fft_time += dummy_time / 1000;
			MPI_Barrier(plan->c_comm);
		}

		if (plan->oneD) {
			plan->T_plan_2i->execute_gpu(plan->T_plan_2i, (float*) data_d,
					timings, 1);
		} else {
			plan->T_plan_2i->execute_gpu(plan->T_plan_2i, (float*) data_d,
					timings, 1, 1, coords[1]);
		}
		MPI_Barrier(plan->c_comm);
		/**************************************************************/
		/*******************  N0/P0 x N1 x N2/P1 **********************/
		/**************************************************************/
		if (xyz[1]) {
			checkCuda_accfft(cudaEventRecord(fft_startEvent, 0));
			for (int i = 0; i < osize_1i[0]; ++i) {
				checkCuda_accfft(
						cufftExecC2C(plan->fplan_1,
								(cufftComplex*) &data_d[i * NY * osize_1i[2]],
								(cufftComplex*) &data_d[i * NY * osize_1i[2]],
								CUFFT_INVERSE));
			}
			checkCuda_accfft(cudaEventRecord(fft_stopEvent, 0));
			checkCuda_accfft(cudaEventSynchronize(fft_stopEvent)); // wait until fft is executed
			checkCuda_accfft(
					cudaEventElapsedTime(&dummy_time, fft_startEvent,
							fft_stopEvent));
			fft_time += dummy_time / 1000;
		}
		MPI_Barrier(plan->c_comm);

		if (!plan->oneD) {
			plan->T_plan_1i->execute_gpu(plan->T_plan_1i, (float*) data_d,
					timings, 1, osize_1i[0], coords[0]);
		}
		MPI_Barrier(plan->c_comm);
		/**************************************************************/
		/*******************  N0/P0 x N1/P1 x N2 **********************/
		/**************************************************************/

		if (xyz[2]) {
			checkCuda_accfft(cudaEventRecord(fft_startEvent, 0));
			checkCuda_accfft(
					cufftExecC2C(plan->fplan_0, (cufftComplex*) data_d,
							(cufftComplex*) data_out_d, CUFFT_INVERSE));
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
	MPI_Barrier(plan->c_comm);

	return;
}

/**
 * Destroy single precision AccFFT CPU plan. This function calls \ref accfft_destroy_plan_gpu.
 * @param plan Input plan to be destroyed.
 */
void accfft_destroy_plan(accfft_plan_gpuf * plan) {
	return (accfft_destroy_plan_gpu(plan));
}

/**
 * Destroy single precision AccFFT GPU plan.
 * @param plan Input plan to be destroyed.
 */
void accfft_destroy_plan_gpu(accfft_plan_gpuf * plan) {

	if (plan->T_plan_1 != NULL)
		delete (plan->T_plan_1);
	if (plan->T_plan_1i != NULL)
		delete (plan->T_plan_1i);
	if (plan->T_plan_2 != NULL)
		delete (plan->T_plan_2);
	if (plan->T_plan_2i != NULL)
		delete (plan->T_plan_2i);
	if (plan->Mem_mgr != NULL)
		delete (plan->Mem_mgr);
	if (plan->fplan_x != -1)
		cufftDestroy(plan->fplan_x);
	if (plan->fplan_y != -1)
		cufftDestroy(plan->fplan_y);
	if (plan->iplan_x != -1)
		cufftDestroy(plan->iplan_x);
	if (plan->iplan_y != -1)
		cufftDestroy(plan->iplan_y);
	if (plan->T_plan_y != NULL)
		delete (plan->T_plan_y);
	if (plan->T_plan_yi != NULL)
		delete (plan->T_plan_yi);

	if (plan->fplan_0 != -1)
		cufftDestroy(plan->fplan_0);
	if (plan->fplan_1 != -1)
		cufftDestroy(plan->fplan_1);
	if (plan->fplan_2 != -1)
		cufftDestroy(plan->fplan_2);

	if (plan->iplan_0 != -1)
		cufftDestroy(plan->iplan_0);
	//if(plan->iplan_1!=-1)cufftDestroy(plan->iplan_1);
	//if(plan->iplan_2!=-1)cufftDestroy(plan->iplan_2);

	MPI_Comm_free(&plan->row_comm);
	MPI_Comm_free(&plan->col_comm);
	return;
}

template<typename T, typename Tc>
void accfft_execute_r2c_gpu_t(accfft_plan_gpuf* plan, T* data, Tc* data_out,
		double * timer, std::bitset<3> XYZ) {
	accfft_execute_r2c_gpuf(plan, data, data_out, timer, XYZ);
	return;
}
template<typename Tc, typename T>
void accfft_execute_c2r_gpu_t(accfft_plan_gpuf* plan, Tc* data, T* data_out,
		double * timer, std::bitset<3> XYZ) {
	accfft_execute_c2r_gpuf(plan, data, data_out, timer, XYZ);
	return;
}
template void accfft_execute_r2c_gpu_t<float, Complexf>(accfft_plan_gpuf* plan,
		float* data, Complexf* data_out, double * timer, std::bitset<3> XYZ);
template void accfft_execute_c2r_gpu_t<Complexf, float>(accfft_plan_gpuf* plan,
		Complexf* data, float* data_out, double * timer, std::bitset<3> XYZ);

// templates for execution only in z direction
void accfft_execute_z_gpuf(accfft_plan_gpuf* plan, int direction, float * data_d,
		float * data_out_d, double * timer) {

	if (data_d == NULL)
		data_d = plan->data;
	if (data_out_d == NULL)
		data_out_d = plan->data_out;

	int * coords = plan->coord;
	int procid = plan->procid;
	double fft_time = 0;
	double timings[5] = { 0 };

	cudaEvent_t fft_startEvent, fft_stopEvent;
	checkCuda_accfft(cudaEventCreate(&fft_startEvent));
	checkCuda_accfft(cudaEventCreate(&fft_stopEvent));
	int NY = plan->N[1];
	float dummy_time = 0;

  if (direction == -1) {
    /**************************************************************/
    /*******************  N0/P0 x N1/P1 x N2 **********************/
    /**************************************************************/
    // FFT in Z direction
    checkCuda_accfft(cudaEventRecord(fft_startEvent, 0));
    checkCuda_accfft(
        cufftExecR2C(plan->fplan_0, (cufftReal*) data_d,
          (cufftComplex*) data_out_d));
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
        cufftExecC2R(plan->iplan_0, (cufftComplex*) data_d,
          (cufftReal*) data_out_d));
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
  MPI_Barrier(plan->c_comm);

  return;
}

template<typename T, typename Tc>
void accfft_execute_r2c_z_gpu_t(accfft_plan_gpuf* plan, T* data, Tc* data_out,
    double * timer) {
	if (plan->r2c_plan_baked) {
		accfft_execute_z_gpuf(plan, -1, data, (float*) data_out, timer);
	} else {
		if (plan->procid == 0)
			std::cout
					<< "Error. r2c plan has not been made correctly. Please first create the plan before calling execute functions."
					<< std::endl;
	}
  return;
}

template<typename Tc, typename T>
void accfft_execute_c2r_z_gpu_t(accfft_plan_gpuf* plan, Tc* data, T* data_out,
    double * timer) {
	if (plan->r2c_plan_baked) {
		accfft_execute_z_gpuf(plan, 1, (float*) data, data_out, timer);
	} else {
		if (plan->procid == 0)
			std::cout
					<< "Error. r2c plan has not been made correctly. Please first create the plan before calling execute functions."
					<< std::endl;
	}
  return;
}

template void accfft_execute_r2c_z_gpu_t<float, Complexf>(accfft_plan_gpuf* plan,
    float* data, Complexf* data_out, double * timer);
template void accfft_execute_c2r_z_gpu_t<Complexf, float>(accfft_plan_gpuf* plan,
    Complexf* data, float* data_out, double * timer);

// templates for execution only in y direction
void accfft_execute_y_gpuf(accfft_plan_gpuf* plan, int direction, float * data_d,
		float * data_out_d, double * timer) {

	if (data_d == NULL)
		data_d = plan->data;
	if (data_out_d == NULL)
		data_out_d = plan->data_out;

	int * coords = plan->coord;
	int procid = plan->procid;
	double fft_time = 0;
	double timings[5] = { 0 };

	cudaEvent_t memcpy_startEvent, memcpy_stopEvent;
	cudaEvent_t fft_startEvent, fft_stopEvent;
	checkCuda_accfft(cudaEventCreate(&memcpy_startEvent));
	checkCuda_accfft(cudaEventCreate(&memcpy_stopEvent));
	checkCuda_accfft(cudaEventCreate(&fft_startEvent));
	checkCuda_accfft(cudaEventCreate(&fft_stopEvent));
	int NY = plan->N[1];
	float dummy_time = 0;
  int64_t alloc_max = plan->alloc_max;

	int *osize_0 = plan->osize_0; // *ostart_0 =plan->ostart_0;
	int *osize_1 = plan->osize_1; // *ostart_1 =plan->ostart_1;
	//int *osize_2 =plan->osize_2, *ostart_2 =plan->ostart_2;
	int *osize_1i = plan->osize_1i;  //*ostart_1i=plan->ostart_1i;
	//int *osize_2i=plan->osize_2i,*ostart_2i=plan->ostart_2i;
  int *osize_y = plan->osize_y;
  int *osize_yi = plan->osize_yi;
  int64_t N_local = plan->isize[0] * plan->isize[1] * plan->isize[2];
  float* cwork_d = plan->Mem_mgr->buffer_d3;

	if (direction == -1) {
		/**************************************************************/
		/*******************  N0/P0 x N1/P1 x N2 **********************/
		/**************************************************************/
    cudaMemcpy(cwork_d, data_d, alloc_max, cudaMemcpyDeviceToDevice);
		if (!plan->oneD) {
			plan->T_plan_y->execute_gpu(plan->T_plan_y, cwork_d, timings, 2,
					osize_y[0], coords[0]);
		}
		/**************************************************************/
		/*******************  N0/P0 x N1 x N2/P1 **********************/
		/**************************************************************/
			checkCuda_accfft(cudaEventRecord(fft_startEvent, 0));
    for (int i = 0; i < plan->osize_y[0]; ++i) {
      checkCuda_accfft(
          cufftExecR2C(plan->fplan_y,
            (cufftReal*) &cwork_d[i
            * osize_y[1] * osize_y[2]],
            (cufftComplex*) &data_out_d[2 * i
            * osize_yi[1] * osize_y[2]]));
    }
			checkCuda_accfft(cudaEventRecord(fft_stopEvent, 0));
			checkCuda_accfft(cudaEventSynchronize(fft_stopEvent)); // wait until fft is executed
			checkCuda_accfft(
					cudaEventElapsedTime(&dummy_time, fft_startEvent,
							fft_stopEvent));
			fft_time += dummy_time / 1000;
			MPI_Barrier(plan->c_comm);
	} else if (direction == 1) {
		/**************************************************************/
		/*******************  N0/P0 x N1 x N2/P1 **********************/
		/**************************************************************/
			checkCuda_accfft(cudaEventRecord(fft_startEvent, 0));
    for (int i = 0; i < osize_yi[0]; ++i) {
      checkCuda_accfft(
          cufftExecC2R(plan->iplan_y,
            (cufftComplex*) &data_d[2 * i * osize_yi[1]
            * osize_yi[2]],
            (cufftReal*) &cwork_d[i * osize_y[1]
            * osize_yi[2]]));
    }
			checkCuda_accfft(cudaEventRecord(fft_stopEvent, 0));
			checkCuda_accfft(cudaEventSynchronize(fft_stopEvent)); // wait until fft is executed
			checkCuda_accfft(
					cudaEventElapsedTime(&dummy_time, fft_startEvent,
							fft_stopEvent));
			fft_time += dummy_time / 1000;
			MPI_Barrier(plan->c_comm);

		if (!plan->oneD) {
			plan->T_plan_yi->execute_gpu(plan->T_plan_yi, (float*) cwork_d,
					timings, 1, osize_yi[0], coords[0]);
		}
		MPI_Barrier(plan->c_comm);
    cudaMemcpy(data_out_d, cwork_d, N_local * sizeof(float), cudaMemcpyDeviceToDevice);
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
	MPI_Barrier(plan->c_comm);

	return;
}

template<typename T, typename Tc>
void accfft_execute_r2c_y_gpu_t(accfft_plan_gpuf* plan, T* data, Tc* data_out,
    double * timer) {
	if (plan->r2c_plan_baked) {
		accfft_execute_y_gpuf(plan, -1, data, (float*) data_out, timer);
	} else {
		if (plan->procid == 0)
			std::cout
					<< "Error. r2c plan has not been made correctly. Please first create the plan before calling execute functions."
					<< std::endl;
	}
  return;
}

template<typename Tc, typename T>
void accfft_execute_c2r_y_gpu_t(accfft_plan_gpuf* plan, Tc* data, T* data_out,
    double * timer) {
	if (plan->r2c_plan_baked) {
		accfft_execute_y_gpuf(plan, 1, (float*) data, data_out, timer);
	} else {
		if (plan->procid == 0)
			std::cout
					<< "Error. r2c plan has not been made correctly. Please first create the plan before calling execute functions."
					<< std::endl;
	}
  return;
}

template void accfft_execute_r2c_y_gpu_t<float, Complexf>(accfft_plan_gpuf* plan,
    float* data, Complexf* data_out, double * timer);
template void accfft_execute_c2r_y_gpu_t<Complexf, float>(accfft_plan_gpuf* plan,
    Complexf* data, float* data_out, double * timer);
// templates for execution only in x direction
void accfft_execute_x_gpuf(accfft_plan_gpuf* plan, int direction, float * data_d,
		float * data_out_d, double * timer) {

	if (data_d == NULL)
		data_d = plan->data;
	if (data_out_d == NULL)
		data_out_d = plan->data_out;

	int * coords = plan->coord;
	int procid = plan->procid;
	double fft_time = 0;
	double timings[5] = { 0 };

	cudaEvent_t memcpy_startEvent, memcpy_stopEvent;
	cudaEvent_t fft_startEvent, fft_stopEvent;
	checkCuda_accfft(cudaEventCreate(&memcpy_startEvent));
	checkCuda_accfft(cudaEventCreate(&memcpy_stopEvent));
	checkCuda_accfft(cudaEventCreate(&fft_startEvent));
	checkCuda_accfft(cudaEventCreate(&fft_stopEvent));
	int NY = plan->N[1];
	float dummy_time = 0;
  int64_t alloc_max = plan->alloc_max;

	int *osize_0 = plan->osize_0; // *ostart_0 =plan->ostart_0;
	int *osize_1 = plan->osize_1; // *ostart_1 =plan->ostart_1;
	//int *osize_2 =plan->osize_2, *ostart_2 =plan->ostart_2;
	int *osize_1i = plan->osize_1i;  //*ostart_1i=plan->ostart_1i;
	//int *osize_2i=plan->osize_2i,*ostart_2i=plan->ostart_2i;
  int *osize_y = plan->osize_y;
  int *osize_yi = plan->osize_yi;
  int *osize_x = plan->osize_x;
  int *osize_xi = plan->osize_xi;
  int64_t N_local = plan->isize[0] * plan->isize[1] * plan->isize[2];
  float* cwork_d = plan->Mem_mgr->buffer_d3;

	if (direction == -1) {
		/**************************************************************/
		/*******************  N0/P0 x N1/P1 x N2 **********************/
		/**************************************************************/
    cudaMemcpy(cwork_d, data_d, N_local * sizeof(float), cudaMemcpyDeviceToDevice);
		plan->T_plan_x->execute_gpu(plan->T_plan_x, cwork_d, timings, 2);
		/**************************************************************/
		/*******************  N0/P0 x N1 x N2/P1 **********************/
		/**************************************************************/
			checkCuda_accfft(cudaEventRecord(fft_startEvent, 0));
    checkCuda_accfft(
        cufftExecR2C(plan->fplan_x, (cufftReal*) cwork_d,
          (cufftComplex*) data_out_d));
			checkCuda_accfft(cudaEventRecord(fft_stopEvent, 0));
			checkCuda_accfft(cudaEventSynchronize(fft_stopEvent)); // wait until fft is executed
			checkCuda_accfft(
					cudaEventElapsedTime(&dummy_time, fft_startEvent,
							fft_stopEvent));
			fft_time += dummy_time / 1000;
			MPI_Barrier(plan->c_comm);

	} else if (direction == 1) {
		/**************************************************************/
		/*******************  N0/P0 x N1 x N2/P1 **********************/
		/**************************************************************/
			checkCuda_accfft(cudaEventRecord(fft_startEvent, 0));
    checkCuda_accfft(
        cufftExecC2R(plan->iplan_x, (cufftComplex*) data_d,
          (cufftReal*) data_d));
			checkCuda_accfft(cudaEventRecord(fft_stopEvent, 0));
			checkCuda_accfft(cudaEventSynchronize(fft_stopEvent)); // wait until fft is executed
			checkCuda_accfft(
					cudaEventElapsedTime(&dummy_time, fft_startEvent,
							fft_stopEvent));
			fft_time += dummy_time / 1000;
			MPI_Barrier(plan->c_comm);

		plan->T_plan_xi->execute_gpu(plan->T_plan_xi, data_d, timings, 1);
		MPI_Barrier(plan->c_comm);
    cudaMemcpy(data_out_d, data_d, N_local * sizeof(float), cudaMemcpyDeviceToDevice);
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
	MPI_Barrier(plan->c_comm);

	return;
}

template<typename T, typename Tc>
void accfft_execute_r2c_x_gpu_t(accfft_plan_gpuf* plan, T* data, Tc* data_out,
    double * timer) {
	if (plan->r2c_plan_baked) {
		accfft_execute_x_gpuf(plan, -1, data, (float*) data_out, timer);
	} else {
		if (plan->procid == 0)
			std::cout
					<< "Error. r2c plan has not been made correctly. Please first create the plan before calling execute functions."
					<< std::endl;
	}
  return;
}

template<typename Tc, typename T>
void accfft_execute_c2r_x_gpu_t(accfft_plan_gpuf* plan, Tc* data, T* data_out,
    double * timer) {
	if (plan->r2c_plan_baked) {
		accfft_execute_x_gpuf(plan, 1, (float*) data, data_out, timer);
	} else {
		if (plan->procid == 0)
			std::cout
					<< "Error. r2c plan has not been made correctly. Please first create the plan before calling execute functions."
					<< std::endl;
	}
  return;
}

template void accfft_execute_r2c_x_gpu_t<float, Complexf>(accfft_plan_gpuf* plan,
    float* data, Complexf* data_out, double * timer);
template void accfft_execute_c2r_x_gpu_t<Complexf, float>(accfft_plan_gpuf* plan,
    Complexf* data, float* data_out, double * timer);
