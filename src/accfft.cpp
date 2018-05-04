/**
 * @file
 * CPU functions of AccFFT
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
#include <cstdlib>
#include "accfft.h"
#include "accfft_common.h"
#include "dtypes.h"
#ifdef ACCFFT_MKL
#include "mkl.h"
#endif
#define VERBOSE 0


/**
 * Get the local sizes of the distributed global data for a R2C transform
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
int accfft_local_size_dft_r2c(int * n, int * isize, int * istart, int * osize,
		int *ostart, MPI_Comm c_comm) {
  return accfft_local_size_dft_r2c_t<double>(n, isize, istart, osize, ostart, c_comm);
}

/**
 * Creates a 3D R2C parallel FFT plan.If data_out point to the same location as the input
 * data, then an inplace plan will be created. Otherwise the plan would be outplace.
 * @param n Integer array of size 3, corresponding to the global data size
 * @param data Input data in spatial domain
 * @param data_out Output data in frequency domain
 * @param c_comm Cartesian communicator returned by \ref accfft_create_comm
 * @param flags AccFFT flags, See \ref flags for more details.
 * @return
 */
accfft_plan* accfft_plan_dft_3d_r2c(int * n, double * data, double * data_out,
		MPI_Comm c_comm, unsigned flags) {
	accfft_plan *plan = new accfft_plan;
	int nprocs, procid;
	MPI_Comm_rank(c_comm, &procid);
	MPI_Comm_size(c_comm, &nprocs);
	plan->procid = procid;
	MPI_Cart_get(c_comm, 2, plan->np, plan->periods, plan->coord);
	plan->c_comm = c_comm;
	int *coord = plan->coord;
	MPI_Comm_split(c_comm, coord[0], coord[1], &plan->row_comm);
	MPI_Comm_split(c_comm, coord[1], coord[0], &plan->col_comm);

	plan->N[0] = n[0];
	plan->N[1] = n[1];
	plan->N[2] = n[2];

	plan->data = data;
	plan->data_out = data_out;
	if (data_out == data) {
		plan->inplace = true;
	} else {
		plan->inplace = false;
	}

	unsigned fftw_flags;
	if (flags == ACCFFT_ESTIMATE)
		fftw_flags = FFTW_ESTIMATE;
	else
		fftw_flags = FFTW_MEASURE;

	if (plan->np[1] == 1)
		plan->oneD = true;
	else
		plan->oneD = false;

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

	int alloc_local;
	int alloc_max = 0;
	int n_tuples_o, n_tuples_i;
	plan->inplace == true ? n_tuples_i = (n[2] / 2 + 1) * 2 : n_tuples_i = n[2];
	n_tuples_o = (n[2] / 2 + 1) * 2;

	//int isize[3],osize[3],istart[3],ostart[3];
	alloc_max = accfft_local_size_dft_r2c_t<double>(n, plan->isize, plan->istart,
			plan->osize, plan->ostart, c_comm);
	plan->alloc_max = alloc_max;

	dfft_get_local_size_t<double>(n[0], n[1], n_tuples_o / 2, osize_0, ostart_0, c_comm);
	dfft_get_local_size_t<double>(n[0], n_tuples_o / 2, n[1], osize_1, ostart_1, c_comm);
	dfft_get_local_size_t<double>(n[1], n_tuples_o / 2, n[0], osize_2, ostart_2, c_comm);

	std::swap(osize_1[1], osize_1[2]);
	std::swap(ostart_1[1], ostart_1[2]);

	std::swap(ostart_2[1], ostart_2[2]);
	std::swap(ostart_2[0], ostart_2[1]);
	std::swap(osize_2[1], osize_2[2]);
	std::swap(osize_2[0], osize_2[1]);

  // osize_y is the configuration after y transpose
	dfft_get_local_size_t<double>(n[0], n[2], n[1], osize_y, ostart_y, c_comm);
	std::swap(osize_y[1], osize_y[2]);
	std::swap(ostart_y[1], ostart_y[2]);
  osize_yi[0] = osize_y[0];
  osize_yi[1] = osize_y[1] / 2 + 1;
  osize_yi[2] = osize_y[2];

	std::swap(osize_y[1], osize_y[2]); // N0/P0 x N2/P1 x N1
	std::swap(osize_yi[1], osize_yi[2]);


  // for x only fft. The strategy is to divide (N1/P1 x N2) by P0 completely.
  // So we treat the last size to be just 1.
	dfft_get_local_size_t<double>(isize[1] * n[2], n[2], n[0], osize_x, ostart_x, c_comm);
  osize_x[1] = osize_x[0];
  osize_x[0] = osize_x[2];
  osize_x[2] = 1;
  std::swap(osize_x[0], osize_x[1]); // switch to (N1/P1xN2 )/P0 x N0 x 1

	dfft_get_local_size_t<double>(isize[1] * n[2], n[2], n[0] / 2 + 1, osize_xi, ostart_x, c_comm);
  osize_xi[1] = osize_xi[0];
  osize_xi[0] = osize_xi[2];
  osize_xi[2] = 1;
  std::swap(osize_xi[0], osize_xi[1]); // switch to (N1/P1xN2 )/P0 x N0 x 1

  ostart_x[0] = 0;
  ostart_x[1] = -1<<8; // starts have no meaning in this approach
  ostart_x[2] = -1<<8;

	for (int i = 0; i < 3; i++) {
		osize_1i[i] = osize_1[i];
		osize_2i[i] = osize_2[i];
		ostart_1i[i] = ostart_1[i];
		ostart_2i[i] = ostart_2[i];
	}

	// FFT Plans
	{
    double* dummy_data = (double*) accfft_alloc(plan->alloc_max);
		plan->fplan_0 = fftw_plan_many_dft_r2c(1, &n[2],
				osize_0[0] * osize_0[1], //int rank, const int *n, int howmany
				data, NULL,         //double *in, const int *inembed,
				1, n_tuples_i,      //int istride, int idist,
				(fftw_complex*) data_out, NULL, //fftw_complex *out, const int *onembed,
				1, n_tuples_o / 2,        // int ostride, int odist,
				fftw_flags);
		if (plan->fplan_0 == NULL)
			std::cout << "!!! fplan_0 not created in r2c plan!!!" << std::endl;

		plan->iplan_0 = fftw_plan_many_dft_c2r(1, &n[2],
				osize_0[0] * osize_0[1], //int rank, const int *n, int howmany
				(fftw_complex*) data_out, NULL, //double *in, const int *inembed,
				1, n_tuples_o / 2,      //int istride, int idist,
				data, NULL, //fftw_complex *out, const int *onembed,
				1, n_tuples_i,        // int ostride, int odist,
				fftw_flags);
		if (plan->iplan_0 == NULL)
			std::cout << "!!! iplan_0 not created in r2c plan!!!" << std::endl;

		// ---- fplan_1
		fftw_iodim dims, howmany_dims[2];
		dims.n = osize_1[1];
		dims.is = osize_1[2];
		dims.os = osize_1[2];

		howmany_dims[0].n = osize_1[2];
		howmany_dims[0].is = 1;
		howmany_dims[0].os = 1;

		howmany_dims[1].n = osize_1[0];
		howmany_dims[1].is = osize_1[1] * osize_1[2];
		howmany_dims[1].os = osize_1[1] * osize_1[2];
#ifdef ACCFFT_MKL
		plan->fplan_1 = fftw_plan_many_dft(1, &n[1], osize_1[2] * osize_1[0], //int rank, const int *n, int howmany
		(fftw_complex*) data_out, NULL,        //double *in, const int *inembed,
				1, n[1],      //int istride, int idist,
				(fftw_complex*) data_out, NULL, //fftw_complex *out, const int *onembed,
				1, n[1],        // int ostride, int odist,
				FFTW_FORWARD, fftw_flags);
		if (plan->fplan_1 == NULL)
			std::cout << "!!! fplan1 not created in r2c plan !!!" << std::endl;

		plan->iplan_1 = fftw_plan_many_dft(1, &n[1], osize_1[2] * osize_1[0], //int rank, const int *n, int howmany
		(fftw_complex*) data_out, NULL,        //double *in, const int *inembed,
				1, n[1],      //int istride, int idist,
				(fftw_complex*) data_out, NULL, //fftw_complex *out, const int *onembed,
				1, n[1],        // int ostride, int odist,
				FFTW_BACKWARD, fftw_flags);
		if (plan->iplan_1 == NULL)
			std::cout << "!!! iplan1 not created in r2c plan !!!" << std::endl;

		//plan->fplan_1 = fftw_plan_guru_dft(1, &dims, 1, howmany_dims,
		//		(fftw_complex*) data_out, (fftw_complex*) data_out, -1,
		//		fftw_flags);
		//if (plan->fplan_1 == NULL)
		//	std::cout << "!!! fplan1 not created in r2c plan!!!" << std::endl;
		//plan->iplan_1 = fftw_plan_guru_dft(1, &dims, 1, howmany_dims,
		//		(fftw_complex*) data_out, (fftw_complex*) data_out, 1,
		//		fftw_flags);
		//if (plan->iplan_1 == NULL)
		//	std::cout << "!!! iplan1 not created in r2c plan !!!" << std::endl;
#else
		plan->fplan_1 = fftw_plan_guru_dft(1, &dims, 2, howmany_dims,
				(fftw_complex*) data_out, (fftw_complex*) data_out, -1,
				fftw_flags);
		if (plan->fplan_1 == NULL)
			std::cout << "!!! fplan1 not created in r2c plan!!!" << std::endl;
		plan->iplan_1 = fftw_plan_guru_dft(1, &dims, 2, howmany_dims,
				(fftw_complex*) data_out, (fftw_complex*) data_out, 1,
				fftw_flags);
		if (plan->iplan_1 == NULL)
			std::cout << "!!! iplan1 not created in r2c plan !!!" << std::endl;
#endif

		// ---fplan_y

    // PCOUT << "osize_y[0] = " << osize_y[0] << " [1] = " << osize_y[1] << " [2] = " << osize_y[2] << std::endl;
		plan->fplan_y = fftw_plan_many_dft_r2c(1, &n[1],
				osize_y[0]*osize_y[1], //int rank, const int *n, int howmany
				dummy_data, NULL,         //double *in, const int *inembed,
				1, n[1],      //int istride, int idist,
				(fftw_complex*) data_out, NULL, //fftw_complex *out, const int *onembed,
				1,n[1]/2+1,        // int ostride, int odist,
				fftw_flags);
		if (plan->fplan_y == NULL)
			std::cout << "!!! fplan_y not created in r2c plan!!!" << std::endl;
		plan->iplan_y = fftw_plan_many_dft_c2r(1, &n[1],
				osize_y[0]*osize_y[1], //int rank, const int *n, int howmany
				(fftw_complex*) data_out, NULL, //double *in, const int *inembed,
				1, n[1]/2+1,      //int istride, int idist,
				data_out, NULL, //fftw_complex *out, const int *onembed,
			  1, n[1],        // int ostride, int odist,
				fftw_flags);
		if (plan->iplan_y == NULL)
			std::cout << "!!! iplan_y not created in r2c plan!!!" << std::endl;
		//plan->fplan_y = fftw_plan_guru_dft_r2c(1, &dims, 2, howmany_dims,
		//		(double*) dummy_data, (fftw_complex*) data_out,
		//		fftw_flags);
		//if (plan->fplan_y == NULL)
		//	std::cout << "!!! fplan_y not created in r2c plan!!!" << std::endl;

#ifdef ACCFFT_MKL
		plan->fplan_2 = fftw_plan_many_dft(1, &n[0], osize_2[2] * osize_2[1], //int rank, const int *n, int howmany
		(fftw_complex*) data_out, NULL,        //double *in, const int *inembed,
				1,n[0],      //int istride, int idist,
				(fftw_complex*) data_out, NULL, //fftw_complex *out, const int *onembed,
				1,n[0],        // int ostride, int odist,
				FFTW_FORWARD, fftw_flags);
		if (plan->fplan_2 == NULL)
			std::cout << "!!! fplan2 not created in r2c plan !!!" << std::endl;
		plan->iplan_2 = fftw_plan_many_dft(1, &n[0], osize_2[2] * osize_2[1], //int rank, const int *n, int howmany
		(fftw_complex*) data_out, NULL,        //double *in, const int *inembed,
				1,n[0],      //int istride, int idist,
				(fftw_complex*) data_out, NULL, //fftw_complex *out, const int *onembed,
				1,n[0],        // int ostride, int odist,
				FFTW_BACKWARD, fftw_flags);
		if (plan->iplan_2 == NULL)
			std::cout << "!!! iplan2 not created in r2c plan !!!" << std::endl;

		//plan->iplan_2 = fftw_plan_many_dft(1, &n[0], osize_2[2] * osize_2[1], //int rank, const int *n, int howmany
		//(fftw_complex*) data_out, NULL,        //double *in, const int *inembed,
		//		1,n[0]      //int istride, int idist,
		//		(fftw_complex*) data_out, NULL, //fftw_complex *out, const int *onembed,
		//		1,n[0]        // int ostride, int odist,
		//		FFTW_BACKWARD, fftw_flags);
		//if (plan->iplan_2 == NULL)
		//	std::cout << "!!! iplan2 not created in r2c plan !!!" << std::endl;
		//plan->fplan_2 = fftw_plan_many_dft(1, &n[0], osize_2[2] * osize_2[1], //int rank, const int *n, int howmany
		//(fftw_complex*) data_out, NULL,        //double *in, const int *inembed,
		//		osize_2[2] * osize_2[1], 1,      //int istride, int idist,
		//		(fftw_complex*) data_out, NULL, //fftw_complex *out, const int *onembed,
		//		osize_2[2] * osize_2[1], 1,        // int ostride, int odist,
		//		FFTW_FORWARD, fftw_flags);
		//if (plan->fplan_2 == NULL)
		//	std::cout << "!!! fplan2 not created in r2c plan !!!" << std::endl;
		//plan->iplan_2 = fftw_plan_many_dft(1, &n[0], osize_2[2] * osize_2[1], //int rank, const int *n, int howmany
		//(fftw_complex*) data_out, NULL,        //double *in, const int *inembed,
		//		osize_2[2] * osize_2[1], 1,      //int istride, int idist,
		//		(fftw_complex*) data_out, NULL, //fftw_complex *out, const int *onembed,
		//		osize_2[2] * osize_2[1], 1,        // int ostride, int odist,
		//		FFTW_BACKWARD, fftw_flags);
		//if (plan->iplan_2 == NULL)
		//	std::cout << "!!! iplan2 not created in r2c plan !!!" << std::endl;
#else
		plan->fplan_2 = fftw_plan_many_dft(1, &n[0], osize_2[2] * osize_2[1], //int rank, const int *n, int howmany
		(fftw_complex*) data_out, NULL,        //double *in, const int *inembed,
				osize_2[2] * osize_2[1], 1,      //int istride, int idist,
				(fftw_complex*) data_out, NULL, //fftw_complex *out, const int *onembed,
				osize_2[2] * osize_2[1], 1,        // int ostride, int odist,
				FFTW_FORWARD, fftw_flags);
		if (plan->fplan_2 == NULL)
			std::cout << "!!! fplan2 not created in r2c plan !!!" << std::endl;

		plan->iplan_2 = fftw_plan_many_dft(1, &n[0], osize_2[2] * osize_2[1], //int rank, const int *n, int howmany
		(fftw_complex*) data_out, NULL,        //double *in, const int *inembed,
				osize_2[2] * osize_2[1], 1,      //int istride, int idist,
				(fftw_complex*) data_out, NULL, //fftw_complex *out, const int *onembed,
				osize_2[2] * osize_2[1], 1,        // int ostride, int odist,
				FFTW_BACKWARD, fftw_flags);
		if (plan->iplan_2 == NULL)
			std::cout << "!!! iplan2 not created in r2c plan !!!" << std::endl;
#endif
    // fplan_x
		plan->fplan_x = fftw_plan_many_dft_r2c(1, &n[0],
				osize_x[0], //int rank, const int *n, int howmany
				dummy_data, NULL,         //double *in, const int *inembed,
				1, n[0],      //int istride, int idist,
				(fftw_complex*) data_out, NULL, //fftw_complex *out, const int *onembed,
				1,n[0]/2+1,        // int ostride, int odist,
				fftw_flags);
		if (plan->fplan_x == NULL)
			std::cout << "!!! fplan_x not created in r2c plan!!!" << std::endl;
		plan->iplan_x = fftw_plan_many_dft_c2r(1, &n[0],
				osize_x[0], //int rank, const int *n, int howmany
				(fftw_complex*) data_out, NULL, //double *in, const int *inembed,
				1, n[0]/2+1,      //int istride, int idist,
				data_out, NULL, //fftw_complex *out, const int *onembed,
			  1, n[0],        // int ostride, int odist,
				fftw_flags);
		if (plan->iplan_x == NULL)
			std::cout << "!!! iplan_x not created in r2c plan!!!" << std::endl;

    //
		//plan->fplan_x = fftw_plan_many_dft_r2c(1, &n[0],
		//		osize_x[0] * osize_x[2], //int rank, const int *n, int howmany
		//		dummy_data, NULL,         //double *in, const int *inembed,
		//		osize_x[0] * osize_x[2], 1,      //int istride, int idist,
		//		(fftw_complex*) data_out, NULL, //fftw_complex *out, const int *onembed,
		//		osize_xi[0] * osize_xi[2], 1,        // int ostride, int odist,
		//		fftw_flags);
		//if (plan->fplan_x == NULL)
		//	std::cout << "!!! fplan_x not created in r2c plan!!!" << std::endl;

		//plan->iplan_x = fftw_plan_many_dft_c2r(1, &n[0],
		//		osize_xi[0] * osize_xi[2], //int rank, const int *n, int howmany
		//		(fftw_complex*) data_out, NULL, //double *in, const int *inembed,
		//		osize_xi[0] * osize_xi[2], 1,      //int istride, int idist,
		//		data_out, NULL, //fftw_complex *out, const int *onembed,
		//	  osize_x[0] * osize_x[2], 1,        // int ostride, int odist,
		//		fftw_flags);
		//if (plan->iplan_x == NULL)
		//	std::cout << "!!! iplan_x not created in r2c plan!!!" << std::endl;
    accfft_free(dummy_data);
	}

	// 1D Decomposition
	if (plan->oneD) {
		plan->Mem_mgr = new Mem_Mgr<double>(n[0], n[1], n_tuples_o, c_comm, 1, plan->alloc_max);
		plan->T_plan_2 = new T_Plan<double>(n[0], n[1], n_tuples_o,
				plan->Mem_mgr, c_comm);
		plan->T_plan_2i = new T_Plan<double>(n[1], n[0], n_tuples_o,
				plan->Mem_mgr, c_comm);
		plan->T_plan_1 = NULL;
		plan->T_plan_1i = NULL;

		plan->T_plan_2->alloc_local = alloc_max;
		plan->T_plan_2i->alloc_local = alloc_max;

		if (flags == ACCFFT_MEASURE) {
			plan->T_plan_2->which_fast_method(plan->T_plan_2, data_out, 2);
		} else {
			plan->T_plan_2->method = 2;
			plan->T_plan_2->kway = 2;
		}
		plan->T_plan_2i->method = -plan->T_plan_2->method;
		plan->T_plan_2i->kway = plan->T_plan_2->kway;
		plan->T_plan_2i->kway_async = plan->T_plan_2->kway_async;
		plan->data = data;

		plan->data = data;

	} // end 1D Decomp r2c

	// 2D Decomposition
	if (!plan->oneD) {
		plan->Mem_mgr = new Mem_Mgr<double>(n[1], n_tuples_o / 2, 2,
				plan->row_comm, osize_0[0], alloc_max);
		plan->T_plan_1 = new T_Plan<double>(n[1], n_tuples_o / 2, 2,
				plan->Mem_mgr, plan->row_comm, osize_0[0]);
		plan->T_plan_2 = new T_Plan<double>(n[0], n[1], osize_2[2] * 2,
				plan->Mem_mgr, plan->col_comm);
		plan->T_plan_2i = new T_Plan<double>(n[1], n[0], osize_2i[2] * 2,
				plan->Mem_mgr, plan->col_comm);
		plan->T_plan_1i = new T_Plan<double>(n_tuples_o / 2, n[1], 2,
				plan->Mem_mgr, plan->row_comm, osize_1i[0]);



		plan->T_plan_1->alloc_local = plan->alloc_max;
		plan->T_plan_2->alloc_local = plan->alloc_max;
		plan->T_plan_2i->alloc_local = plan->alloc_max;
		plan->T_plan_1i->alloc_local = plan->alloc_max;


		if (flags == ACCFFT_MEASURE) {
			if (coord[0] == 0) {
				plan->T_plan_1->which_fast_method(plan->T_plan_1,
						(double*) data_out, 2, osize_0[0], coord[0]);
			}
		} else {
			if (coord[0] == 0) {
				plan->T_plan_1->method = 2;
				plan->T_plan_1->kway = 2;
			}
		}

		MPI_Bcast(&plan->T_plan_1->method, 1, MPI_INT, 0, c_comm);
		MPI_Bcast(&plan->T_plan_1->kway, 1, MPI_INT, 0, c_comm);
		MPI_Bcast(&plan->T_plan_1->kway_async, 1, par::Mpi_datatype<bool>::value(), 0, c_comm);

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



		plan->data = data;
	} // end 2D r2c

  // T_plan_x has to be created for both oneD and !oneD
  // Note that the T method is set via T_plan_2 for oneD case
  // to transpose N0/P0 x N1/P1 x N2 -> N0 x (N1/P1 x N2)/P0
	plan->T_plan_x = new T_Plan<double>(n[0], isize[1] * isize[2], 1,
			plan->Mem_mgr, plan->col_comm, 1);
  // to transpose N0 x (N1/P1 x N2)/P0 -> N0/P0 x N1/P1 x N2
	plan->T_plan_xi = new T_Plan<double>(isize[1] * isize[2], n[0], 1,
			plan->Mem_mgr, plan->col_comm, 1);
	plan->T_plan_x->alloc_local = plan->alloc_max;
	plan->T_plan_xi->alloc_local = plan->alloc_max;
	plan->T_plan_x->method = plan->T_plan_2->method;
	plan->T_plan_xi->method = plan->T_plan_2->method; // notice that we do not use minus for T_planxi
	plan->T_plan_x->kway = plan->T_plan_2->kway;
	plan->T_plan_xi->kway = plan->T_plan_2->kway;
	plan->T_plan_x->kway_async = plan->T_plan_2->kway_async;
	plan->T_plan_xi->kway_async = plan->T_plan_2->kway_async;


  // to transpose N0/P0 x N1/P1 x N2 -> N0/P0 x N1 x N2/P1
	plan->T_plan_y = new T_Plan<double>(n[1], n[2], 1,
			plan->Mem_mgr, plan->row_comm, plan->isize[0]);
  // to transpose N0/P0 x N1 x N2/P1 -> N0/P0 x N1/P1 x N2
	plan->T_plan_yi = new T_Plan<double>(n[2], n[1], 1,
			plan->Mem_mgr, plan->row_comm, plan->isize[0]);
	plan->T_plan_y->alloc_local = plan->alloc_max;
	plan->T_plan_yi->alloc_local = plan->alloc_max;
	plan->T_plan_y->method = plan->T_plan_2->method;
	plan->T_plan_yi->method = plan->T_plan_2->method; // method should not be set to minus
	plan->T_plan_y->kway = plan->T_plan_2->kway;
	plan->T_plan_yi->kway = plan->T_plan_2->kway;
	plan->T_plan_y->kway_async = plan->T_plan_2->kway_async;
	plan->T_plan_yi->kway_async = plan->T_plan_2->kway_async;


	plan->r2c_plan_baked = true;

	return plan;

} // end accfft_plan_dft_3d_r2c

/**
 * Get the local sizes of the distributed global data for a C2C transform
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
int accfft_local_size_dft_c2c(int * n, int * isize, int * istart, int * osize,
		int *ostart, MPI_Comm c_comm) {
  return accfft_local_size_dft_c2c_t<double>(n, isize, istart, osize, ostart, c_comm);
}

/**
 * Creates a 3D C2C parallel FFT plan. If data_out point to the same location as the input
 * data, then an inplace plan will be created. Otherwise the plan would be outplace.
 * @param n Integer array of size 3, corresponding to the global data size
 * @param data Input data in spatial domain
 * @param data_out Output data in frequency domain
 * @param c_comm Cartesian communicator returned by \ref accfft_create_comm
 * @param flags AccFFT flags, See \ref flags for more details.
 * @return
 */
accfft_plan* accfft_plan_dft_3d_c2c(int * n, Complex * data, Complex * data_out,
		MPI_Comm c_comm, unsigned flags) {

	accfft_plan *plan = new accfft_plan;
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

	plan->data_c = data;
	plan->data_out_c = data_out;
	if (data_out == data) {
		plan->inplace = true;
	} else {
		plan->inplace = false;
	}

	unsigned fftw_flags;
	if (flags == ACCFFT_ESTIMATE)
		fftw_flags = FFTW_ESTIMATE;
	else
		fftw_flags = FFTW_MEASURE;

	if (plan->np[1] == 1)
		plan->oneD = true;
	else
		plan->oneD = false;

	// FFT Plans
	int *osize_0 = plan->osize_0, *ostart_0 = plan->ostart_0;
	int *osize_1 = plan->osize_1, *ostart_1 = plan->ostart_1;
	int *osize_2 = plan->osize_2, *ostart_2 = plan->ostart_2;
	int *osize_1i = plan->osize_1i, *ostart_1i = plan->ostart_1i;
	int *osize_2i = plan->osize_2i, *ostart_2i = plan->ostart_2i;

	int alloc_local;
	int alloc_max = 0, n_tuples = (n[2] / 2 + 1) * 2;

	//int isize[3],osize[3],istart[3],ostart[3];
	alloc_max = accfft_local_size_dft_c2c(n, plan->isize, plan->istart,
			plan->osize, plan->ostart, c_comm);
	plan->alloc_max = alloc_max;

	dfft_get_local_size_t<double>(n[0], n[1], n[2], osize_0, ostart_0, c_comm);
	dfft_get_local_size_t<double>(n[0], n[2], n[1], osize_1, ostart_1, c_comm);
	dfft_get_local_size_t<double>(n[1], n[2], n[0], osize_2, ostart_2, c_comm);

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

	{
		// fplan_0
		plan->fplan_0 = fftw_plan_many_dft(1, &n[2], osize_0[0] * osize_0[1], //int rank, const int *n, int howmany
		data, NULL,         //double *in, const int *inembed,
				1, n[2],      //int istride, int idist,
				data_out, NULL, //fftw_complex *out, const int *onembed,
				1, n[2],        // int ostride, int odist,
				FFTW_FORWARD, fftw_flags);
		if (plan->fplan_0 == NULL)
			std::cout << "!!! fplan0 not created in c2c plan!!!" << std::endl;
		plan->iplan_0 = fftw_plan_many_dft(1, &n[2], osize_0[0] * osize_0[1], //int rank, const int *n, int howmany
		data_out, NULL,         //double *in, const int *inembed,
				1, n[2],      //int istride, int idist,
				data, NULL, //fftw_complex *out, const int *onembed,
				1, n[2],        // int ostride, int odist,
				FFTW_BACKWARD, fftw_flags);
		if (plan->iplan_0 == NULL)
			std::cout << "!!! iplan0 not created in c2c plan!!!" << std::endl;

		// fplan_1
		fftw_iodim dims, howmany_dims[2];
		dims.n = osize_1[1];
		dims.is = osize_1[2];
		dims.os = osize_1[2];

		howmany_dims[0].n = osize_1[2];
		howmany_dims[0].is = 1;
		howmany_dims[0].os = 1;

		howmany_dims[1].n = osize_1[0];
		howmany_dims[1].is = osize_1[1] * osize_1[2];
		howmany_dims[1].os = osize_1[1] * osize_1[2];

		plan->fplan_1 = fftw_plan_guru_dft(1, &dims, 2, howmany_dims, data_out,
				data_out, -1, fftw_flags);
		if (plan->fplan_1 == NULL)
			std::cout << "!!! fplan1 not created in c2c plan !!!" << std::endl;
		plan->iplan_1 = fftw_plan_guru_dft(1, &dims, 2, howmany_dims,
				(fftw_complex*) data_out, (fftw_complex*) data_out, 1,
				fftw_flags);
		if (plan->iplan_1 == NULL)
			std::cout << "!!! iplan1 not created in c2c plan !!!" << std::endl;

		// fplan_2
		plan->fplan_2 = fftw_plan_many_dft(1, &n[0], osize_2[1] * osize_2[2], //int rank, const int *n, int howmany
		data_out, NULL,         //double *in, const int *inembed,
				osize_2[1] * osize_2[2], 1,     //int istride, int idist,
				data_out, NULL, //fftw_complex *out, const int *onembed,
				osize_2[1] * osize_2[2], 1,       // int ostride, int odist,
				FFTW_FORWARD, fftw_flags);
		if (plan->fplan_2 == NULL)
			std::cout << "!!! fplan2 not created in c2c plan!!!" << std::endl;
		plan->iplan_2 = fftw_plan_many_dft(1, &n[0], osize_2[1] * osize_2[2], //int rank, const int *n, int howmany
		data_out, NULL,         //double *in, const int *inembed,
				osize_2[1] * osize_2[2], 1,     //int istride, int idist,
				data_out, NULL, //fftw_complex *out, const int *onembed,
				osize_2[1] * osize_2[2], 1,       // int ostride, int odist,
				FFTW_BACKWARD, fftw_flags);
		if (plan->iplan_2 == NULL)
			std::cout << "!!! fplan2 not created in c2c plan!!!" << std::endl;

	}

	// 1D Decomposition
	if (plan->oneD) {
		int NX = n[0], NY = n[1], NZ = n[2];
		plan->Mem_mgr = new Mem_Mgr<double>(NX, NY, (NZ) * 2, c_comm, 1, plan->alloc_max);
		plan->T_plan_2 = new T_Plan<double>(NX, NY, (NZ) * 2, plan->Mem_mgr,
				c_comm);
		plan->T_plan_2i = new T_Plan<double>(NY, NX, NZ * 2, plan->Mem_mgr,
				c_comm);

		plan->T_plan_1 = NULL;
		plan->T_plan_1i = NULL;

		plan->alloc_max = alloc_max;
		plan->T_plan_2->alloc_local = alloc_max;
		plan->T_plan_2i->alloc_local = alloc_max;

		if (flags == ACCFFT_MEASURE) {
			plan->T_plan_2->which_fast_method(plan->T_plan_2,
					(double*) data_out, 2);
		} else {
			plan->T_plan_2->method = 2;
			plan->T_plan_2->kway = 2;
		}
		plan->T_plan_2i->method = -plan->T_plan_2->method;
		plan->T_plan_2i->kway = plan->T_plan_2->kway;
		plan->T_plan_2i->kway_async = plan->T_plan_2->kway_async;

	} // end 1D decomp c2c

	// 2D Decomposition
	if (!plan->oneD) {
		// the reaseon for n_tuples/2 is to avoid splitting of imag and real parts of complex numbers

		plan->Mem_mgr = new Mem_Mgr<double>(n[1], n[2], 2, plan->row_comm,
				osize_0[0], alloc_max);
		plan->T_plan_1 = new T_Plan<double>(n[1], n[2], 2, plan->Mem_mgr,
				plan->row_comm, osize_0[0]);
		plan->T_plan_2 = new T_Plan<double>(n[0], n[1], 2 * osize_2[2],
				plan->Mem_mgr, plan->col_comm);
		plan->T_plan_2i = new T_Plan<double>(n[1], n[0], 2 * osize_2i[2],
				plan->Mem_mgr, plan->col_comm);
		plan->T_plan_1i = new T_Plan<double>(n[2], n[1], 2, plan->Mem_mgr,
				plan->row_comm, osize_1i[0]);

		plan->T_plan_1->alloc_local = plan->alloc_max;
		plan->T_plan_2->alloc_local = plan->alloc_max;
		plan->T_plan_2i->alloc_local = plan->alloc_max;
		plan->T_plan_1i->alloc_local = plan->alloc_max;

		if (flags == ACCFFT_MEASURE) {
			if (coord[0] == 0) {
				plan->T_plan_1->which_fast_method(plan->T_plan_1,
						(double*) data_out, 2, osize_0[0], coord[0]);
			}
		} else {
			if (coord[0] == 0) {
				plan->T_plan_1->method = 2;
				plan->T_plan_1->kway = 2;
			}
		}

		MPI_Bcast(&plan->T_plan_1->method, 1, MPI_INT, 0, c_comm);
		MPI_Bcast(&plan->T_plan_1->kway, 1, MPI_INT, 0, c_comm);
		MPI_Bcast(&plan->T_plan_1->kway_async, 1, par::Mpi_datatype<bool>::value(), 0, c_comm);

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

	} // end 2D Decomp c2c

	plan->c2c_plan_baked = true;
	return plan;

} // end accfft_plan_dft_3d_c2c

/**
 * Execute R2C plan. This function is blocking and only returns after the transform is completed.
 * @note For inplace transforms, data_out should point to the same memory address as data, AND
 * the plan must have been created as inplace.
 * @param plan FFT plan created by \ref accfft_plan_dft_3d_r2c.
 * @param data Input data in spatial domain.
 * @param data_out Output data in frequency domain.
 * @param timer See \ref timer for more details.
 * @param XYZ a bit set field that determines which directions FFT should be executed
 */
void accfft_execute_r2c(accfft_plan_t<double, Complex, fftw_plan>* plan, double * data, Complex * data_out,
		double * timer, std::bitset<3> XYZ) {
	if (plan->r2c_plan_baked) {
		accfft_execute(plan, -1, data, (double*) data_out, timer, XYZ);
	} else {
		if (plan->procid == 0)
			std::cout
					<< "Error. r2c plan has not been made correctly. Please first create the plan before calling execute functions."
					<< std::endl;
	}
	return;
}

/**
 * Execute C2R plan. This function is blocking and only returns after the transform is completed.
 * @note For inplace transform, data_out should point to the same memory address as data, AND
 * the plan must have been created as inplace.
 * @param plan FFT plan created by \ref accfft_plan_dft_3d_r2c.
 * @param data Input data in frequency domain.
 * @param data_out Output data in frequency domain.
 * @param timer See \ref timer for more details.
 * @param XYZ a bit set field that determines which directions FFT should be executed
 */
void accfft_execute_c2r(accfft_plantd* plan, Complex * data, double * data_out,
		double * timer, std::bitset<3> XYZ) {
	if (plan->r2c_plan_baked) {
		accfft_execute(plan, 1, (double*) data, data_out, timer, XYZ);
	} else {
		if (plan->procid == 0)
			std::cout
					<< "Error. r2c plan has not been made correctly. Please first create the plan before calling execute functions."
					<< std::endl;
	}
	return;
}

/**
 * Execute C2C plan. This function is blocking and only returns after the transform is completed.
 * @note For inplace transforms, data_out should point to the same memory address as data, AND
 * the plan must have been created as inplace.
 * @param plan FFT plan created by \ref accfft_plan_dft_3d_r2c.
 * @param data Input data in frequency domain.
 * @param data_out Output data in frequency domain.
 * @param timer See \ref timer for more details.
 * @param XYZ a bit set field that determines which directions FFT should be executed
 */
void accfft_execute_c2c(accfft_plantd* plan, int direction, Complex * data,
		Complex * data_out, double * timer, std::bitset<3> XYZ) {
	if (!plan->c2c_plan_baked) {
		if (plan->procid == 0)
			std::cout
					<< "Error. r2c plan has not been made correctly. Please first create the plan before calling execute functions."
					<< std::endl;
		return;
	}

	if (data == NULL)
		data = plan->data_c;
	if (data_out == NULL)
		data_out = plan->data_out_c;
	int * coords = plan->coord;
	int procid = plan->procid;
	double fft_time = 0;
	double timings[5] = { 0 };

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
		fft_time -= MPI_Wtime();
		if (XYZ[2])
			fftw_execute_dft(plan->fplan_0, data, data_out);
		else
			data_out = data;
		fft_time += MPI_Wtime();

		if (!plan->oneD) {
			plan->T_plan_1->execute(plan->T_plan_1, (double*) data_out, timings,
					2, osize_0[0], coords[0]);
		}
		/**************************************************************/
		/*******************  N0/P0 x N1 x N2/P1 **********************/
		/**************************************************************/
		fft_time -= MPI_Wtime();
		if (XYZ[1])
			fftw_execute_dft(plan->fplan_1, (fftw_complex*) data_out,
					(fftw_complex*) data_out);
		fft_time += MPI_Wtime();

		if (!plan->oneD) {
			plan->T_plan_2->execute(plan->T_plan_2, (double*) data_out, timings,
					2, 1, coords[1]);
		} else {
			plan->T_plan_2->execute(plan->T_plan_2, (double*) data_out, timings,
					2);
		}
		/**************************************************************/
		/*******************  N0 x N1/P0 x N2/P1 **********************/
		/**************************************************************/
		fft_time -= MPI_Wtime();
		if (XYZ[0])
			fftw_execute_dft(plan->fplan_2, (fftw_complex*) data_out,
					(fftw_complex*) data_out);
		fft_time += MPI_Wtime();
	} else if (direction == 1) {
		fft_time -= MPI_Wtime();
		if (XYZ[0])
			fftw_execute_dft(plan->iplan_2, (fftw_complex*) data,
					(fftw_complex*) data);
		fft_time += MPI_Wtime();

		if (!plan->oneD) {
			plan->T_plan_2i->execute(plan->T_plan_2i, (double*) data, timings,
					1, 1, coords[1]);
		} else {
			plan->T_plan_2i->execute(plan->T_plan_2i, (double*) data, timings,
					1);
		}
		/**************************************************************/
		/*******************  N0/P0 x N1 x N2/P1 **********************/
		/**************************************************************/
		fft_time -= MPI_Wtime();
		if (XYZ[1])
			fftw_execute_dft(plan->iplan_1, (fftw_complex*) data,
					(fftw_complex*) data);
		fft_time += MPI_Wtime();

		if (!plan->oneD) {
			plan->T_plan_1i->execute(plan->T_plan_1i, (double*) data, timings,
					1, osize_1i[0], coords[0]);
		}
		//MPI_Barrier(plan->c_comm);
		/**************************************************************/
		/*******************  N0/P0 x N1/P1 x N2 **********************/
		/**************************************************************/

		// IFFT in Z direction
		fft_time -= MPI_Wtime();
		if (XYZ[2])
			fftw_execute_dft(plan->iplan_0, data, data_out);
		else
			data_out = data;
		fft_time += MPI_Wtime();

	}

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
	//MPI_Barrier(plan->c_comm);

	return;
}

void accfft_destroy_plan(accfft_plantd * plan) {
	if (plan != NULL) {
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
		if (plan->fplan_0 != NULL)
			fftw_destroy_plan(plan->fplan_0);
		if (plan->fplan_1 != NULL)
			fftw_destroy_plan(plan->fplan_1);
		if (plan->fplan_2 != NULL)
			fftw_destroy_plan(plan->fplan_2);
		if (plan->iplan_0 != NULL)
			fftw_destroy_plan(plan->iplan_0);
		if (plan->iplan_1 != NULL)
			fftw_destroy_plan(plan->iplan_1);
		if (plan->iplan_2 != NULL)
			fftw_destroy_plan(plan->iplan_2);
		if (plan->fplan_x != NULL)
			fftw_destroy_plan(plan->fplan_x);
		if (plan->fplan_y != NULL)
			fftw_destroy_plan(plan->fplan_y);
		if (plan->iplan_x != NULL)
			fftw_destroy_plan(plan->iplan_x);
		if (plan->iplan_y != NULL)
			fftw_destroy_plan(plan->iplan_y);
		if (plan->T_plan_x != NULL)
			delete (plan->T_plan_x);
		if (plan->T_plan_xi != NULL)
			delete (plan->T_plan_xi);
		if (plan->T_plan_y != NULL)
			delete (plan->T_plan_y);
		if (plan->T_plan_yi != NULL)
			delete (plan->T_plan_yi);

		MPI_Comm_free(&plan->row_comm);
		MPI_Comm_free(&plan->col_comm);
		delete plan;
	}
}

template<typename T, typename Tc, typename Tp>
void accfft_execute_r2c_t(Tp* plan, T* data, Tc* data_out,
		double * timer, std::bitset<3> XYZ) {
	accfft_execute_r2c((accfft_plantd*)plan, data, data_out, timer, XYZ);
	return;
}
template<typename Tc, typename T, typename Tp>
void accfft_execute_c2r_t(Tp* plan, Tc* data, T* data_out,
		double * timer, std::bitset<3> XYZ) {
	accfft_execute_c2r((accfft_plantd*)plan, data, data_out, timer, XYZ);
	return;
}
template void accfft_execute_r2c_t<double, Complex, accfft_plantd>(accfft_plantd* plan,
		double* data, Complex* data_out, double * timer, std::bitset<3> XYZ);
template void accfft_execute_c2r_t<Complex, double, accfft_plantd>(accfft_plantd* plan,
		Complex* data, double* data_out, double * timer, std::bitset<3> XYZ);
template void accfft_execute_r2c_t<double, Complex, accfft_plan>(accfft_plan* plan,
		double* data, Complex* data_out, double * timer, std::bitset<3> XYZ);
template void accfft_execute_c2r_t<Complex, double, accfft_plan>(accfft_plan* plan,
		Complex* data, double* data_out, double * timer, std::bitset<3> XYZ);


// templates for execution only in z direction
void accfft_execute_z(accfft_plantd* plan, int direction, double * data,
		double * data_out, double * timer) {

	if (data == NULL)
		data = plan->data;
	if (data_out == NULL)
		data_out = plan->data_out;
	int * coords = plan->coord;
	int procid = plan->procid;
	double fft_time = 0;
	double timings[5] = { 0 };

	if (direction == -1) {
		/**************************************************************/
		/*******************  N0/P0 x N1/P1 x N2 **********************/
		/**************************************************************/
		// FFT in Z direction
		fft_time -= MPI_Wtime();
			fftw_execute_dft_r2c(plan->fplan_0, (double*) data,
					(fftw_complex*) data_out);
		fft_time += MPI_Wtime();
	} else if (direction == 1) {
		/**************************************************************/
		/*******************  N0/P0 x N1/P1 x N2 **********************/
		/**************************************************************/
		// IFFT in Z direction
		fft_time -= MPI_Wtime();
			fftw_execute_dft_c2r(plan->iplan_0, (fftw_complex*) data,
					(double*) data_out);
		fft_time += MPI_Wtime();
	}

	//MPI_Barrier(plan->c_comm);
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

template<typename T, typename Tc, typename Tp>
void accfft_execute_r2c_z_t(Tp* plan, T* data, Tc* data_out,
		double * timer) {

	if (plan->r2c_plan_baked) {
		accfft_execute_z((accfft_plantd*)plan, -1, data, (double*) data_out, timer);
	} else {
		if (plan->procid == 0)
			std::cout
					<< "Error. r2c plan has not been made correctly. Please first create the plan before calling execute functions."
					<< std::endl;
	}
	return;
}

template<typename Tc, typename T, typename Tp>
void accfft_execute_c2r_z_t(Tp* plan, Tc* data, T* data_out,
		double * timer) {
	if (plan->r2c_plan_baked) {
		accfft_execute_z((accfft_plantd*)plan, 1, (double*) data, data_out, timer);
	} else {
		if (plan->procid == 0)
			std::cout
					<< "Error. r2c plan has not been made correctly. Please first create the plan before calling execute functions."
					<< std::endl;
	}
	return;
}

template void accfft_execute_r2c_z_t<double, Complex, accfft_plantd>(accfft_plantd* plan,
		double* data, Complex* data_out, double * timer);
template void accfft_execute_c2r_z_t<Complex, double, accfft_plantd>(accfft_plantd* plan,
		Complex* data, double* data_out, double * timer);
template void accfft_execute_r2c_z_t<double, Complex, accfft_plan>(accfft_plan* plan,
		double* data, Complex* data_out, double * timer);
template void accfft_execute_c2r_z_t<Complex, double, accfft_plan>(accfft_plan* plan,
		Complex* data, double* data_out, double * timer);

// templates for execution only in y direction
void accfft_execute_y(accfft_plantd* plan, int direction, double * data,
		double * data_out, double * timer) {

	if (data == NULL)
		data = plan->data;
	if (data_out == NULL)
		data_out = plan->data_out;
	int * coords = plan->coord;
	int procid = plan->procid;
	double fft_time = 0;
	double timings[5] = { 0 };
	// 1D Decomposition
	int *osize_0 = plan->osize_0, *ostart_0 = plan->ostart_0;
	int *osize_1 = plan->osize_1, *ostart_1 = plan->ostart_1;
	int *osize_2 = plan->osize_2, *ostart_2 = plan->ostart_2;
	int *osize_1i = plan->osize_1i, *ostart_1i = plan->ostart_1i;
	int *osize_2i = plan->osize_2i, *ostart_2i = plan->ostart_2i;
  int *osize_y = plan->osize_y;
  int64_t N_local = plan->isize[0] * plan->isize[1] * plan->isize[2];
  double* cwork = plan->Mem_mgr->buffer_3;
	if (direction == -1) {
		/**************************************************************/
		/*******************  N0/P0 x N1/P1 x N2 **********************/
		/**************************************************************/
    // timings[0] += -MPI_Wtime();
    // memcpy(cwork, data, N_local * sizeof(double));
    // timings[0] += +MPI_Wtime();
		// if (!plan->oneD) {
		// 	plan->T_plan_y->execute(plan->T_plan_y, cwork, timings, 2,
		// 			plan->osize_y[0], coords[0]);
		// }

		plan->T_plan_y->execute(plan->T_plan_y, data, timings, 0,
				plan->osize_y[0], coords[0], cwork);
    // memcpy(data_out, cwork, plan->alloc_max); //snafu
		/**************************************************************/
		/*******************  N0/P0 x N2/P1 x N1 **********************/
		/**************************************************************/
    fft_time -= MPI_Wtime();
      fftw_execute_dft_r2c(plan->fplan_y, (double*) cwork,
          (fftw_complex*) data_out);
    fft_time += MPI_Wtime();
	} else if (direction == 1) {
		/**************************************************************/
		/*******************  N0/P0 x N1 x N2/P1 **********************/
		/**************************************************************/
    fft_time -= MPI_Wtime();
      fftw_execute_dft_c2r(plan->iplan_y, (fftw_complex*) data,
          (double*) cwork);
    fft_time += MPI_Wtime();

    // memcpy(cwork, data, N_local*sizeof(double)); //snafu
		// if (!plan->oneD) {
		//	plan->T_plan_yi->execute(plan->T_plan_yi, cwork, timings, 1,
		//			plan->osize_yi[0], coords[0]);
		// }

		plan->T_plan_yi->execute(plan->T_plan_yi, cwork, timings, 0,
				plan->osize_yi[0], coords[0], data_out);

    // timings[0] += -MPI_Wtime();
    // memcpy(data_out, cwork, N_local * sizeof(double));
    // timings[0] += +MPI_Wtime();
	}

	//MPI_Barrier(plan->c_comm);
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

template<typename T, typename Tc, typename Tp>
void accfft_execute_r2c_y_t(Tp* plan, T* data, Tc* data_out,
		double * timer) {

	if (plan->r2c_plan_baked) {
		accfft_execute_y((accfft_plantd*)plan, -1, data, (double*) data_out, timer);
	} else {
		if (plan->procid == 0)
			std::cout
					<< "Error. r2c plan has not been made correctly. Please first create the plan before calling execute functions."
					<< std::endl;
	}
	return;
}

template<typename Tc, typename T, typename Tp>
void accfft_execute_c2r_y_t(Tp* plan, Tc* data, T* data_out,
		double * timer) {
	if (plan->r2c_plan_baked) {
		accfft_execute_y((accfft_plantd*)plan, 1, (double*) data, data_out, timer);
	} else {
		if (plan->procid == 0)
			std::cout
					<< "Error. r2c plan has not been made correctly. Please first create the plan before calling execute functions."
					<< std::endl;
	}
	return;
}

template void accfft_execute_r2c_y_t<double, Complex, accfft_plantd>(accfft_plantd* plan,
		double* data, Complex* data_out, double * timer);
template void accfft_execute_c2r_y_t<Complex, double, accfft_plantd>(accfft_plantd* plan,
		Complex* data, double* data_out, double * timer);
template void accfft_execute_r2c_y_t<double, Complex, accfft_plan>(accfft_plan* plan,
		double* data, Complex* data_out, double * timer);
template void accfft_execute_c2r_y_t<Complex, double, accfft_plan>(accfft_plan* plan,
		Complex* data, double* data_out, double * timer);

// templates for execution only in x direction
void accfft_execute_x(accfft_plantd* plan, int direction, double * data,
		double * data_out, double * timer) {

	if (data == NULL)
		data = plan->data;
	if (data_out == NULL)
		data_out = plan->data_out;
	int * coords = plan->coord;
	int procid = plan->procid;
	double fft_time = 0;
	double timings[5] = { 0 };
	// 1D Decomposition
	int *osize_0 = plan->osize_0, *ostart_0 = plan->ostart_0;
	int *osize_1 = plan->osize_1, *ostart_1 = plan->ostart_1;
	int *osize_2 = plan->osize_2, *ostart_2 = plan->ostart_2;
	int *osize_1i = plan->osize_1i, *ostart_1i = plan->ostart_1i;
	int *osize_2i = plan->osize_2i, *ostart_2i = plan->ostart_2i;
  int* osize_x = plan->osize_x;
  int64_t N_local = plan->isize[0] * plan->isize[1] * plan->isize[2];
  int64_t alloc_max = plan->alloc_max;

  double* cwork = plan->Mem_mgr->buffer_3;
  // double alpha = 1;
	if (direction == -1) {
		/**************************************************************/
		/*******************  N0/P0 x N1/P1 x N2 **********************/
		/**************************************************************/
    // timings[0] += -MPI_Wtime();
    // memcpy(cwork, data, N_local * sizeof(double));
    // timings[0] += +MPI_Wtime();
    plan->T_plan_x->execute(plan->T_plan_x, data, timings, 0, 1, 0, cwork);
		/**************************************************************/
		/****************  (N1/P1 x N2)/P0 x N0 x 1 *******************/
		/**************************************************************/
    // mkl_dimatcopy('r', 't',  //const char ordering, const char trans,
    //                osize_x[1], osize_x[0], //size_t rows, size_t cols,
    //                alpha, cwork, //const MKL_Complex16 alpha, MKL_Complex16 * AB,
    //                osize_x[0], osize_x[1]); //size_t lda, size_t ldb

    // memcpy(data_out, cwork, plan->alloc_max); //snafu
		/**************************************************************/
		/*******************  N0 x N1/P0 x N2/P1 **********************/
		/**************************************************************/
    fft_time -= MPI_Wtime();
    fftw_execute_dft_r2c(plan->fplan_x, (double*) cwork,
      (fftw_complex*) data_out);
    fft_time += MPI_Wtime();
	} else if (direction == 1) {
		// IFFT in X direction
    fft_time -= MPI_Wtime();
    fftw_execute_dft_c2r(plan->iplan_x, (fftw_complex*) data,
        (double*) data);
    fft_time += MPI_Wtime();

    // mkl_dimatcopy('r', 't',  //const char ordering, const char trans,
    //                osize_x[0], osize_x[1], //size_t rows, size_t cols,
    //                alpha, data, //const MKL_Complex16 alpha, MKL_Complex16 * AB,
    //                osize_x[1], osize_x[0]); //size_t lda, size_t ldb

		/**************************************************************/
		/****************  (N1/P1 x N2)/P0 x N0 x 1 *******************/
		/**************************************************************/
    // plan->T_plan_xi->execute(plan->T_plan_xi, data, timings);
    plan->T_plan_xi->execute(plan->T_plan_xi, data, timings, 0, 1, 0, data_out);
		/**************************************************************/
		/*******************  N0 x N1/P0 x N2/P1 **********************/
		/**************************************************************/

    // timings[0] += -MPI_Wtime();
    // memcpy(data_out, data, N_local * sizeof(double));
    // timings[0] += +MPI_Wtime();
	}

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

template<typename T, typename Tc, typename Tp>
void accfft_execute_r2c_x_t(Tp* plan, T* data, Tc* data_out,
		double * timer) {

	if (plan->r2c_plan_baked) {
		accfft_execute_x((accfft_plantd*)plan, -1, data, (double*) data_out, timer);
	} else {
		if (plan->procid == 0)
			std::cout
					<< "Error. r2c plan has not been made correctly. Please first create the plan before calling execute functions."
					<< std::endl;
	}
	return;
}

template<typename Tc, typename T, typename Tp>
void accfft_execute_c2r_x_t(Tp* plan, Tc* data, T* data_out,
		double * timer) {
	if (plan->r2c_plan_baked) {
		accfft_execute_x((accfft_plantd*)plan, 1, (double*) data, data_out, timer);
	} else {
		if (plan->procid == 0)
			std::cout
					<< "Error. r2c plan has not been made correctly. Please first create the plan before calling execute functions."
					<< std::endl;
	}
	return;
}

template void accfft_execute_r2c_x_t<double, Complex, accfft_plantd>(accfft_plantd* plan,
		double* data, Complex* data_out, double * timer);
template void accfft_execute_c2r_x_t<Complex, double, accfft_plantd>(accfft_plantd* plan,
		Complex* data, double* data_out, double * timer);
template void accfft_execute_r2c_x_t<double, Complex, accfft_plan>(accfft_plan* plan,
		double* data, Complex* data_out, double * timer);
template void accfft_execute_c2r_x_t<Complex, double, accfft_plan>(accfft_plan* plan,
		Complex* data, double* data_out, double * timer);

void accfft_execute(accfft_plantd* plan, int direction, double * data,
		double * data_out, double * timer, std::bitset<3> XYZ) {

	if (data == NULL)
		data = plan->data;
	if (data_out == NULL)
		data_out = plan->data_out;
	int * coords = plan->coord;
	int procid = plan->procid;
	double fft_time = 0;
	double timings[5] = { 0 };
	// 1D Decomposition
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
		fft_time -= MPI_Wtime();
		if (XYZ[2])
			fftw_execute_dft_r2c(plan->fplan_0, (double*) data,
					(fftw_complex*) data_out);
		else
			data_out = data;
		fft_time += MPI_Wtime();

		// Perform N0/P0 transpose
		if (!plan->oneD) {
			plan->T_plan_1->execute(plan->T_plan_1, data_out, timings, 2,
					osize_0[0], coords[0]);
		}
		/**************************************************************/
		/*******************  N0/P0 x N1 x N2/P1 **********************/
		/**************************************************************/

#ifdef ACCFFT_MKL
		if (XYZ[1]){
      MKL_Complex16 alpha;
      alpha.real = 1.0;
      alpha.imag = 0;
      timings[0] += -MPI_Wtime();
      for(int i = 0; i < osize_1[0]; ++i)
        mkl_zimatcopy ('r', 't',  //const char ordering, const char trans,
                       osize_1[1], osize_1[2], //size_t rows, size_t cols,
                       alpha, (MKL_Complex16*)&data_out[2*i*osize_1[1]*osize_1[2]], //const MKL_Complex16 alpha, MKL_Complex16 * AB,
                       osize_1[2],osize_1[1]); //size_t lda, size_t ldb
      timings[0] += +MPI_Wtime();

		  fft_time -= MPI_Wtime();
		  fftw_execute_dft(plan->fplan_1, (fftw_complex*) data_out,
				(fftw_complex*) data_out);
		  fft_time += MPI_Wtime();

      timings[0] += -MPI_Wtime();
      for(int i = 0; i < osize_1[0]; ++i)
        mkl_zimatcopy ('r', 't',  //const char ordering, const char trans,
                       osize_1[2], osize_1[1],//size_t rows, size_t cols,
                       alpha, (MKL_Complex16*)&data_out[2*i*(osize_1[1])*osize_1[2]], //const MKL_Complex16 alpha, MKL_Complex16 * AB,
                       osize_1[1],osize_1[2]); //size_t lda, size_t ldb
      timings[0] += +MPI_Wtime();
      //for(int i = 0; i < osize_1[0]; ++i)
		  //	fftw_execute_dft(plan->fplan_1, (fftw_complex*) &data_out[2*i*osize_1[1]*osize_1[2]],
		  //			(fftw_complex*) &data_out[2*i*(osize_1[1])*osize_1[2]]);
    }
#else
		fft_time -= MPI_Wtime();
		if (XYZ[1])
			fftw_execute_dft(plan->fplan_1, (fftw_complex*) data_out,
					(fftw_complex*) data_out);
		fft_time += MPI_Wtime();
#endif

		if (plan->oneD) {
			plan->T_plan_2->execute(plan->T_plan_2, data_out, timings, 2);
		} else {
			plan->T_plan_2->execute(plan->T_plan_2, data_out, timings, 2, 1,
					coords[1]);
		}
		/**************************************************************/
		/*******************  N0 x N1/P0 x N2/P1 **********************/
		/**************************************************************/
#ifdef ACCFFT_MKL
		if (XYZ[0]){
      MKL_Complex16 alpha;
      alpha.real = 1.0;
      alpha.imag = 0;
      timings[0] += -MPI_Wtime();
      mkl_zimatcopy('r', 't',  //const char ordering, const char trans,
                     osize_2[0], osize_2[1] * osize_2[2], //size_t rows, size_t cols,
                     alpha, (MKL_Complex16*)data_out, //const MKL_Complex16 alpha, MKL_Complex16 * AB,
                     osize_2[1] * osize_2[2], osize_2[0]); //size_t lda, size_t ldb
      timings[0] += +MPI_Wtime();

		  fft_time -= MPI_Wtime();
			fftw_execute_dft(plan->fplan_2, (fftw_complex*) data_out,
					(fftw_complex*) data_out);
		  fft_time += MPI_Wtime();

      timings[0] += -MPI_Wtime();
      mkl_zimatcopy('r', 't',  //const char ordering, const char trans,
                     osize_2[1] * osize_2[2], osize_2[0],//size_t rows, size_t cols,
                     alpha, (MKL_Complex16*)data_out, //const MKL_Complex16 alpha, MKL_Complex16 * AB,
                     osize_2[0], osize_2[1] * osize_2[2]); //size_t lda, size_t ldb
      timings[0] += +MPI_Wtime();

      //for(int i = 0; i < osize_1[0]; ++i)
		  //	fftw_execute_dft(plan->fplan_1, (fftw_complex*) &data_out[2*i*osize_1[1]*osize_1[2]],
		  //			(fftw_complex*) &data_out[2*i*(osize_1[1])*osize_1[2]]);
    }
#else
		fft_time -= MPI_Wtime();
		if (XYZ[0])
			fftw_execute_dft(plan->fplan_2, (fftw_complex*) data_out,
					(fftw_complex*) data_out);
		fft_time += MPI_Wtime();
#endif
	} else if (direction == 1) {
#ifdef ACCFFT_MKL
		if (XYZ[0]){
      MKL_Complex16 alpha;
      alpha.real = 1.0;
      alpha.imag = 0;
      timings[0] += -MPI_Wtime();
      mkl_zimatcopy('r', 't',  //const char ordering, const char trans,
                     osize_2[0], osize_2[1] * osize_2[2], //size_t rows, size_t cols,
                     alpha, (MKL_Complex16*)data, //const MKL_Complex16 alpha, MKL_Complex16 * AB,
                     osize_2[1] * osize_2[2], osize_2[0]); //size_t lda, size_t ldb
      timings[0] += +MPI_Wtime();

		  fft_time -= MPI_Wtime();
			fftw_execute_dft(plan->iplan_2, (fftw_complex*) data,
					(fftw_complex*) data);
		  fft_time += MPI_Wtime();

      timings[0] += -MPI_Wtime();
      mkl_zimatcopy('r', 't',  //const char ordering, const char trans,
                     osize_2[1] * osize_2[2], osize_2[0],//size_t rows, size_t cols,
                     alpha, (MKL_Complex16*)data, //const MKL_Complex16 alpha, MKL_Complex16 * AB,
                     osize_2[0], osize_2[1] * osize_2[2]); //size_t lda, size_t ldb
      timings[0] += +MPI_Wtime();

      //for(int i = 0; i < osize_1[0]; ++i)
		  //	fftw_execute_dft(plan->fplan_1, (fftw_complex*) &data_out[2*i*osize_1[1]*osize_1[2]],
		  //			(fftw_complex*) &data_out[2*i*(osize_1[1])*osize_1[2]]);
    }
#else
		fft_time -= MPI_Wtime();
		if (XYZ[0])
			fftw_execute_dft(plan->iplan_2, (fftw_complex*) data,
					(fftw_complex*) data);
		fft_time += MPI_Wtime();
#endif

		if (plan->oneD) {
			plan->T_plan_2i->execute(plan->T_plan_2i, data, timings, 1);
		} else {
			plan->T_plan_2i->execute(plan->T_plan_2i, data, timings, 1, 1,
					coords[1]);
		}

		/**************************************************************/
		/*******************  N0/P0 x N1 x N2/P1 **********************/
		/**************************************************************/
#ifdef ACCFFT_MKL
		if (XYZ[1]){
      MKL_Complex16 alpha;
      alpha.real =1.0;
      alpha.imag = 0;
      for(int i = 0; i < osize_1[0]; ++i){
        mkl_zimatcopy ('r','t',  //const char ordering, const char trans,
                       osize_1i[1], osize_1i[2],//size_t rows, size_t cols,
                       alpha, (MKL_Complex16*)&data[2*i*osize_1i[1]*osize_1i[2]], //const MKL_Complex16 alpha, MKL_Complex16 * AB,
                       osize_1i[2], osize_1i[1]); //size_t lda, size_t ldb
      }

		  fft_time -= MPI_Wtime();
		  fftw_execute_dft(plan->iplan_1, (fftw_complex*) data,
				(fftw_complex*) data);
		  fft_time += MPI_Wtime();

      for(int i = 0; i < osize_1[0]; ++i)
        mkl_zimatcopy ('r','t',  //const char ordering, const char trans,
                       osize_1i[2], osize_1i[1],//size_t rows, size_t cols,
                       alpha, (MKL_Complex16*)&data[2*i*(osize_1i[1])*osize_1i[2]], //const MKL_Complex16 alpha, MKL_Complex16 * AB,
                       osize_1i[1],osize_1i[2]); //size_t lda, size_t ldb
      //for(int i = 0; i < osize_1[0]; ++i)
		  //  fftw_execute_dft(plan->iplan_1, (fftw_complex*) &data[2*i*osize_1[1]*osize_1[2]],
			//	  (fftw_complex*) &data[2*i*(osize_1[1])*osize_1[2]]);
    }
#else
		fft_time -= MPI_Wtime();
		if (XYZ[1])
			fftw_execute_dft(plan->iplan_1, (fftw_complex*) data,
					(fftw_complex*) data);
		fft_time += MPI_Wtime();
#endif

		if (!plan->oneD) {
			plan->T_plan_1i->execute(plan->T_plan_1i, data, timings, 1,
					osize_1i[0], coords[0]);
		}
		/**************************************************************/
		/*******************  N0/P0 x N1/P1 x N2 **********************/
		/**************************************************************/
		// IFFT in Z direction
		fft_time -= MPI_Wtime();
		if (XYZ[2])
			fftw_execute_dft_c2r(plan->iplan_0, (fftw_complex*) data,
					(double*) data_out);
		else
			data_out = data;
		fft_time += MPI_Wtime();

	}

	//MPI_Barrier(plan->c_comm);
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

void accfft_execute_c2r(accfft_plan* plan, Complex * data,
		double * data_out, double * timer, std::bitset<3> xyz) {
  return accfft_execute_c2r((accfft_plantd*)plan, data, data_out, timer, xyz);
}
void accfft_execute_r2c(accfft_plan* plan, double * data,
		Complex * data_out, double * timer, std::bitset<3> xyz){
  return accfft_execute_r2c((accfft_plantd*)plan, data, data_out, timer, xyz);
}
void accfft_execute_c2c(accfft_plan* plan, int direction, Complex * data,
		Complex * data_out, double * timer, std::bitset<3> xyz) {
  return accfft_execute_c2c((accfft_plantd*)plan, direction, data, data_out, timer, xyz);
}

/**
 * Destroy AccFFT CPU plan.
 * @param plan Input plan to be destroyed.
 */
void accfft_destroy_plan(accfft_plan * plan) {
  return accfft_destroy_plan((accfft_plantd*)plan);
}

void accfft_execute(accfft_plan* plan, int direction, double * data,
		double * data_out, double * timer, std::bitset<3> XYZ) {
  return accfft_execute((accfft_plantd*)plan, direction, data, data_out, timer, XYZ);
}
