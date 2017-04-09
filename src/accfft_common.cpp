/** @file
 * common functions in AccFFT.
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
#include <omp.h>
#include <iostream>
#include <cmath>
#include <math.h>
#include "accfft_common.h"

/**
 * Allocates aligned memory to enable SIMD
 * @param size Allocation size in Bytes
 */
void* accfft_alloc(ptrdiff_t size) {
	void * ptr = fftw_malloc(size);
	return ptr;
}
/**
 * Free memory allocated by \ref accfft_alloc
 * @param ptr Address of the memory to be freed.
 */
void accfft_free(void * ptr) {
	fftw_free(ptr);
	return;
}

// uses 2D comm to compute N0/P0 x N1/P1 x tuple for each proc
template<typename T>
int dfft_get_local_size_t(int N0, int N1, int tuple, int * isize, int * istart,
		MPI_Comm c_comm) {
	int nprocs, procid;
	MPI_Comm_rank(c_comm, &procid);

	int coords[2], np[2], periods[2];
	MPI_Cart_get(c_comm, 2, np, periods, coords);
	isize[2] = tuple;
	isize[0] = ceil(N0 / (double) np[0]);
	isize[1] = ceil(N1 / (double) np[1]);

	istart[0] = isize[0] * (coords[0]);
	istart[1] = isize[1] * (coords[1]);
	istart[2] = 0;

	if ((N0 - isize[0] * coords[0]) < isize[0]) {
		isize[0] = N0 - isize[0] * coords[0];
		isize[0] *= (int) isize[0] > 0;
		istart[0] = N0 - isize[0];
	}
	if ((N1 - isize[1] * coords[1]) < isize[1]) {
		isize[1] = N1 - isize[1] * coords[1];
		isize[1] *= (int) isize[1] > 0;
		istart[1] = N1 - isize[1];
	}

	if (VERBOSE >= 2) {
		for (int r = 0; r < np[0]; r++)
			for (int c = 0; c < np[1]; c++) {
				if ((coords[0] == r) && (coords[1] == c))
					std::cout << coords[0] << "," << coords[1] << " isize[0]= "
							<< isize[0] << " isize[1]= " << isize[1]
							<< " isize[2]= " << isize[2] << " istart[0]= "
							<< istart[0] << " istart[1]= " << istart[1]
							<< " istart[2]= " << istart[2] << std::endl;
			}
	}
	int alloc_local = isize[0] * isize[1] * isize[2] * sizeof(T);

	return alloc_local;
}

template int dfft_get_local_size_t<double>(int N0, int N1, int N2, int * isize, int * istart,
    MPI_Comm c_comm);
template int dfft_get_local_size_t<float>(int N0, int N1, int N2, int * isize, int * istart,
    MPI_Comm c_comm);

template<typename T>
int accfft_local_size_dft_r2c_t(int * n, int * isize, int * istart, int * osize,
		int *ostart, MPI_Comm c_comm) {

	int procid;
	MPI_Comm_rank(c_comm, &procid);
	//1D & 2D Decomp
	int osize_0[3] = { 0 }, ostart_0[3] = { 0 };
	int osize_1[3] = { 0 }, ostart_1[3] = { 0 };
	int osize_2[3] = { 0 }, ostart_2[3] = { 0 };
	int osize_y[3] = { 0 }, ostart_y[3] = { 0 };
	int osize_yi[3] = { 0 }, ostart_yi[3] = { 0 };
	int osize_x[3] = { 0 }, ostart_x[3] = { 0 };
	int osize_xi[3] = { 0 }, ostart_xi[3] = { 0 };

	int alloc_local;
	int alloc_max = 0, n_tuples;
	//inplace==true ? n_tuples=(n[2]/2+1)*2:  n_tuples=n[2];
	n_tuples = (n[2] / 2 + 1) * 2; //SNAFU
	alloc_local = dfft_get_local_size_t<T>(n[0], n[1], n_tuples, osize_0,
			ostart_0, c_comm);
	alloc_max = std::max(alloc_max, alloc_local);
	alloc_local = dfft_get_local_size_t<T>(n[0], n_tuples / 2, n[1], osize_1,
			ostart_1, c_comm);
	alloc_max = std::max(alloc_max, alloc_local * 2);
	alloc_local = dfft_get_local_size_t<T>(n[1], n_tuples / 2, n[0], osize_2,
			ostart_2, c_comm);
	alloc_max = std::max(alloc_max, alloc_local * 2);

	std::swap(osize_1[1], osize_1[2]);
	std::swap(ostart_1[1], ostart_1[2]);

	std::swap(ostart_2[1], ostart_2[2]);
	std::swap(ostart_2[0], ostart_2[1]);
	std::swap(osize_2[1], osize_2[2]);
	std::swap(osize_2[0], osize_2[1]);

	dfft_get_local_size_t<T>(n[0], n[1], n[2], isize, istart, c_comm);
	osize[0] = osize_2[0];
	osize[1] = osize_2[1];
	osize[2] = osize_2[2];

	ostart[0] = ostart_2[0];
	ostart[1] = ostart_2[1];
	ostart[2] = ostart_2[2];

  // for y only fft
	alloc_local = dfft_get_local_size_t<T>(n[0], n[2], n[1], osize_y, ostart_y, c_comm);
	alloc_max = std::max(alloc_max, alloc_local);
	alloc_local = dfft_get_local_size_t<T>(n[0], n[2], (n[1] / 2 + 1), osize_yi, ostart_yi, c_comm);
	alloc_max = std::max(alloc_max, 2 * alloc_local);
	std::swap(osize_y[1], osize_y[2]);
	std::swap(ostart_y[1], ostart_y[2]);
	std::swap(osize_yi[1], osize_yi[2]);
	std::swap(ostart_yi[1], ostart_yi[2]);


  // for x only fft. The strategy is to divide (N1/P1 x N2) by P0 completely.
  // So we treat the last size to be just 1.
	dfft_get_local_size_t<T>(isize[1] * n[2], n[2], n[0], osize_x, ostart_x, c_comm);
  osize_x[1] = osize_x[0];
  osize_x[0] = osize_x[2];
  osize_x[2] = 1;
  std::swap(osize_x[0], osize_x[1]); // switch to (N1/P1xN2 )/P0 x N0 x 1
  ostart_x[0] = 0;
  ostart_x[1] = -1<<8; // starts have no meaning in this approach
  ostart_x[2] = -1<<8;
	alloc_local = osize_x[0] * osize_x[1] * osize_x[2] * sizeof(T);
  alloc_max = std::max(alloc_max, alloc_local);

	dfft_get_local_size_t<T>(isize[1] * n[2], n[2], n[0] / 2 + 1, osize_xi, ostart_xi, c_comm);
  osize_xi[1] = osize_xi[0];
  osize_xi[0] = osize_xi[2];
  osize_xi[2] = 1;
  std::swap(osize_xi[0], osize_xi[1]); // switch to (N1/P1xN2 )/P0 x N0 x 1
	alloc_local = osize_xi[0] * osize_xi[1] * osize_xi[2] * sizeof(T);
	alloc_max = std::max(alloc_max, 2 * alloc_local);
  ostart_xi[0] = 0;
  ostart_xi[1] = -1<<8; // starts have no meaning in this approach
  ostart_xi[2] = -1<<8;


	//isize[0]=osize_0[0];
	//isize[1]=osize_0[1];
	//isize[2]=n[2];//osize_0[2];

	return alloc_max;

} // end accfft_local_size_dft_r2c

template int accfft_local_size_dft_r2c_t<double>(int * n, int * isize,
		int * istart, int * osize, int *ostart, MPI_Comm c_comm);
template int accfft_local_size_dft_r2c_t<Complex>(int * n, int * isize,
		int * istart, int * osize, int *ostart, MPI_Comm c_comm);
template int accfft_local_size_dft_r2c_t<float>(int * n, int * isize,
		int * istart, int * osize, int *ostart, MPI_Comm c_comm);
template int accfft_local_size_dft_r2c_t<Complexf>(int * n, int * isize,
		int * istart, int * osize, int *ostart, MPI_Comm c_comm);

template<typename T>
int accfft_local_size_dft_c2c_t(int * n, int * isize, int * istart, int * osize,
		int *ostart, MPI_Comm c_comm) {

	int osize_0[3] = { 0 }, ostart_0[3] = { 0 };
	int osize_1[3] = { 0 }, ostart_1[3] = { 0 };
	int osize_2[3] = { 0 }, ostart_2[3] = { 0 };
	int osize_1i[3] = { 0 }, ostart_1i[3] = { 0 };
	int osize_2i[3] = { 0 }, ostart_2i[3] = { 0 };

	int alloc_local;
	int alloc_max = 0, n_tuples = n[2] * 2;
	alloc_local = dfft_get_local_size_t<T>(n[0], n[1], n[2], osize_0, ostart_0,
			c_comm);
	alloc_max = std::max(alloc_max, alloc_local);
	alloc_local = dfft_get_local_size_t<T>(n[0], n[2], n[1], osize_1, ostart_1,
			c_comm);
	alloc_max = std::max(alloc_max, alloc_local);
	alloc_local = dfft_get_local_size_t<T>(n[1], n[2], n[0], osize_2, ostart_2,
			c_comm);
	alloc_max = std::max(alloc_max, alloc_local);
	alloc_max *= 2; // because of c2c

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

	//isize[0]=osize_0[0];
	//isize[1]=osize_0[1];
	//isize[2]=n[2];//osize_0[2];
	dfft_get_local_size_t<T>(n[0], n[1], n[2], isize, istart, c_comm);

	osize[0] = osize_2[0];
	osize[1] = osize_2[1];
	osize[2] = osize_2[2];

	ostart[0] = ostart_2[0];
	ostart[1] = ostart_2[1];
	ostart[2] = ostart_2[2];

	return alloc_max;

}

template int accfft_local_size_dft_c2c_t<double>(int * n, int * isize,
		int * istart, int * osize, int *ostart, MPI_Comm c_comm);
template int accfft_local_size_dft_c2c_t<Complex>(int * n, int * isize,
		int * istart, int * osize, int *ostart, MPI_Comm c_comm);
template int accfft_local_size_dft_c2c_t<float>(int * n, int * isize,
		int * istart, int * osize, int *ostart, MPI_Comm c_comm);
template int accfft_local_size_dft_c2c_t<Complexf>(int * n, int * isize,
		int * istart, int * osize, int *ostart, MPI_Comm c_comm);
/**
 * Creates a Cartesian communicator of size c_dims[0]xc_dims[1] from its input.
 * If c_dims[0]xc_dims[1] would not match the size of in_comm, then the function prints
 * an error and automatically sets c_dims to the correct values.
 *
 * @param in_comm Input MPI communicator handle
 * @param c_dims A 2D integer array, which sets the size of the Cartesian array to c_dims[0]xc_dims[1]
 * @param c_comm A pointer to the Cartesian communicator which will be created
 */
void accfft_create_comm(MPI_Comm in_comm, int * c_dims, MPI_Comm *c_comm) {

	int nprocs, procid;
	MPI_Comm_rank(in_comm, &procid);
	MPI_Comm_size(in_comm, &nprocs);

	if (c_dims[0] * c_dims[1] != nprocs) {
		c_dims[0] = 0;
		c_dims[1] = 0;
		MPI_Dims_create(nprocs, 2, c_dims);
		//std::swap(c_dims[0],c_dims[1]);
		PCOUT
				<< "Input c_dim[0] * c_dims[1] != nprocs. Automatically switching to c_dims[0] = "
				<< c_dims[0] << " , c_dims_1 = " << c_dims[1] << std::endl;
	}

	/* Create Cartesian Communicator */
	int period[2], reorder;
	int coord[2];
	period[0] = 0;
	period[1] = 0;
	reorder = 1;

	MPI_Cart_create(in_comm, 2, c_dims, period, reorder, c_comm);
	//PCOUT<<"dim[0]= "<<c_dims[0]<<" dim[1]= "<<c_dims[1]<<std::endl;

	//MPI_Cart_coords(c_comm, procid, 2, coord);

	return;

}
/**
 * Initialize AccFFT library.
 * @return 0 if successful.
 */
int accfft_init() {
	return 0;
}

/**
 * Initializes the library.
 * @param nthreads The number of OpenMP threads to use for execution of local FFT.
 * @return 0 if successful
 */
int accfft_init(int nthreads) {
	int threads_ok = 1;
	if (threads_ok)
		threads_ok = fftw_init_threads();
	if (threads_ok)
		fftw_plan_with_nthreads(nthreads);

	return (!threads_ok);
}

/**
 * Cleanup all CPU resources
 */
void accfft_cleanup() {
	fftw_cleanup_threads();
	fftw_cleanup();
}
