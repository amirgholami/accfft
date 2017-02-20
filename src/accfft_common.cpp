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

int dfft_get_local_size(int N0, int N1, int N2, int * isize, int * istart,
		MPI_Comm c_comm) {
	int nprocs, procid;
	MPI_Comm_rank(c_comm, &procid);

	int coords[2], np[2], periods[2];
	MPI_Cart_get(c_comm, 2, np, periods, coords);
	isize[2] = N2;
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
		MPI_Barrier(c_comm);
		for (int r = 0; r < np[0]; r++)
			for (int c = 0; c < np[1]; c++) {
				MPI_Barrier(c_comm);
				if ((coords[0] == r) && (coords[1] == c))
					std::cout << coords[0] << "," << coords[1] << " isize[0]= "
							<< isize[0] << " isize[1]= " << isize[1]
							<< " isize[2]= " << isize[2] << " istart[0]= "
							<< istart[0] << " istart[1]= " << istart[1]
							<< " istart[2]= " << istart[2] << std::endl;
				MPI_Barrier(c_comm);
			}
		MPI_Barrier(c_comm);
	}
	int alloc_local = isize[0] * isize[1] * isize[2] * sizeof(double);

	return alloc_local;
}

int dfft_get_local_sizef(int N0, int N1, int N2, int * isize, int * istart,
		MPI_Comm c_comm) {
	int nprocs, procid;
	MPI_Comm_rank(c_comm, &procid);

	int coords[2], np[2], periods[2];
	MPI_Cart_get(c_comm, 2, np, periods, coords);
	isize[2] = N2;
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
		MPI_Barrier(c_comm);
		for (int r = 0; r < np[0]; r++)
			for (int c = 0; c < np[1]; c++) {
				MPI_Barrier(c_comm);
				if ((coords[0] == r) && (coords[1] == c))
					std::cout << coords[0] << "," << coords[1] << " isize[0]= "
							<< isize[0] << " isize[1]= " << isize[1]
							<< " isize[2]= " << isize[2] << " istart[0]= "
							<< istart[0] << " istart[1]= " << istart[1]
							<< " istart[2]= " << istart[2] << std::endl;
				MPI_Barrier(c_comm);
			}
		MPI_Barrier(c_comm);
	}
	int alloc_local = isize[0] * isize[1] * isize[2] * sizeof(float);

	return alloc_local;
}

int dfft_get_local_size_gpu(int N0, int N1, int N2, int * isize, int * istart,
		MPI_Comm c_comm) {
	int procid;
	MPI_Comm_rank(c_comm, &procid);

	int coords[2], np[2], periods[2];
	MPI_Cart_get(c_comm, 2, np, periods, coords);
	isize[2] = N2;
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
		MPI_Barrier(c_comm);
		for (int r = 0; r < np[0]; r++)
			for (int c = 0; c < np[1]; c++) {
				MPI_Barrier(c_comm);
				if ((coords[0] == r) && (coords[1] == c))
					std::cout << coords[0] << "," << coords[1] << " isize[0]= "
							<< isize[0] << " isize[1]= " << isize[1]
							<< " isize[2]= " << isize[2] << " istart[0]= "
							<< istart[0] << " istart[1]= " << istart[1]
							<< " istart[2]= " << istart[2] << std::endl;
				MPI_Barrier(c_comm);
			}
		MPI_Barrier(c_comm);
	}
	int alloc_local = isize[0] * isize[1] * isize[2] * sizeof(double);

	return alloc_local;
} // end dfft_get_local_size_gpu

int dfft_get_local_size_gpuf(int N0, int N1, int N2, int * isize, int * istart,
		MPI_Comm c_comm) {
	int procid;
	MPI_Comm_rank(c_comm, &procid);

	int coords[2], np[2], periods[2];
	MPI_Cart_get(c_comm, 2, np, periods, coords);
	isize[2] = N2;
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
		MPI_Barrier(c_comm);
		for (int r = 0; r < np[0]; r++)
			for (int c = 0; c < np[1]; c++) {
				MPI_Barrier(c_comm);
				if ((coords[0] == r) && (coords[1] == c))
					std::cout << coords[0] << "," << coords[1] << " isize[0]= "
							<< isize[0] << " isize[1]= " << isize[1]
							<< " isize[2]= " << isize[2] << " istart[0]= "
							<< istart[0] << " istart[1]= " << istart[1]
							<< " istart[2]= " << istart[2] << std::endl;
				MPI_Barrier(c_comm);
			}
		MPI_Barrier(c_comm);
	}
	int alloc_local = isize[0] * isize[1] * isize[2] * sizeof(float);

	return alloc_local;
}

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
