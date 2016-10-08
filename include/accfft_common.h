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

#ifndef ACCFFT_COMMON_H
#define ACCFFT_COMMON_H
#include <mpi.h>
#include <fftw3.h>
#include <omp.h>
#include <iostream>
#include <cmath>
#include <math.h>
#define VERBOSE 0
#define PCOUT if(procid==0) std::cout
typedef double Complex[2];
typedef float Complexf[2];

#define ACCFFT_FORWARD -1
#define ACCFFT_BACKWARD +1
#define ACCFFT_INVERSE +1
#define ACCFFT_ESTIMATE 1
#define ACCFFT_MEASURE 2
#define ACCFFT_PATIENT 4

void accfft_create_comm(MPI_Comm in_comm, int * c_dims, MPI_Comm *c_comm);
int accfft_init();
void* accfft_alloc(ptrdiff_t size);
void accfft_free(void * ptr);
#endif
#ifndef _PNETCDF_IO_H_
#define _PNETCDF_IO_H_

void read_pnetcdf(const std::string &filename, MPI_Offset starts[3],
		MPI_Offset counts[3], MPI_Comm c_comm, int gsizes[3],
		double *localData);

void write_pnetcdf(const std::string &filename, MPI_Offset starts[3],
		MPI_Offset counts[3], MPI_Comm c_comm, int gsizes[3],
		double *localData);

#endif // _PNETCDF_IO_H_
#ifndef _PNETCDF_IO_F_H_
#define _PNETCDF_IO_F_H_

void read_pnetcdf(const std::string &filename, MPI_Offset starts[3],
		MPI_Offset counts[3], MPI_Comm c_comm, int gsizes[3], float *localData);

void write_pnetcdf(const std::string &filename, MPI_Offset starts[3],
		MPI_Offset counts[3], MPI_Comm c_comm, int gsizes[3], float *localData);
#endif // _PNETCDF_IO_F_H_
