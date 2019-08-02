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

#ifndef _ACCFFT_UTILS_H
#define _ACCFFT_UTILS_H

#include "accfft_operators.h"
#include "accfft_operators_gpu.h"

#ifdef USE_PNETCDF
template void read_pnetcdf(const std::string &, MPI_Offset starts[3],
        MPI_Offset counts[3], MPI_Comm, int gsizes[3], float *);
template void write_pnetcdf(const std::string &, MPI_Offset starts[3],
        MPI_Offset counts[3], MPI_Comm, int gsizes[3], double *);
template void read_pnetcdf(const std::string &, MPI_Offset starts[3],
        MPI_Offset counts[3], MPI_Comm, int gsizes[3], float *);
template void write_pnetcdf(const std::string &, MPI_Offset starts[3],
        MPI_Offset counts[3], MPI_Comm, int gsizes[3], double *);
#endif

#endif // _ACCFFT_UTILS_H
