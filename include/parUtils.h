/*
 *  Copyright (c) 2014-2015, George Biros
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

/*
 * The following people have contributed to this file:
 * @author Rahul S. Sampath, rahul.sampath@gmail.com
 * @author Hari Sundar, hsundar@gmail.com
 * @author Shravan Veerapaneni, shravan@seas.upenn.edu
 * @author Santi Swaroop Adavani, santis@gmail.com
 * @author Dhairya Malhotra, dhairya.malhotra88@gmail.com
 * @author Amir Gholami, gholami@accfft.org
 */

#ifndef __PAR_UTILS_H_
#define __PAR_UTILS_H_

#include <mpi.h>
#include <vector>

#ifndef KWAY
#define KWAY 128
#endif

namespace par {

template<class T, bool ASYNC>
int Mpi_Alltoallv_dense(T* sendbuf, int* sendcnts, int* sdispls, T* recvbuf,
		int* recvcnts, int* rdispls, MPI_Comm comm, int kway = KWAY);

template<class T, bool ASYNC>
int Mpi_Alltoallv_dense_gpu(T* sendbuf, int* sendcnts, int* sdispls, T* recvbuf,
		int* recvcnts, int* rdispls, MPI_Comm comm, int kway = KWAY);

}

#include "parUtils.txx"

#endif

