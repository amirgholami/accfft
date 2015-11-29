/**
 * @file
 * CPU functions of AccFFT operators
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
#include <string.h>
#include <accfft.h>

#include <../src/operators.txx>


template void grad_mult_wave_nunmberx<Complex>(Complex* wA, Complex* A, int* N,MPI_Comm c_comm,std::bitset<3> xyz );
template void grad_mult_wave_nunmbery<Complex>(Complex* wA, Complex* A, int* N,MPI_Comm c_comm,std::bitset<3> xyz );
template void grad_mult_wave_nunmberz<Complex>(Complex* wA, Complex* A, int* N,MPI_Comm c_comm,std::bitset<3> xyz );
template void grad_mult_wave_nunmber_laplace<Complex>(Complex* wA, Complex* A, int* N,MPI_Comm c_comm );

template void accfft_grad_t<double,accfft_plan>(double * A_x, double *A_y, double *A_z,double *A,accfft_plan *plan, std::bitset<3> XYZ, double* timer);
template void accfft_laplace_t<double,accfft_plan>(double * LA,double *A,accfft_plan *plan, double* timer);
template void accfft_divergence_t<double,accfft_plan>(double* divA, double * A_x, double *A_y, double *A_z,accfft_plan *plan, double* timer);


void accfft_grad(double * A_x, double *A_y, double *A_z,double *A,accfft_plan *plan, std::bitset<3> XYZ, double* timer){
  accfft_grad_t<double,accfft_plan>(A_x, A_y, A_z, A, plan, XYZ, timer);
}

void accfft_laplace(double * LA,double *A,accfft_plan *plan, double* timer){
  accfft_laplace_t<double,accfft_plan>(LA,A,plan,timer);
}

void accfft_divergence(double* divA, double * A_x, double *A_y, double *A_z,accfft_plan *plan, double* timer){
  accfft_divergence_t<double,accfft_plan>(divA, A_x, A_y, A_z, plan, timer);
}
