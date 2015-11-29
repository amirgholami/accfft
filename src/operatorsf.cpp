/**
 * @file
 * Single Precision CPU functions of AccFFT operators
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
#include <accfftf.h>
#include <accfft_operators.h>

#include <../src/operators.txx>

template void grad_mult_wave_nunmberx<Complexf>(Complexf* wA, Complexf* A, int* N,MPI_Comm c_comm,std::bitset<3> xyz );
template void grad_mult_wave_nunmbery<Complexf>(Complexf* wA, Complexf* A, int* N,MPI_Comm c_comm,std::bitset<3> xyz );
template void grad_mult_wave_nunmberz<Complexf>(Complexf* wA, Complexf* A, int* N,MPI_Comm c_comm,std::bitset<3> xyz );
template void grad_mult_wave_nunmber_laplace<Complexf>(Complexf* wA, Complexf* A, int* N,MPI_Comm c_comm );

template void accfft_grad_t<float,accfft_planf>(float * A_x, float *A_y, float* A_z, float* A,accfft_planf *plan, std::bitset<3> XYZ, double* timer);
template void accfft_laplace_t<float,accfft_planf>(float * LA,float *A,accfft_planf *plan, double* timer);
template void accfft_divergence_t<float,accfft_planf>(float* divA,float* A_x, float* A_y, float*A_z,accfft_planf *plan, double* timer);


void accfft_gradf(float * A_x, float *A_y, float* A_z, float* A,accfft_planf *plan, std::bitset<3> XYZ, double* timer){
  accfft_grad_t<float,accfft_planf>(A_x, A_y, A_z, A, plan, XYZ, timer);

}

void accfft_laplacef(float * LA,float *A,accfft_planf *plan, double* timer){
  accfft_laplace_t<float,accfft_planf>(LA, A, plan, timer);
}
void accfft_divergencef(float* divA,float* A_x, float* A_y, float*A_z,accfft_planf *plan, double* timer){
  accfft_divergence_t<float,accfft_planf>(divA, A_x, A_y, A_z, plan, timer);
}
