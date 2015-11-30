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
#include <cuda_runtime_api.h>
#include <accfft_gpu.h>

#include <../src/operators_gpu.txx>


template void accfft_grad_gpu_t<double,accfft_plan_gpu>(double * A_x, double *A_y, double *A_z,double *A,accfft_plan_gpu *plan, std::bitset<3> XYZ, double* timer);
template void accfft_laplace_gpu_t<double,accfft_plan_gpu>(double * LA,double *A,accfft_plan_gpu *plan, double* timer);
template void accfft_divergence_gpu_t<double,accfft_plan_gpu>(double* divA, double * A_x, double *A_y, double *A_z,accfft_plan_gpu *plan, double* timer);


void accfft_grad_gpu(double * A_x, double *A_y, double *A_z,double *A,accfft_plan_gpu *plan, std::bitset<3> XYZ, double* timer){
  accfft_grad_gpu_t<double,accfft_plan_gpu>(A_x, A_y, A_z, A, plan, XYZ, timer);
}
void accfft_laplace_gpu(double * LA,double *A,accfft_plan_gpu *plan, double* timer){
  accfft_laplace_gpu_t<double,accfft_plan_gpu>(LA,A,plan,timer);
}

void accfft_divergence_gpu(double* divA, double * A_x, double *A_y, double *A_z,accfft_plan_gpu *plan, double* timer){
  accfft_divergence_gpu_t<double,accfft_plan_gpu>(divA, A_x, A_y, A_z, plan, timer);
}
