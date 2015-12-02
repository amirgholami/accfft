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

/* Double Precision Instantiation */
template void accfft_grad_gpu_t<double,accfft_plan_gpu>(double * A_x, double *A_y, double *A_z,double *A,accfft_plan_gpu *plan, std::bitset<3> XYZ, double* timer);
template void accfft_laplace_gpu_t<double,accfft_plan_gpu>(double * LA,double *A,accfft_plan_gpu *plan, double* timer);
template void accfft_divergence_gpu_t<double,accfft_plan_gpu>(double* divA, double * A_x, double *A_y, double *A_z,accfft_plan_gpu *plan, double* timer);



/**
 * Computes double precision gradient of its input real data A, and returns the x, y, and z components
 * and writes the output into A_x, A_y, and A_z respectively.
 * All the arrays must reside in the device (i.e. GPU) and must have been allocated with proper size using cudaMalloc.
 * @param A_x The x component of \f$\nabla A\f$
 * @param A_y The y component of \f$\nabla A\f$
 * @param A_z The z component of \f$\nabla A\f$
 * @param plan FFT plan created by \ref accfft_plan_dft_3d_r2c_gpu. Must be an outplace plan, otherwise the function will return
 * without computing the gradient.
 * @param XYZ a bit set field of size 3 that determines which gradient components are needed. If XYZ={111} then
 * all the components are computed and if XYZ={100}, then only the x component is computed. This can save the user
 * some time, when just one or two of the gradient components are needed.
 * @param timer See \ref timer for more details.
 */
void accfft_grad_gpu(double * A_x, double *A_y, double *A_z,double *A,accfft_plan_gpu *plan, std::bitset<3> XYZ, double* timer){
  accfft_grad_gpu_t<double,accfft_plan_gpu>(A_x, A_y, A_z, A, plan, XYZ, timer);
}

/**
 * Computes double precision Laplacian of its input real data A,
 * and writes the output into LA.
 * All the arrays must reside in the device (i.e. GPU) and must have been allocated with proper size using cudaMalloc.
 * @param LA  \f$\Delta A\f$
 * @param plan FFT plan created by \ref accfft_plan_dft_3d_r2c_gpu. Must be an outplace plan, otherwise the function will return
 * without computing the gradient.
 * @param timer See \ref timer for more details.
 */
void accfft_laplace_gpu(double * LA,double *A,accfft_plan_gpu *plan, double* timer){
  accfft_laplace_gpu_t<double,accfft_plan_gpu>(LA,A,plan,timer);
}

/**
 * Computes double precision divergence of its input vector data A_x, A_y, and A_x.
 * The output data is written to divA.
 * All the arrays must reside in the device (i.e. GPU) and must have been allocated with proper size using cudaMalloc.
 * @param divA  \f$\nabla\cdot(A_x i + A_y j+ A_z k)\f$
 * @param A_x The x component of \f$\nabla A\f$
 * @param A_y The y component of \f$\nabla A\f$
 * @param A_z The z component of \f$\nabla A\f$
 * @param plan FFT plan created by \ref accfft_plan_dft_3d_r2c_gpu. Must be an outplace plan, otherwise the function will return
 * without computing the gradient.
 * @param timer See \ref timer for more details.
 */
void accfft_divergence_gpu(double* divA, double * A_x, double *A_y, double *A_z,accfft_plan_gpu *plan, double* timer){
  accfft_divergence_gpu_t<double,accfft_plan_gpu>(divA, A_x, A_y, A_z, plan, timer);
}
