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
#include <accfft_gpuf.h>

#include "operators_gpu.txx"

/* Single Precision Instantiation */
template void accfft_grad_gpu_t<float, accfft_plan_gpuf>(float* A_x, float* A_y,
		float *A_z, float* A, accfft_plan_gpuf *plan, std::bitset<3> *pXYZ,
		double* timer);
template void accfft_laplace_gpu_t<float, accfft_plan_gpuf>(float* LA, float* A,
		accfft_plan_gpuf *plan, double* timer);
template void accfft_divergence_gpu_t<float, accfft_plan_gpuf>(float* divA,
		float* A_x, float *A_y, float* A_z, accfft_plan_gpuf *plan,
		double* timer);
template void accfft_biharmonic_gpu_t<float, accfft_plan_gpuf>(float* LA,
		float* A, accfft_plan_gpuf *plan, double* timer);

/**
 * Computes single precision gradient of its input real data A, and returns the x, y, and z components
 * and writes the output into A_x, A_y, and A_z respectively.
 * All the arrays must reside in the device (i.e. GPU) and must have been allocated with proper size using cudaMalloc.
 * @param A_x The x component of \f$\nabla A\f$
 * @param A_y The y component of \f$\nabla A\f$
 * @param A_z The z component of \f$\nabla A\f$
 * @param plan FFT plan created by \ref accfft_plan_dft_3d_r2c_gpuf. Must be an outplace plan, otherwise the function will return
 * without computing the gradient.
 * @param pXYZ a bit set pointer field of size 3 that determines which gradient components are needed. If XYZ={111} then
 * all the components are computed and if XYZ={100}, then only the x component is computed. This can save the user
 * some time, when just one or two of the gradient components are needed.
 * @param timer See \ref timer for more details.
 */
void accfft_grad_gpuf(float * A_x, float *A_y, float *A_z, float*A,
		accfft_plan_gpuf *plan, std::bitset<3> *pXYZ, double* timer) {
	accfft_grad_gpu_t<float, accfft_plan_gpuf>(A_x, A_y, A_z, A, plan, pXYZ,
			timer);
}

/**
 * Computes single precision Laplacian of its input real data A,
 * and writes the output into LA.
 * All the arrays must reside in the device (i.e. GPU) and must have been allocated with proper size using cudaMalloc.
 * @param LA  \f$\Delta A\f$
 * @param plan FFT plan created by \ref accfft_plan_dft_3d_r2c_gpuf. Must be an outplace plan, otherwise the function will return
 * without computing the gradient.
 * @param timer See \ref timer for more details.
 */
void accfft_laplace_gpuf(float* LA, float* A, accfft_plan_gpuf *plan,
		double* timer) {
	accfft_laplace_gpu_t<float, accfft_plan_gpuf>(LA, A, plan, timer);
}

/**
 * Computes single precision divergence of its input vector data A_x, A_y, and A_x.
 * The output data is written to divA.
 * All the arrays must reside in the device (i.e. GPU) and must have been allocated with proper size using cudaMalloc.
 * @param divA  \f$\nabla\cdot(A_x i + A_y j+ A_z k)\f$
 * @param A_x The x component of \f$\nabla A\f$
 * @param A_y The y component of \f$\nabla A\f$
 * @param A_z The z component of \f$\nabla A\f$
 * @param plan FFT plan created by \ref accfft_plan_dft_3d_r2c_gpuf. Must be an outplace plan, otherwise the function will return
 * without computing the gradient.
 * @param timer See \ref timer for more details.
 */
void accfft_divergence_gpuf(float* divA, float* A_x, float* A_y, float* A_z,
		accfft_plan_gpuf *plan, double* timer) {
	accfft_divergence_gpu_t<float, accfft_plan_gpuf>(divA, A_x, A_y, A_z, plan,
			timer);
}

/**
 * Computes single precision Biharmonic of its input real data A,
 * and writes the output into BA.
 * All the arrays must reside in the device (i.e. GPU) and must have been allocated with proper size using cudaMalloc.
 * @param BA  \f$\Delta^2 A\f$
 * @param plan FFT plan created by \ref accfft_plan_dft_3d_r2c_gpuf. Must be an outplace plan, otherwise the function will return
 * without computing the gradient.
 * @param timer See \ref timer for more details.
 */
void accfft_biharmonic_gpuf(float* BA, float* A, accfft_plan_gpuf *plan,
		double* timer) {
	accfft_biharmonic_gpu_t<float, accfft_plan_gpuf>(BA, A, plan, timer);
}
