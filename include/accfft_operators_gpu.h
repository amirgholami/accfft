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

#ifndef ACCFFT_OPERATORS_GPU_H
#define ACCFFT_OPERATORS_GPU_H
#include <mpi.h>
#include <fftw3.h>
#include <omp.h>
#include <iostream>
#include <cmath>
#include <math.h>
#include <accfft_gpu.h>

template<typename T, typename Tp> void accfft_grad_gpu_slow_t(T* A_x, T* A_y, T* A_z,
		T* A, Tp *plan, std::bitset<3> *pXYZ = NULL, double* timer = NULL);
template<typename T, typename Tp> void accfft_grad_gpu_t(T* A_x, T* A_y, T* A_z,
		T* A, Tp *plan, std::bitset<3> *pXYZ = NULL, double* timer = NULL);
template<typename T, typename Tp> void accfft_laplace_gpu_t(T* LA, T* A,
		Tp* plan, double* timer = NULL);
template<typename T, typename Tp> void accfft_divergence_gpu_t(T* divA, T* A_x,
		T* A_y, T* A_z, Tp* plan, double* timer = NULL);
template<typename T, typename Tp> void accfft_divergence_gpu_slow_t(T* divA, T* A_x,
		T* A_y, T* A_z, Tp* plan, double* timer = NULL);
template<typename T, typename Tp> void accfft_biharmonic_gpu_t(T* LA, T* A,
		Tp* plan, double* timer = NULL);

void accfft_grad_gpu(double* A_x, double* A_y, double* A_z, double* A,
		accfft_plan_gpu* plan, std::bitset<3> *pXYZ = NULL,
		double* timer = NULL);
void accfft_laplace_gpu(double* LA, double* A, accfft_plan_gpu* plan,
		double* timer = NULL);
void accfft_divergence_gpu(double* divA, double* A_x, double* A_y, double* A_z,
		accfft_plan_gpu* plan, double* timer = NULL);
void accfft_divergence_slow_gpu(double* divA, double* A_x, double* A_y, double* A_z,
		accfft_plan_gpu* plan, double* timer = NULL);
void accfft_biharmonic_gpu(double* LA, double* A, accfft_plan_gpu* plan,
		double* timer = NULL);

#include <accfft_gpuf.h>
void accfft_grad_gpuf(float* A_x, float* A_y, float* A_z, float* A,
		accfft_plan_gpuf *plan, std::bitset<3> *pXYZ = NULL, double* timer =
				NULL);
void accfft_laplace_gpuf(float* LA, float* A, accfft_plan_gpuf* plan,
		double* timer = NULL);
void accfft_divergence_slow_gpuf(float* divA, float* A_x, float* A_y, float* A_z,
		accfft_plan_gpuf* plan, double* timer = NULL);
void accfft_divergence_gpuf(float* divA, float* A_x, float* A_y, float* A_z,
		accfft_plan_gpuf* plan, double* timer = NULL);
void accfft_biharmonic_gpuf(float* LA, float* A, accfft_plan_gpuf* plan,
		double* timer = NULL);
#endif
