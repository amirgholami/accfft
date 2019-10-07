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

/* Note: This file is #include-d by accfft_utils.h
 * and requires linking with -laccfft_utils
 */

#ifndef ACCFFT_OPERATORS_GPU_H
#define ACCFFT_OPERATORS_GPU_H

template<typename T, typename Tp> void accfft_grad_gpu_slow(T* A_x, T* A_y, T* A_z,
		T* A, Tp *plan, std::bitset<3> *pXYZ = NULL, double* timer = NULL);
template<typename T, typename Tp> void accfft_grad_gpu(T* A_x, T* A_y, T* A_z,
		T* A, Tp *plan, std::bitset<3> *pXYZ = NULL, double* timer = NULL);
template<typename T, typename Tp> void accfft_laplace_gpu(T* LA, T* A,
		Tp* plan, double* timer = NULL);
template<typename T, typename Tp> void accfft_divergence_gpu(T* divA, T* A_x,
		T* A_y, T* A_z, Tp* plan, double* timer = NULL);
template<typename T, typename Tp> void accfft_divergence_gpu_slow(T* divA, T* A_x,
		T* A_y, T* A_z, Tp* plan, double* timer = NULL);
template<typename T, typename Tp> void accfft_biharmonic_gpu(T* LA, T* A,
		Tp* plan, double* timer = NULL);

template<typename Tc>
void grad_mult_wave_numberx_gpu(Tc* wA, Tc* A, int* N, int * osize,
                        int * ostart, std::bitset<3> xyz);
template<typename Tc>
void grad_mult_wave_numbery_gpu(Tc* wA, Tc* A, int* N, int * osize,
                        int * ostart, std::bitset<3> xyz);
template<typename Tc>
void grad_mult_wave_numberz_gpu(Tc* wA, Tc* A, int* N, int * osize,
                        int * ostart, std::bitset<3> xyz);

template<typename Tc>
void laplace_mult_wave_number_gpu(Tc *wA, Tc *A, int *N,
                        int *osize, int *ostart);
template<typename Tc>
void biharmonic_mult_wave_number_gpu(Tc *wA, Tc *A, int *N,
                        int *osize, int *ostart);
template<typename T>
void axpy_gpu(const long long int n, const T alpha, T *x, T *y);

#endif
