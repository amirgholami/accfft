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

#include "accfft.h"

#ifndef ACCFFT_OPERATORS_H
#define ACCFFT_OPERATORS_H

template<typename T, typename Tp> void accfft_grad_slow(T* A_x, T* A_y, T* A_z,
		T* A, Tp *plan, std::bitset<3>* pXYZ = NULL, double* timer = NULL);
template<typename T, typename Tp> void accfft_grad(T* A_x, T* A_y, T* A_z,
		T* A, Tp *plan, std::bitset<3>* pXYZ = NULL, double* timer = NULL);
template<typename T, typename Tp> void accfft_laplace(T* LA, T* A, Tp* plan,
		double* timer = NULL);
template<typename T, typename Tp> void accfft_inv_laplace(T* LA, T* A, Tp* plan,
		double* timer = NULL);
template<typename T, typename Tp> void accfft_divergence_slow(T* divA, T* A_x,
		T* A_y, T* A_z, Tp* plan, double* timer = NULL);
template<typename T, typename Tp> void accfft_divergence(T* divA, T* A_x,
		T* A_y, T* A_z, Tp* plan, double* timer = NULL);
template<typename T, typename Tp> void accfft_biharmonic(T* LA, T* A,
		Tp* plan, double* timer = NULL);
template<typename T, typename Tp> void accfft_inv_biharmonic(T* LA, T* A,
		Tp* plan, double* timer = NULL);
#endif
