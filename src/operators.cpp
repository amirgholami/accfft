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


template void grad_mult_wave_numberx<Complex>(Complex* wA, Complex* A, int* N,MPI_Comm c_comm,std::bitset<3> xyz );
template void grad_mult_wave_numbery<Complex>(Complex* wA, Complex* A, int* N,MPI_Comm c_comm,std::bitset<3> xyz );
template void grad_mult_wave_numberz<Complex>(Complex* wA, Complex* A, int* N,MPI_Comm c_comm,std::bitset<3> xyz );
template void grad_mult_wave_number_laplace<Complex>(Complex* wA, Complex* A, int* N,MPI_Comm c_comm );

template void accfft_grad_t<double,accfft_plan>(double * A_x, double *A_y, double *A_z,double *A,accfft_plan *plan, std::bitset<3> XYZ, double* timer);
template void accfft_laplace_t<double,accfft_plan>(double * LA,double *A,accfft_plan *plan, double* timer);
template void accfft_divergence_t<double,accfft_plan>(double* divA, double * A_x, double *A_y, double *A_z,accfft_plan *plan, double* timer);


/**
 * Computes double precision gradient of its input real data A, and returns the x, y, and z components
 * and writes the output into A_x, A_y, and A_z respectively.
 * @param A_x The x component of \f$\nabla A\f$
 * @param A_y The y component of \f$\nabla A\f$
 * @param A_z The z component of \f$\nabla A\f$
 * @param plan FFT plan created by \ref accfft_plan_dft_3d_r2c. Must be an outplace plan, otherwise the function will return
 * without computing the gradient.
 * @param XYZ a bit set field of size 3 that determines which gradient components are needed. If XYZ={111} then
 * all the components are computed and if XYZ={100}, then only the x component is computed. This can save the user
 * some time, when just one or two of the gradient components are needed.
 * @param timer See \ref timer for more details.
 */
void accfft_grad(double * A_x, double *A_y, double *A_z,double *A,accfft_plan *plan, std::bitset<3> XYZ, double* timer){
  accfft_grad_t<double,accfft_plan>(A_x, A_y, A_z, A, plan, XYZ, timer);
}

/**
 * Computes double precision Laplacian of its input real data A,
 * and writes the output into LA.
 * @param LA  \f$\Delta A\f$
 * @param plan FFT plan created by \ref accfft_plan_dft_3d_r2c. Must be an outplace plan, otherwise the function will return
 * without computing the gradient.
 * @param timer See \ref timer for more details.
 */
void accfft_laplace(double * LA,double *A,accfft_plan *plan, double* timer){
  accfft_laplace_t<double,accfft_plan>(LA,A,plan,timer);
}

/**
 * Computes double precision divergence of its input vector data A_x, A_y, and A_x.
 * The output data is written to divA.
 * @param divA  \f$\nabla\cdot(A_x i + A_y j+ A_z k)\f$
 * @param A_x The x component of \f$\nabla A\f$
 * @param A_y The y component of \f$\nabla A\f$
 * @param A_z The z component of \f$\nabla A\f$
 * @param plan FFT plan created by \ref accfft_plan_dft_3d_r2c. Must be an outplace plan, otherwise the function will return
 * without computing the gradient.
 * @param timer See \ref timer for more details.
 */
void accfft_divergence(double* divA, double * A_x, double *A_y, double *A_z,accfft_plan *plan, double* timer){
  accfft_divergence_t<double,accfft_plan>(divA, A_x, A_y, A_z, plan, timer);
}
