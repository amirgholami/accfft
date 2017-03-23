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

#ifndef OPERATORS_TXX
#define OPERATORS_TXX
#include "accfft_operators.h"
template<typename Tc>
static void grad_mult_wave_numberx(Tc* wA, Tc* A, int* N, MPI_Comm c_comm,
	int* size, int* start,	std::bitset<3> xyz) {

	int procid;
	MPI_Comm_rank(c_comm, &procid);
	double scale = 1;
	if (xyz[0])
		scale *= N[0];
	if (xyz[1])
		scale *= N[1];
	if (xyz[2])
		scale *= N[2];
	scale = 1. / scale;

//#pragma omp parallel
	{
		long int X, wave;
		long int ptr = 0;
//#pragma omp for
		for (int i = 0; i < size[0]; i++) {
      ptr = i * size[1] * size[2];
			for (int j = 0; j < size[1] * size[2]; j++) {
					X = (i + start[0]);
					wave = X;

					if (X > N[0] / 2)
						wave -= N[0];
					if (X == N[0] / 2)
						wave = 0; // Filter Nyquist

					//ptr = (i * size[1] + j) * size[2] + k;
					wA[ptr][0] = -scale * wave * A[ptr][1];
					wA[ptr][1] = scale * wave * A[ptr][0];
          ++ptr;
			}
		}
	}

	return;
}

template<typename Tc>
static void grad_mult_wave_numbery(Tc* wA, Tc* A, int* N, MPI_Comm c_comm,
		int* size, int* start, std::bitset<3> xyz) {

	int procid;
	MPI_Comm_rank(c_comm, &procid);

	double scale = 1;
	if (xyz[0])
		scale *= N[0];
	if (xyz[1])
		scale *= N[1];
	if (xyz[2])
		scale *= N[2];
	//PCOUT<<scale<<std::endl;
	scale = 1. / scale;


//#pragma omp parallel
	{
		long int X, Y, Z, wave;
		long int ptr;
//#pragma omp for
		for (int i = 0; i < size[0]; i++) {
			for (int j = 0; j < size[1]; j++) {
				for (int k = 0; k < size[2]; k++) {
					//X = (i + start[0]);
					Y = (j + start[1]);
					//Z = (k + start[2]);

					wave = Y;

					if (Y > N[1] / 2)
						wave -= N[1];
					if (Y == N[1] / 2)
						wave = 0; // Filter Nyquist

					ptr = (i * size[1] + j) * size[2] + k;
					wA[ptr][0] = -scale * wave * A[ptr][1];
					wA[ptr][1] = scale * wave * A[ptr][0];
				}
			}
		}
	}

	return;
}

template<typename Tc>
static void grad_mult_wave_numberz(Tc* wA, Tc* A, int* N, MPI_Comm c_comm,
		int* size, int* start, std::bitset<3> xyz) {

	int procid;
	MPI_Comm_rank(c_comm, &procid);
	double scale = 1;
	if (xyz[0])
		scale *= N[0];
	if (xyz[1])
		scale *= N[1];
	if (xyz[2])
		scale *= N[2];
	//PCOUT<<scale<<std::endl;
	scale = 1. / scale;

	// int istart[3], isize[3], osize[3], ostart[3];
	// accfft_local_size_dft_r2c_t<Tc>(N, isize, istart, osize, ostart, c_comm);
	//PCOUT<<osize[0]<<'\t'<<osize[1]<<'\t'<<osize[2]<<std::endl;

//#pragma omp parallel
	{
		long int  Z, wave;
		long int ptr;
//#pragma omp for
		for (int i = 0; i < size[0]; i++) {
			for (int j = 0; j < size[1]; j++) {
				for (int k = 0; k < size[2]; k++) {
					//X = (i + start[0]);
					//Y = (j + start[1]);
					Z = (k + start[2]);

					wave = Z;

					if (Z > N[2] / 2)
						wave -= N[2];
					if (Z == N[2] / 2)
						wave = 0; // Filter Nyquist

					ptr = (i * size[1] + j) * size[2] + k;
					wA[ptr][0] = -scale * wave * A[ptr][1];
					wA[ptr][1] = scale * wave * A[ptr][0];
				}
			}
		}
	}

	return;
}

template<typename Tc>
static void grad_mult_wave_number_laplace(Tc* wA, Tc* A, int* N,
		MPI_Comm c_comm) {

	int procid;
	MPI_Comm_rank(c_comm, &procid);
	const double scale = 1. / (N[0] * N[1] * N[2]);

	int istart[3], isize[3], osize[3], ostart[3];
	accfft_local_size_dft_r2c_t<Tc>(N, isize, istart, osize, ostart, c_comm);

//#pragma omp parallel
	{
		long int X, Y, Z, wx, wy, wz, wave;
		long int ptr;
//#pragma omp for
		for (int i = 0; i < osize[0]; i++) {
			for (int j = 0; j < osize[1]; j++) {
				for (int k = 0; k < osize[2]; k++) {
					X = (i + ostart[0]);
					Y = (j + ostart[1]);
					Z = (k + ostart[2]);

					wx = X;
					wy = Y;
					wz = Z;

					if (X > N[0] / 2)
						wx -= N[0];
					if (X == N[0] / 2)
						wx = 0;

					if (Y > N[1] / 2)
						wy -= N[1];
					if (Y == N[1] / 2)
						wy = 0;

					if (Z > N[2] / 2)
						wz -= N[2];
					if (Z == N[2] / 2)
						wz = 0;

					wave = -wx * wx - wy * wy - wz * wz;

					ptr = (i * osize[1] + j) * osize[2] + k;
					wA[ptr][0] = scale * wave * A[ptr][0];
					wA[ptr][1] = scale * wave * A[ptr][1];
				}
			}
		}
	}

	return;
}

template<typename Tc>
static void biharmonic_mult_wave_number(Tc* wA, Tc* A, int* N,
		MPI_Comm c_comm) {

	int procid;
	MPI_Comm_rank(c_comm, &procid);
	const double scale = 1. / (N[0] * N[1] * N[2]);

	int istart[3], isize[3], osize[3], ostart[3];
	accfft_local_size_dft_r2c_t<Tc>(N, isize, istart, osize, ostart, c_comm);

//#pragma omp parallel
	{
		long int X, Y, Z, wx, wy, wz, wave;
		long int ptr;
//#pragma omp for
		for (int i = 0; i < osize[0]; i++) {
			for (int j = 0; j < osize[1]; j++) {
				for (int k = 0; k < osize[2]; k++) {
					X = (i + ostart[0]);
					Y = (j + ostart[1]);
					Z = (k + ostart[2]);

					wx = X;
					wy = Y;
					wz = Z;

					if (X > N[0] / 2)
						wx -= N[0];
					if (X == N[0] / 2)
						wx = 0;

					if (Y > N[1] / 2)
						wy -= N[1];
					if (Y == N[1] / 2)
						wy = 0;

					if (Z > N[2] / 2)
						wz -= N[2];
					if (Z == N[2] / 2)
						wz = 0;

					wave = -wx * wx - wy * wy - wz * wz;
					wave *= wave;

					ptr = (i * osize[1] + j) * osize[2] + k;
					wA[ptr][0] = scale * wave * A[ptr][0];
					wA[ptr][1] = scale * wave * A[ptr][1];
				}
			}
		}
	}

	return;
}

template<typename Tc>
static void mult_wave_number_inv_laplace(Tc* wA, Tc* A, int* N,
		MPI_Comm c_comm) {

	int procid;
	MPI_Comm_rank(c_comm, &procid);
	const double scale = 1. / (N[0] * N[1] * N[2]);

	int istart[3], isize[3], osize[3], ostart[3];
	accfft_local_size_dft_r2c_t<Tc>(N, isize, istart, osize, ostart, c_comm);

//#pragma omp parallel
	{
		long int X, Y, Z, wx, wy, wz;
		double wave;
		long int ptr;
//#pragma omp for
		for (int i = 0; i < osize[0]; i++) {
			for (int j = 0; j < osize[1]; j++) {
				for (int k = 0; k < osize[2]; k++) {
					X = (i + ostart[0]);
					Y = (j + ostart[1]);
					Z = (k + ostart[2]);

					wx = X;
					wy = Y;
					wz = Z;

					if (X > N[0] / 2)
						wx -= N[0];
					if(X == N[0] / 2)
					  wx=0;

					if (Y > N[1] / 2)
						wy -= N[1];
					if(Y == N[1] / 2)
					  wy=0;

					if (Z > N[2] / 2)
						wz -= N[2];
					if(Z == N[2] / 2)
					  wz=0;

					wave = -wx * wx - wy * wy - wz * wz;
					if (wave == 0)
						wave = 0;
          else
					  wave = 1. / wave;

					ptr = (i * osize[1] + j) * osize[2] + k;
					wA[ptr][0] = scale * wave * A[ptr][0];
					wA[ptr][1] = scale * wave * A[ptr][1];
				}
			}
		}
	}

	return;
}

template<typename Tc>
static void mult_wave_number_inv_biharmonic(Tc* wA, Tc* A, int* N,
		MPI_Comm c_comm) {

	int procid;
	MPI_Comm_rank(c_comm, &procid);
	const double scale = 1. / (N[0] * N[1] * N[2]);

	int istart[3], isize[3], osize[3], ostart[3];
	accfft_local_size_dft_r2c_t<Tc>(N, isize, istart, osize, ostart, c_comm);

//#pragma omp parallel
	{
		long int X, Y, Z, wx, wy, wz;
		double wave;
		long int ptr;
//#pragma omp for
		for (int i = 0; i < osize[0]; i++) {
			for (int j = 0; j < osize[1]; j++) {
				for (int k = 0; k < osize[2]; k++) {
					X = (i + ostart[0]);
					Y = (j + ostart[1]);
					Z = (k + ostart[2]);

					wx = X;
					wy = Y;
					wz = Z;

					if (X > N[0] / 2)
						wx -= N[0];
					//if(X==N[0]/2)
					//  wx=0;

					if (Y > N[1] / 2)
						wy -= N[1];
					//if(Y==N[1]/2)
					//  wy=0;

					if (Z > N[2] / 2)
						wz -= N[2];
					//if(Z==N[2]/2)
					//  wz=0;

					wave = -wx * wx - wy * wy - wz * wz;
					wave *= wave;
					if (wave == 0)
						wave = 1;
					wave = 1. / wave;
					ptr = (i * osize[1] + j) * osize[2] + k;
					wA[ptr][0] = scale * wave * A[ptr][0];
					wA[ptr][1] = scale * wave * A[ptr][1];
				}
			}
		}
	}

	return;
}

template<typename T, typename Tp>
void accfft_grad_slow_t(T* A_x, T* A_y, T*A_z, T* A, Tp* plan, std::bitset<3>* pXYZ,
		double* timer) {
	typedef T Tc[2];
	int procid;
	MPI_Comm c_comm = plan->c_comm;
	MPI_Comm_rank(c_comm, &procid);
	if (!plan->r2c_plan_baked) {
		PCOUT << "Error in accfft_grad! plan is not correctly made."
				<< std::endl;
		return;
	}
	std::bitset < 3 > XYZ;
	if (pXYZ != NULL) {
		XYZ = *pXYZ;
	} else {
		XYZ[0] = 1;
		XYZ[1] = 1;
		XYZ[2] = 1;
	}
	double timings[7] = { 0 };

	double self_exec_time = -MPI_Wtime();
	int *N = plan->N;

	int isize[3], osize[3], istart[3], ostart[3];
	long long int alloc_max;
	/* Get the local pencil size and the allocation size */
	alloc_max = accfft_local_size_dft_r2c_t<T>(N, isize, istart, osize, ostart,
			c_comm);
	//PCOUT<<"istart[0]= "<<istart[0]<<" istart[1]= "<<istart[1]<<" istart[2]="<<istart[2]<<std::endl;
	//PCOUT<<"ostart[0]= "<<ostart[0]<<" ostart[1]= "<<ostart[1]<<" ostart[2]="<<ostart[2]<<std::endl;

	Tc* A_hat = (Tc*) accfft_alloc(alloc_max);
	Tc* tmp = (Tc*) accfft_alloc(alloc_max);
	std::bitset < 3 > scale_xyz(0);
	scale_xyz[0] = 1;
	scale_xyz[1] = 1;
	scale_xyz[2] = 1;

	MPI_Barrier(c_comm);

	/* Forward transform */
	accfft_execute_r2c_t<T, Tc, Tp>(plan, A, A_hat, timings);

	/* Multiply x Wave Numbers */
	if (XYZ[0]) {
    timings[6] += -MPI_Wtime();
		grad_mult_wave_numberx<Tc>(tmp, A_hat, N, c_comm, osize, ostart, scale_xyz);
    timings[6] += +MPI_Wtime();
		MPI_Barrier(c_comm);

		/* Backward transform */
		accfft_execute_c2r_t<Tc, T, Tp>(plan, tmp, A_x, timings);
	}
	/* Multiply y Wave Numbers */
	if (XYZ[1]) {
    timings[6] += -MPI_Wtime();
		grad_mult_wave_numbery<Tc>(tmp, A_hat, N, c_comm, osize, ostart, scale_xyz);
    timings[6] += +MPI_Wtime();
		/* Backward transform */
		accfft_execute_c2r_t<Tc, T, Tp>(plan, tmp, A_y, timings);
	}

	/* Multiply z Wave Numbers */
	if (XYZ[2]) {
    timings[6] += -MPI_Wtime();
		grad_mult_wave_numberz<Tc>(tmp, A_hat, N, c_comm, osize, ostart, scale_xyz);
    timings[6] += +MPI_Wtime();
		/* Backward transform */
		accfft_execute_c2r_t<Tc, T, Tp>(plan, tmp, A_z, timings);
	}

	accfft_free(A_hat);
	accfft_free(tmp);

	self_exec_time += MPI_Wtime();

	if (timer == NULL) {
		//delete [] timings;
	} else {
		timer[0] += timings[0];
		timer[1] += timings[1];
		timer[2] += timings[2];
		timer[3] += timings[3];
		timer[4] += timings[4];
		timer[6] += timings[6];
	}
	return;
}

template<typename T, typename Tp>
void accfft_grad_t(T* A_x, T* A_y, T*A_z, T* A, Tp* plan, std::bitset<3>* pXYZ,
		double* timer) {
	typedef T Tc[2];
	int procid;
	MPI_Comm c_comm = plan->c_comm;
	MPI_Comm_rank(c_comm, &procid);
	if (!plan->r2c_plan_baked) {
		PCOUT << "Error in accfft_grad! plan is not correctly made."
				<< std::endl;
		return;
	}
	std::bitset < 3 > XYZ;
	if (pXYZ != NULL) {
		XYZ = *pXYZ;
	} else {
		XYZ[0] = 1;
		XYZ[1] = 1;
		XYZ[2] = 1;
	}
	double timings[7] = { 0 };

	double self_exec_time = -MPI_Wtime();
	int *N = plan->N;

	int isize[3], osize[3], istart[3], ostart[3];
	long long int alloc_max;
	/* Get the local pencil size and the allocation size */
	alloc_max = accfft_local_size_dft_r2c_t<T>(N, isize, istart, osize, ostart,
			c_comm);
    if(0){
      std::cout << " osize[0] = " << osize[0]
                << " osize[1] = " << osize[1]
                << " osize[2] = " << osize[2] << std::endl;
      std::cout << " ostart[0] = " << ostart[0]
                << " ostart[1] = " << ostart[1]
                << " ostart[2] = " << ostart[2] << std::endl;
    }
	//PCOUT<<"istart[0]= "<<istart[0]<<" istart[1]= "<<istart[1]<<" istart[2]="<<istart[2]<<std::endl;
	//PCOUT<<"ostart[0]= "<<ostart[0]<<" ostart[1]= "<<ostart[1]<<" ostart[2]="<<ostart[2]<<std::endl;

	Tc* A_hat = (Tc*) accfft_alloc(alloc_max);
	Tc* tmp = (Tc*) accfft_alloc(alloc_max);
	std::bitset < 3 > scale_xyz(0);
	MPI_Barrier(c_comm);

	/* Multiply x Wave Numbers */
	if (XYZ[0]) {
    // if(procid==2){
    //   std::cout << " isize[0] = " << plan->isize[0]
    //             << " isize[1] = " << plan->isize[1]
    //             << " isize[2] = " << plan->isize[2] << std::endl;
    //   int* osize_xi = plan->osize_xi;
    //   int* osize_x = plan->osize_x;
    //   int* ostart_xi = plan->ostart_2;
    //   std::cout << " osize_x[0] = " << osize_x[0]
    //             << " osize_x[1] = " << osize_x[1]
    //             << " osize_x[2] = " << osize_x[2] << std::endl;
    //   std::cout << " osize_xi[0] = " << osize_xi[0]
    //             << " osize_xi[1] = " << osize_xi[1]
    //             << " osize_xi[2] = " << osize_xi[2] << std::endl;
    //   std::cout << "alloc_max = " << alloc_max << std::endl;
    // }
	  accfft_execute_r2c_x_t<T, Tc>(plan, A, A_hat, timings);
	  scale_xyz[0] = 1;
	  scale_xyz[1] = 0;
	  scale_xyz[2] = 0;
    timings[6] += -MPI_Wtime();
    grad_mult_wave_numberx<Tc>(tmp, A_hat, N, c_comm, plan->osize_xi, plan->ostart_2, scale_xyz);
    timings[6] += +MPI_Wtime();

    /* Backward transform */
    accfft_execute_c2r_x_t<Tc, T>(plan, tmp, A_x, timings);
    // if(procid == 0) {
    //   int64_t N_local = plan->isize[0] * plan->isize[1] * plan->isize[2];
    //   double err = 0;
    //   for(int i = 0; i < N_local; ++i)
    //     err += A_x[i] - A[i];
    //   std::cout << "err = " << err << std::endl;
    // }
  }
  /* Multiply y Wave Numbers */
  if (XYZ[1]) {
    accfft_execute_r2c_y_t<T, Tc>(plan, A, A_hat, timings);
    scale_xyz[0] = 0;
    scale_xyz[1] = 1;
    scale_xyz[2] = 0;
    timings[6] += -MPI_Wtime();
    grad_mult_wave_numbery<Tc>(tmp, A_hat, N, c_comm,
        plan->osize_yi, plan->ostart_y, scale_xyz);
    timings[6] += +MPI_Wtime();
    /* Backward transform */
    accfft_execute_c2r_y_t<Tc, T>(plan, tmp, A_y, timings);
  }

  /* Multiply z Wave Numbers */
  if (XYZ[2]) {
    accfft_execute_r2c_z_t<T, Tc>(plan, A, A_hat, timings);
    scale_xyz[0] = 0;
    scale_xyz[1] = 0;
    scale_xyz[2] = 1;
    timings[6] += -MPI_Wtime();
    grad_mult_wave_numberz<Tc>(tmp, A_hat, N, c_comm,
        plan->osize_0, plan->ostart_0, scale_xyz);
    timings[6] += +MPI_Wtime();
    /* Backward transform */
    accfft_execute_c2r_z_t<Tc, T>(plan, tmp, A_z, timings);
  }
  accfft_free(A_hat);
  accfft_free(tmp);

  self_exec_time += MPI_Wtime();

  if (timer == NULL) {
    //delete [] timings;
  } else {
    timer[0] += timings[0];
    timer[1] += timings[1];
    timer[2] += timings[2];
    timer[3] += timings[3];
    timer[4] += timings[4];
		timer[6] += timings[6];
  }
  return;
}

template<typename T, typename Tp>
void accfft_laplace_t(T* LA, T* A, Tp* plan, double* timer) {
  typedef T Tc[2];
  int procid;
  MPI_Comm c_comm = plan->c_comm;
  MPI_Comm_rank(c_comm, &procid);
  if (!plan->r2c_plan_baked) {
    PCOUT << "Error in accfft_grad! plan is not correctly made."
      << std::endl;
    return;
  }

  double timings[7] = { 0 };

  double self_exec_time = -MPI_Wtime();
  int *N = plan->N;

  int isize[3], osize[3], istart[3], ostart[3];
  long long int alloc_max;
  /* Get the local pencil size and the allocation size */
  alloc_max = accfft_local_size_dft_r2c_t<T>(N, isize, istart, osize, ostart,
      c_comm);

  Tc* A_hat = (Tc*) accfft_alloc(alloc_max);
  Tc* tmp = (Tc*) accfft_alloc(alloc_max);

  MPI_Barrier(c_comm);

  /* Forward transform */
  accfft_execute_r2c_t<T, Tc, Tp>(plan, A, A_hat, timings);

  /* Multiply x Wave Numbers */
  timings[6] += -MPI_Wtime();
  grad_mult_wave_number_laplace<Tc>(tmp, A_hat, N, c_comm);
  timings[6] += +MPI_Wtime();
  MPI_Barrier(c_comm);

  /* Backward transform */
  accfft_execute_c2r_t<Tc, T, Tp>(plan, tmp, LA, timings);

  accfft_free(A_hat);
  accfft_free(tmp);

  self_exec_time += MPI_Wtime();

  if (timer == NULL) {
    //delete [] timings;
  } else {
    timer[0] += timings[0];
    timer[1] += timings[1];
    timer[2] += timings[2];
    timer[3] += timings[3];
    timer[4] += timings[4];
		timer[6] += timings[6];
  }
  return;
}

template<typename T, typename Tp>
void accfft_divergence_slow_t(T* div_A, T* A_x, T* A_y, T* A_z, Tp* plan,
    double* timer) {
  typedef T Tc[2];
  int procid;
  MPI_Comm c_comm = plan->c_comm;
  MPI_Comm_rank(c_comm, &procid);
  if (!plan->r2c_plan_baked) {
    PCOUT << "Error in accfft_grad! plan is not correctly made."
      << std::endl;
    return;
  }

  double timings[7] = { 0 };

  double self_exec_time = -MPI_Wtime();
  int *N = plan->N;

  int isize[3], osize[3], istart[3], ostart[3];
  long long int alloc_max;
  /* Get the local pencil size and the allocation size */
  alloc_max = accfft_local_size_dft_r2c_t<T>(N, isize, istart, osize, ostart,
      c_comm);

  Tc* A_hat = (Tc*) accfft_alloc(alloc_max);
  Tc* tmp = (Tc*) accfft_alloc(alloc_max);
  T* tmp2 = (T*) accfft_alloc(alloc_max);
  std::bitset < 3 > xyz(0);

  MPI_Barrier(c_comm);

  /* Forward transform in x direction*/
  xyz[0] = 1;
  xyz[1] = 0;
  xyz[2] = 1;
  accfft_execute_r2c_t<T, Tc, Tp>(plan, A_x, A_hat, timings, xyz);
  /* Multiply x Wave Numbers */
  timings[6] += -MPI_Wtime();
  grad_mult_wave_numberx<T[2]>(tmp, A_hat, N, c_comm, osize, ostart, xyz);
  timings[6] += +MPI_Wtime();
  MPI_Barrier(c_comm);
  /* Backward transform */
  accfft_execute_c2r_t<Tc, T, Tp>(plan, tmp, div_A, timings, xyz);

  // memcpy(div_A, tmp2, isize[0] * isize[1] * isize[2] * sizeof(T));

  /* Forward transform in y direction*/
  xyz[0] = 0;
  xyz[1] = 1;
  xyz[2] = 1;
  accfft_execute_r2c_t<T, Tc, Tp>(plan, A_y, A_hat, timings, xyz);
  /* Multiply y Wave Numbers */
  timings[6] += -MPI_Wtime();
  grad_mult_wave_numbery<T[2]>(tmp, A_hat, N, c_comm, osize, ostart, xyz);
  timings[6] += +MPI_Wtime();
  MPI_Barrier(c_comm);
  /* Backward transform */
  accfft_execute_c2r_t<Tc, T, Tp>(plan, tmp, tmp2, timings, xyz);

  timings[6] += -MPI_Wtime();
  for (int i = 0; i < isize[0] * isize[1] * isize[2]; ++i)
    div_A[i] += tmp2[i];
  timings[6] += +MPI_Wtime();

  /* Forward transform in z direction*/
  xyz[0] = 0;
  xyz[1] = 0;
  xyz[2] = 1;
  accfft_execute_r2c_t<T, Tc, Tp>(plan, A_z, A_hat, timings, xyz);
  /* Multiply z Wave Numbers */
  timings[6] += -MPI_Wtime();
  grad_mult_wave_numberz<T[2]>(tmp, A_hat, N, c_comm, osize, ostart, xyz);
  timings[6] += +MPI_Wtime();
  MPI_Barrier(c_comm);
  /* Backward transform */
  accfft_execute_c2r_t<Tc, T, Tp>(plan, tmp, tmp2, timings, xyz);

  timings[6] += -MPI_Wtime();
  for (int i = 0; i < isize[0] * isize[1] * isize[2]; ++i)
    div_A[i] += tmp2[i];
  timings[6] += +MPI_Wtime();

  accfft_free(A_hat);
  accfft_free(tmp);
  accfft_free(tmp2);

  self_exec_time += MPI_Wtime();

  if (timer == NULL) {
    //delete [] timings;
  } else {
    timer[0] += timings[0];
    timer[1] += timings[1];
    timer[2] += timings[2];
    timer[3] += timings[3];
    timer[4] += timings[4];
		timer[6] += timings[6];
  }
  return;
}

template<typename T, typename Tp>
void accfft_divergence_t(T* div_A, T* A_x, T* A_y, T* A_z, Tp* plan,
    double* timer) {
  typedef T Tc[2];
  int procid;
  MPI_Comm c_comm = plan->c_comm;
  MPI_Comm_rank(c_comm, &procid);
  if (!plan->r2c_plan_baked) {
    PCOUT << "Error in accfft_grad! plan is not correctly made."
      << std::endl;
    return;
  }

  double timings[7] = { 0 };

  double self_exec_time = -MPI_Wtime();
  int *N = plan->N;

  int isize[3], osize[3], istart[3], ostart[3];
  long long int alloc_max;
  /* Get the local pencil size and the allocation size */
  alloc_max = accfft_local_size_dft_r2c_t<T>(N, isize, istart, osize, ostart,
      c_comm);

  Tc* A_hat = (Tc*) accfft_alloc(alloc_max);
  Tc* tmp = (Tc*) accfft_alloc(alloc_max);
  T* tmp2 = (T*) accfft_alloc(alloc_max);
  std::bitset < 3 > xyz(0);

  MPI_Barrier(c_comm);

  /* Forward transform in x direction*/
  accfft_execute_r2c_x_t<T, Tc>(plan, A_x, A_hat, timings);
  /* Multiply x Wave Numbers */
  xyz[0] = 1;
  xyz[1] = 0;
  xyz[2] = 0;
  timings[6] += -MPI_Wtime();
  grad_mult_wave_numberx<T[2]>(tmp, A_hat, N, c_comm, plan->osize_xi,
      plan->ostart_2, xyz);
  timings[6] += +MPI_Wtime();
  MPI_Barrier(c_comm);
  /* Backward transform */
  accfft_execute_c2r_x_t<Tc, T>(plan, tmp, div_A, timings);

  // memcpy(div_A, tmp2, isize[0] * isize[1] * isize[2] * sizeof(T));

  /* Forward transform in y direction*/
  accfft_execute_r2c_y_t<T, Tc>(plan, A_y, A_hat, timings);
  /* Multiply y Wave Numbers */
  xyz[0] = 0;
  xyz[1] = 1;
  xyz[2] = 0;
  timings[6] += -MPI_Wtime();
  grad_mult_wave_numbery<T[2]>(tmp, A_hat, N, c_comm, plan->osize_yi,
      plan->ostart_y, xyz);
  timings[6] += +MPI_Wtime();
  MPI_Barrier(c_comm);
  /* Backward transform */
  accfft_execute_c2r_y_t<Tc, T>(plan, tmp, tmp2, timings);

  timings[6] += -MPI_Wtime();
  for (int i = 0; i < isize[0] * isize[1] * isize[2]; ++i)
    div_A[i] += tmp2[i];
  timings[6] += +MPI_Wtime();

  /* Forward transform in z direction*/
  xyz[0] = 0;
  xyz[1] = 0;
  xyz[2] = 1;
  accfft_execute_r2c_z_t<T, Tc>(plan, A_z, A_hat, timings);
  /* Multiply z Wave Numbers */
  timings[6] += -MPI_Wtime();
  grad_mult_wave_numberz<T[2]>(tmp, A_hat, N, c_comm, plan->osize_0,
      plan->ostart_0, xyz);
  timings[6] += +MPI_Wtime();
  MPI_Barrier(c_comm);
  /* Backward transform */
  accfft_execute_c2r_z_t<Tc, T>(plan, tmp, tmp2, timings);

  timings[6] += -MPI_Wtime();
  for (int i = 0; i < isize[0] * isize[1] * isize[2]; ++i)
    div_A[i] += tmp2[i];
  timings[6] += +MPI_Wtime();

  accfft_free(A_hat);
  accfft_free(tmp);
  accfft_free(tmp2);

  self_exec_time += MPI_Wtime();

  if (timer == NULL) {
    //delete [] timings;
  } else {
    timer[0] += timings[0];
    timer[1] += timings[1];
    timer[2] += timings[2];
    timer[3] += timings[3];
    timer[4] += timings[4];
		timer[6] += timings[6];
  }
  return;
}

template<typename T, typename Tp>
void accfft_biharmonic_t(T* LA, T* A, Tp* plan, double* timer) {
  typedef T Tc[2];
  int procid;
  MPI_Comm c_comm = plan->c_comm;
  MPI_Comm_rank(c_comm, &procid);
  if (!plan->r2c_plan_baked) {
    PCOUT << "Error in accfft_grad! plan is not correctly made."
      << std::endl;
    return;
  }

  double timings[7] = { 0 };

  double self_exec_time = -MPI_Wtime();
  int *N = plan->N;

  int isize[3], osize[3], istart[3], ostart[3];
  long long int alloc_max;
  /* Get the local pencil size and the allocation size */
  alloc_max = accfft_local_size_dft_r2c_t<T>(N, isize, istart, osize, ostart,
      c_comm);

  Tc* A_hat = (Tc*) accfft_alloc(alloc_max);
  Tc* tmp = (Tc*) accfft_alloc(alloc_max);

  MPI_Barrier(c_comm);

  /* Forward transform */
  accfft_execute_r2c_t<T, Tc, Tp>(plan, A, A_hat, timings);

  /* Multiply x Wave Numbers */
  timings[6] += -MPI_Wtime();
  biharmonic_mult_wave_number<Tc>(tmp, A_hat, N, c_comm);
  timings[6] += +MPI_Wtime();
  MPI_Barrier(c_comm);

  /* Backward transform */
  accfft_execute_c2r_t<Tc, T, Tp>(plan, tmp, LA, timings);

  accfft_free(A_hat);
  accfft_free(tmp);

  self_exec_time += MPI_Wtime();

  if (timer == NULL) {
    //delete [] timings;
  } else {
    timer[0] += timings[0];
    timer[1] += timings[1];
    timer[2] += timings[2];
    timer[3] += timings[3];
    timer[4] += timings[4];
		timer[6] += timings[6];
  }
  return;
}

template<typename T, typename Tp>
void accfft_inv_laplace_t(T* invLA, T* A, Tp* plan, double* timer) {
  typedef T Tc[2];
  int procid;
  MPI_Comm c_comm = plan->c_comm;
  MPI_Comm_rank(c_comm, &procid);
  if (!plan->r2c_plan_baked) {
    PCOUT << "Error in accfft_grad! plan is not correctly made."
      << std::endl;
    return;
  }

  double timings[7] = { 0 };

  double self_exec_time = -MPI_Wtime();
  int *N = plan->N;

  int isize[3], osize[3], istart[3], ostart[3];
  long long int alloc_max;
  /* Get the local pencil size and the allocation size */
  alloc_max = accfft_local_size_dft_r2c_t<T>(N, isize, istart, osize, ostart,
      c_comm);

  Tc* A_hat = (Tc*) accfft_alloc(alloc_max);
  Tc* tmp = (Tc*) accfft_alloc(alloc_max);

  MPI_Barrier(c_comm);

  /* Forward transform */
  accfft_execute_r2c_t<T, Tc, Tp>(plan, A, A_hat, timings);

  /* Multiply x Wave Numbers */
  timings[6] += -MPI_Wtime();
  mult_wave_number_inv_laplace<Tc>(tmp, A_hat, N, c_comm);
  timings[6] += +MPI_Wtime();
  MPI_Barrier(c_comm);

  /* Backward transform */
  accfft_execute_c2r_t<Tc, T, Tp>(plan, tmp, invLA, timings);

  accfft_free(A_hat);
  accfft_free(tmp);

  self_exec_time += MPI_Wtime();

  if (timer == NULL) {
    //delete [] timings;
  } else {
    timer[0] += timings[0];
    timer[1] += timings[1];
    timer[2] += timings[2];
    timer[3] += timings[3];
    timer[4] += timings[4];
		timer[6] += timings[6];
  }
  return;
}

template<typename T, typename Tp>
void accfft_inv_biharmonic_t(T* invBA, T* A, Tp* plan, double* timer) {
  typedef T Tc[2];
  int procid;
  MPI_Comm c_comm = plan->c_comm;
  MPI_Comm_rank(c_comm, &procid);
  if (!plan->r2c_plan_baked) {
    PCOUT << "Error in accfft_grad! plan is not correctly made."
      << std::endl;
    return;
  }

  double timings[7] = { 0 };

  double self_exec_time = -MPI_Wtime();
  int *N = plan->N;

  int isize[3], osize[3], istart[3], ostart[3];
  long long int alloc_max;
  /* Get the local pencil size and the allocation size */
  alloc_max = accfft_local_size_dft_r2c_t<T>(N, isize, istart, osize, ostart,
      c_comm);

  Tc* A_hat = (Tc*) accfft_alloc(alloc_max);
  Tc* tmp = (Tc*) accfft_alloc(alloc_max);

  MPI_Barrier(c_comm);

  /* Forward transform */
  accfft_execute_r2c_t<T, Tc, Tp>(plan, A, A_hat, timings);

  /* Multiply x Wave Numbers */
  timings[6] += -MPI_Wtime();
  mult_wave_number_inv_biharmonic<Tc>(tmp, A_hat, N, c_comm);
  timings[6] += +MPI_Wtime();
  MPI_Barrier(c_comm);

  /* Backward transform */
  accfft_execute_c2r_t<Tc, T, Tp>(plan, tmp, invBA, timings);

  accfft_free(A_hat);
  accfft_free(tmp);

  self_exec_time += MPI_Wtime();

  if (timer == NULL) {
    //delete [] timings;
  } else {
    timer[0] += timings[0];
    timer[1] += timings[1];
    timer[2] += timings[2];
    timer[3] += timings[3];
    timer[4] += timings[4];
		timer[6] += timings[6];
  }
  return;
}

#endif
