/*
 * File: step6.cpp
 * License: Please see LICENSE file.
 * AccFFT: Massively Parallel FFT Library
 * Created by Amir Gholami on 12/23/2015
 * Email: contact@accfft.org
 */

#include <stdlib.h>
#include <math.h> // M_PI
#include <mpi.h>
#include <accfft.h>
#include <accfft_operators.h>

#define SIGMA 2

void initialize(double *a, int*n, MPI_Comm c_comm);

inline double testcase(double X, double Y, double Z) {

	double pi = M_PI;
	double analytic;
	//analytic=sin(4*Y)*sin(4*X)*sin(4*Z);
	analytic = std::exp(-SIGMA * ((X - pi) * (X - pi) + (Y - pi) * (Y - pi)
							+ (Z - pi) * (Z - pi)));
	if (analytic != analytic)
		analytic = 0; /* Do you think the condition will be false always? */
	return analytic;
} // end testcase

void solution(double* u_true, double * u_0, double T, int *N, accfft_plan* plan,
		MPI_Comm c_comm) {

	int nprocs, procid;
	MPI_Comm_rank(c_comm, &procid);
	MPI_Comm_size(c_comm, &nprocs);

	int istart[3], isize[3], osize[3], ostart[3];
	int alloc_max = accfft_local_size_dft_r2c(N, isize, istart, osize, ostart,
			c_comm);

	// Allocate a scratch buffer
	Complex* u_hat = (Complex*) accfft_alloc(alloc_max);

	// First compute a_hat
	accfft_execute_r2c(plan, u_0, u_hat);

	// Hadamard product with exp(-omega^2*t)
	double scale = 1. / N[0] / N[1] / N[2];

#pragma omp parallel
	{
		long int X, Y, Z, wx, wy, wz;
		double wave;
		long int ptr;
#pragma omp for
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

					wave = std::exp(-(wx * wx + wy * wy + wz * wz) * T);

					ptr = (i * osize[1] + j) * osize[2] + k;
					u_hat[ptr][0] *= scale * wave;
					u_hat[ptr][1] *= scale * wave;
				}
			}
		}

	}

	// Inverse Fourier Transform
	accfft_execute_c2r(plan, u_hat, u_true);

	accfft_free(u_hat);
	return;
} // end solution

void step6(int *n, int nthreads) {
	int nprocs, procid;
	MPI_Comm_rank(MPI_COMM_WORLD, &procid);
	MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

	/* Create Cartesian Communicator */
	int c_dims[2] = { 0 };
	MPI_Comm c_comm;
	accfft_create_comm(MPI_COMM_WORLD, c_dims, &c_comm);
	double factor = 1;
	double dt = 1.0 / n[0] / n[0] / factor;
	int Nt = 128 * factor;
	double T = dt * Nt;

	PCOUT << "dt=\t" << dt << std::endl;
	PCOUT << "Nt=\t" << Nt << std::endl;
	PCOUT << "T=\t" << T << std::endl;

	double f_time = 0 * MPI_Wtime(), i_time = 0, setup_time = 0;
	int alloc_max = 0;

	int isize[3], osize[3], istart[3], ostart[3];
	/* Get the local pencil size and the allocation size */
	alloc_max = accfft_local_size_dft_r2c(n, isize, istart, osize, ostart,
			c_comm);

	/* Offsets for writing the output data */
	std::string filename;
	MPI_Offset istart_mpi[3] = { istart[0], istart[1], istart[2] };
	MPI_Offset isize_mpi[3] = { isize[0], isize[1], isize[2] };

	//data=(double*)accfft_alloc(isize[0]*isize[1]*isize[2]*sizeof(double));
	double * u = (double*) accfft_alloc(alloc_max);
	double * u_0 = (double*) accfft_alloc(alloc_max); // initial condition
	double * u_true = (double*) accfft_alloc(alloc_max); // analytical solution
	Complex* u_hat = (Complex*) accfft_alloc(alloc_max);

	accfft_init(nthreads);

	/* Create FFT plan */
	setup_time = -MPI_Wtime();
	accfft_plan * plan = accfft_plan_dft_3d_r2c(n, u, (double*) u_hat, c_comm,
			ACCFFT_MEASURE);
	setup_time += MPI_Wtime();

	/*  Initialize data at t=0 */
	initialize(u_0, n, c_comm);
	MPI_Barrier(c_comm);

	solution(u_true, u_0, T, n, plan, c_comm);

	double timings[5] = { 0 };
	/* Perform Euler Time Stepping */
	double * u_n = (double*) accfft_alloc(alloc_max);
	double * laplace = (double*) accfft_alloc(alloc_max);

	memcpy(u_n, u_0, alloc_max);

	double exec_time = -MPI_Wtime();
	for (int t = 0; t < Nt; ++t) {
		accfft_laplace(laplace, u_n, plan, timings);
		for (long long int i = 0; i < isize[0] * isize[1] * isize[2]; ++i) {
			u_n[i] += dt * laplace[i];
		}
		MPI_Barrier(c_comm);
	}
	exec_time += MPI_Wtime();

	/* Compute the error between u_n and u_true */
	double err = 0, g_err, norm = 0, g_norm;
	for (long long int i = 0; i < isize[0] * isize[1] * isize[2]; ++i) {
		err += (u_n[i] - u_true[i]) * (u_n[i] - u_true[i]);
		norm += (u_true[i]) * (u_true[i]);
	}
	MPI_Reduce(&err, &g_err, 1, MPI_DOUBLE, MPI_SUM, 0, c_comm);
	MPI_Reduce(&norm, &g_norm, 1, MPI_DOUBLE, MPI_SUM, 0, c_comm);
	g_err = std::sqrt(g_err);
	g_norm = std::sqrt(g_norm);
	PCOUT << "L2 Err\t" << g_err << std::endl;
	PCOUT << "L2 Rel. Err\t" << g_err / g_norm << std::endl;

#ifdef USE_PNETCDF
	/* Write the output */
	filename = "u_0.nc";
	write_pnetcdf(filename,istart_mpi,isize_mpi,c_comm,n,u_0);
	filename = "u_true.nc";
	write_pnetcdf(filename,istart_mpi,isize_mpi,c_comm,n,u_true);
	filename = "u_num.nc";
	write_pnetcdf(filename,istart_mpi,isize_mpi,c_comm,n,u_n);
#endif

	/* Compute some timings statistics */
	double g_setup_time, g_timings[5], g_exec_time;

	MPI_Reduce(timings, g_timings, 5, MPI_DOUBLE, MPI_MAX, 0, c_comm);
	MPI_Reduce(&setup_time, &g_setup_time, 1, MPI_DOUBLE, MPI_MAX, 0, c_comm);
	MPI_Reduce(&exec_time, &g_exec_time, 1, MPI_DOUBLE, MPI_MAX, 0, c_comm);

	PCOUT << "Timing for Laplace Computation for size " << n[0] << "*" << n[1]
			<< "*" << n[2] << std::endl;
	PCOUT << "Setup \t\t" << g_setup_time << std::endl;
	PCOUT << "Evaluation \t" << g_exec_time << std::endl;

	accfft_free(u);
	accfft_free(u_0);
	accfft_free(u_n);
	accfft_free(u_true);
	accfft_free(u_hat);
	MPI_Barrier(c_comm);
	accfft_free(laplace);
	accfft_destroy_plan(plan);
	accfft_cleanup();
	MPI_Comm_free(&c_comm);
	return;

} // end step6

int main(int argc, char **argv) {

	int NX, NY, NZ;
	MPI_Init(&argc, &argv);
	int nprocs, procid;
	MPI_Comm_rank(MPI_COMM_WORLD, &procid);
	MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

	/* Parsing Inputs  */
	if (argc == 1) {
		NX = 128;
		NY = 128;
		NZ = 128;
	} else {
		NX = atoi(argv[1]);
		NY = atoi(argv[2]);
		NZ = atoi(argv[3]);
	}
	int N[3] = { NX, NY, NZ };

	int nthreads = 1;

	step6(N, nthreads);

	MPI_Finalize();
	return 0;
} // end main

void initialize(double *a, int *n, MPI_Comm c_comm) {
	double pi = M_PI;
	int istart[3], isize[3], osize[3], ostart[3];
	accfft_local_size_dft_r2c(n, isize, istart, osize, ostart, c_comm);

#pragma omp parallel
	{
		double X, Y, Z;
		long int ptr;
#pragma omp for
		for (int i = 0; i < isize[0]; i++) {
			for (int j = 0; j < isize[1]; j++) {
				for (int k = 0; k < isize[2]; k++) {
					X = 2 * pi / n[0] * (i + istart[0]);
					Y = 2 * pi / n[1] * (j + istart[1]);
					Z = 2 * pi / n[2] * k;
					ptr = i * isize[1] * n[2] + j * n[2] + k;
					a[ptr] = testcase(X, Y, Z);
				}
			}
		}
	}
	return;
} // end initialize
