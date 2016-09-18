/*
 * File: grad1.cpp
 * License: Please see LICENSE file.
 * AccFFT: Massively Parallel FFT Library
 * Created by Amir Gholami on 12/23/2014
 * Email: contact@accfft.org
 */

#include <stdlib.h>
#include <math.h> // M_PI
#include <mpi.h>
#include <cuda_runtime_api.h>

#include <accfft_gpuf.h>
#include <accfft_operators_gpu.h>
#define SIGMA 32
#define C0 1
#define C1 1

inline double testcase(double X, double Y, double Z) {

	double pi = M_PI;
	double analytic;
	analytic = C0 * sin(X) * sin(Y) * sin(Z)
			+ C1
					* std::exp(
							-SIGMA
									* ((X - pi) * (X - pi) + (Y - pi) * (Y - pi)
											+ (Z - pi) * (Z - pi)));
	if (analytic != analytic)
		analytic = 0; /* Do you think the condition will be false always? */
	return analytic;
} // end testcase
inline double solution_grad(double X, double Y, double Z, int direction) {

	//return testcase(X,Y,Z);
	double pi = M_PI;
	double analytic = 0;
	if (direction == 0)
		analytic = C0 * cos(X) * sin(Y) * sin(Z)
				+ C1 * (-2 * SIGMA * (X - pi))
						* std::exp(
								-SIGMA
										* ((X - pi) * (X - pi)
												+ (Y - pi) * (Y - pi)
												+ (Z - pi) * (Z - pi)));
	else if (direction == 1)
		analytic = C0 * sin(X) * cos(Y) * sin(Z)
				+ C1 * (-2 * SIGMA * (Y - pi))
						* std::exp(
								-SIGMA
										* ((X - pi) * (X - pi)
												+ (Y - pi) * (Y - pi)
												+ (Z - pi) * (Z - pi)));
	else if (direction == 2)
		analytic = C0 * sin(X) * sin(Y) * cos(Z)
				+ C1 * (-2 * SIGMA * (Z - pi))
						* std::exp(
								-SIGMA
										* ((X - pi) * (X - pi)
												+ (Y - pi) * (Y - pi)
												+ (Z - pi) * (Z - pi)));

	if (analytic != analytic)
		analytic = 0; /* Do you think the condition will be false always? */
	return analytic;
} // end solution_grad

inline double solution_laplace(double X, double Y, double Z) {

	double pi = M_PI;
	double analytic = 0;
	analytic += C1 * (4 * SIGMA * SIGMA * (X - pi) * (X - pi))
			* std::exp(
					-SIGMA
							* ((X - pi) * (X - pi) + (Y - pi) * (Y - pi)
									+ (Z - pi) * (Z - pi)));
	analytic += C1 * (4 * SIGMA * SIGMA * (Y - pi) * (Y - pi))
			* std::exp(
					-SIGMA
							* ((X - pi) * (X - pi) + (Y - pi) * (Y - pi)
									+ (Z - pi) * (Z - pi)));
	analytic += C1 * (4 * SIGMA * SIGMA * (Z - pi) * (Z - pi))
			* std::exp(
					-SIGMA
							* ((X - pi) * (X - pi) + (Y - pi) * (Y - pi)
									+ (Z - pi) * (Z - pi)));
	analytic += C1 * (-6 * SIGMA)
			* std::exp(
					-SIGMA
							* ((X - pi) * (X - pi) + (Y - pi) * (Y - pi)
									+ (Z - pi) * (Z - pi)));
	analytic += -C0 * 3 * sin(X) * sin(Y) * sin(Z);
	if (analytic != analytic)
		analytic = 0;
	return analytic;
} // end solution_laplace
inline double solution_biharmonic(double X, double Y, double Z) {

	double pi = M_PI;
	double analytic = 0;
	double SIGMA2 = SIGMA * SIGMA;
	double SIGMA3 = SIGMA * SIGMA2;
	double SIGMA4 = SIGMA * SIGMA3;
	double f = std::exp(
			-SIGMA
					* ((X - pi) * (X - pi) + (Y - pi) * (Y - pi)
							+ (Z - pi) * (Z - pi)));
	double Xp = X - pi;
	double Yp = Y - pi;
	double Zp = Z - pi;
	//double fxx=-2*SIGMA*f+4*SIGMA2*Xp*Xp*f;
	//double fyy=-2*SIGMA*f+4*SIGMA2*Yp*Yp*f;
	//double fzz=-2*SIGMA*f+4*SIGMA2*Zp*Zp*f;
	double fxxxx = 12 * SIGMA2 * f - 48 * SIGMA3 * Xp * Xp * f
			+ 16 * SIGMA4 * Xp * Xp * Xp * Xp * f;
	double fyyyy = 12 * SIGMA2 * f - 48 * SIGMA3 * Yp * Yp * f
			+ 16 * SIGMA4 * Yp * Yp * Yp * Yp * f;
	double fzzzz = 12 * SIGMA2 * f - 48 * SIGMA3 * Zp * Zp * f
			+ 16 * SIGMA4 * Zp * Zp * Zp * Zp * f;
	double fxxyy = 4 * SIGMA2 * f - 8 * SIGMA3 * Yp * Yp * f
			- 8 * SIGMA3 * Xp * Xp * f + 16 * SIGMA4 * Xp * Xp * Yp * Yp * f;
	double fxxzz = 4 * SIGMA2 * f - 8 * SIGMA3 * Zp * Zp * f
			- 8 * SIGMA3 * Xp * Xp * f + 16 * SIGMA4 * Xp * Xp * Zp * Zp * f;
	double fyyzz = 4 * SIGMA2 * f - 8 * SIGMA3 * Zp * Zp * f
			- 8 * SIGMA3 * Yp * Yp * f + 16 * SIGMA4 * Yp * Yp * Zp * Zp * f;

	analytic += C0 * 9 * sin(X) * sin(Y) * sin(Z)
			+ C1 * (fxxxx + 2 * fxxyy + 2 * fxxzz + fyyyy + 2 * fyyzz + fzzzz);
	if (analytic != analytic)
		analytic = 0;
	return analytic;
} // end solution_biharmonic
void check_err_grad(double* a, int*n, MPI_Comm c_comm, int direction);
void check_err_fft(double* a, int*n, MPI_Comm c_comm);
void check_err_laplace(double* a, int*n, MPI_Comm c_comm);
void check_err_biharmonic(double* a, int*n, MPI_Comm c_comm);

void initialize(double *a, int*n, MPI_Comm c_comm);

void grad(int *n) {
	int nprocs, procid;
	MPI_Comm_rank(MPI_COMM_WORLD, &procid);
	MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

	/* Create Cartesian Communicator */
	int c_dims[2] = { 0 };
	MPI_Comm c_comm;
	accfft_create_comm(MPI_COMM_WORLD, c_dims, &c_comm);

	double f_time = 0 * MPI_Wtime(), i_time = 0, setup_time = 0;
	int alloc_max = 0;

	int isize[3], osize[3], istart[3], ostart[3];
	/* Get the local pencil size and the allocation size */
	alloc_max = accfft_local_size_dft_r2c_gpu(n, isize, istart, osize, ostart,
			c_comm);

	//data=(double*)accfft_alloc(isize[0]*isize[1]*isize[2]*sizeof(double));
	double * data_cpu = (double*) accfft_alloc(alloc_max);
	double* data;
	Complex* data_hat;
	cudaMalloc((void**) &data, alloc_max);
	cudaMalloc((void**) &data_hat, alloc_max);

	accfft_init();

	/* Create FFT plan */
	setup_time = -MPI_Wtime();
	accfft_plan_gpu * plan = accfft_plan_dft_3d_r2c_gpu(n, data,
			(double*) data_hat, c_comm, ACCFFT_MEASURE);
	setup_time += MPI_Wtime();

	/*  Initialize data */
	initialize(data_cpu, n, c_comm);
	cudaMemcpy(data, data_cpu, alloc_max, cudaMemcpyHostToDevice);

	MPI_Barrier(c_comm);

	double * gradx, *grady, *gradz;
	cudaMalloc((void**) &gradx, alloc_max);
	cudaMalloc((void**) &grady, alloc_max);
	cudaMalloc((void**) &gradz, alloc_max);
	double timings[5] = { 0 };

	std::bitset < 3 > XYZ = 0;
	XYZ[0] = 1;
	XYZ[1] = 1;
	XYZ[2] = 1;
	double exec_time = -MPI_Wtime();
	accfft_grad_gpu(gradx, grady, gradz, data, plan, &XYZ, timings);
	exec_time += MPI_Wtime();
	/* Check err*/
	PCOUT << ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>" << std::endl;
	PCOUT << ">>>>>>>>Checking Gradx>>>>>>>>" << std::endl;
	cudaMemcpy(data_cpu, gradx, alloc_max, cudaMemcpyDeviceToHost);
	check_err_grad(data_cpu, n, c_comm, 0);
	PCOUT << "<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<" << std::endl;
	PCOUT << "<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<\n" << std::endl;

	PCOUT << ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>" << std::endl;
	PCOUT << ">>>>>>>>Checking Grady>>>>>>>>" << std::endl;
	cudaMemcpy(data_cpu, grady, alloc_max, cudaMemcpyDeviceToHost);
	check_err_grad(data_cpu, n, c_comm, 1);
	PCOUT << "<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<" << std::endl;
	PCOUT << "<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<\n" << std::endl;

	PCOUT << ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>" << std::endl;
	PCOUT << ">>>>>>>>Checking Gradz>>>>>>>>" << std::endl;
	cudaMemcpy(data_cpu, gradz, alloc_max, cudaMemcpyDeviceToHost);
	check_err_grad(data_cpu, n, c_comm, 2);
	PCOUT << "<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<" << std::endl;
	PCOUT << "<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<\n" << std::endl;

	/* Compute some timings statistics */
	double g_setup_time, g_timings[5], g_exec_time;

	MPI_Reduce(timings, g_timings, 5, MPI_DOUBLE, MPI_MAX, 0, c_comm);
	MPI_Reduce(&setup_time, &g_setup_time, 1, MPI_DOUBLE, MPI_MAX, 0, c_comm);
	MPI_Reduce(&exec_time, &g_exec_time, 1, MPI_DOUBLE, MPI_MAX, 0, c_comm);

	PCOUT << "Timing for Grad Computation for size " << n[0] << "*" << n[1]
			<< "*" << n[2] << std::endl;
	PCOUT << "Setup \t\t" << g_setup_time << std::endl;
	PCOUT << "Evaluation \t" << g_exec_time << std::endl;

	accfft_free(data_cpu);
	cudaFree(data);
	cudaFree(data_hat);
	MPI_Barrier(c_comm);
	cudaFree(gradx);
	cudaFree(grady);
	cudaFree(gradz);
	accfft_destroy_plan(plan);
	accfft_cleanup_gpu();
	MPI_Comm_free(&c_comm);
	PCOUT << "-------------------------------------------------------";
	PCOUT << "-------------------------------------------------------";
	PCOUT << "-------------------------------------------------------\n";
	return;

} // end grad

void laplace(int *n) {
	int nprocs, procid;
	MPI_Comm_rank(MPI_COMM_WORLD, &procid);
	MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

	/* Create Cartesian Communicator */
	int c_dims[2] = { 0 };
	MPI_Comm c_comm;
	accfft_create_comm(MPI_COMM_WORLD, c_dims, &c_comm);

	double f_time = 0 * MPI_Wtime(), i_time = 0, setup_time = 0;
	int alloc_max = 0;

	int isize[3], osize[3], istart[3], ostart[3];
	/* Get the local pencil size and the allocation size */
	alloc_max = accfft_local_size_dft_r2c_gpu(n, isize, istart, osize, ostart,
			c_comm);

	//data=(double*)accfft_alloc(isize[0]*isize[1]*isize[2]*sizeof(double));
	double * data_cpu = (double*) accfft_alloc(alloc_max);
	double* data;
	Complex* data_hat;
	cudaMalloc((void**) &data, alloc_max);
	cudaMalloc((void**) &data_hat, alloc_max);

	accfft_init();

	/* Create FFT plan */
	setup_time = -MPI_Wtime();
	accfft_plan_gpu * plan = accfft_plan_dft_3d_r2c_gpu(n, data,
			(double*) data_hat, c_comm, ACCFFT_MEASURE);
	setup_time += MPI_Wtime();

	/*  Initialize data */
	initialize(data_cpu, n, c_comm);
	cudaMemcpy(data, data_cpu, alloc_max, cudaMemcpyHostToDevice);

	MPI_Barrier(c_comm);

	double * laplace;
	cudaMalloc((void**) &laplace, alloc_max);
	double timings[5] = { 0 };

	double exec_time = -MPI_Wtime();
	accfft_laplace_gpu(laplace, data, plan, timings);
	exec_time += MPI_Wtime();
	/* Check err*/
	PCOUT << ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>" << std::endl;
	PCOUT << ">>>>>>>Checking Laplace>>>>>>>" << std::endl;
	cudaMemcpy(data_cpu, laplace, alloc_max, cudaMemcpyDeviceToHost);
	check_err_laplace(data_cpu, n, c_comm);
	PCOUT << "<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<" << std::endl;
	PCOUT << "<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<\n" << std::endl;

	/* Compute some timings statistics */
	double g_setup_time, g_timings[5], g_exec_time;

	MPI_Reduce(timings, g_timings, 5, MPI_DOUBLE, MPI_MAX, 0, c_comm);
	MPI_Reduce(&setup_time, &g_setup_time, 1, MPI_DOUBLE, MPI_MAX, 0, c_comm);
	MPI_Reduce(&exec_time, &g_exec_time, 1, MPI_DOUBLE, MPI_MAX, 0, c_comm);

	PCOUT << "Timing for Grad Computation for size " << n[0] << "*" << n[1]
			<< "*" << n[2] << std::endl;
	PCOUT << "Setup \t\t" << g_setup_time << std::endl;
	PCOUT << "Evaluation \t" << g_exec_time << std::endl;

	accfft_free(data_cpu);
	cudaFree(data);
	cudaFree(data_hat);
	MPI_Barrier(c_comm);
	cudaFree(laplace);
	accfft_destroy_plan(plan);
	accfft_cleanup_gpu();
	MPI_Comm_free(&c_comm);
	PCOUT << "-------------------------------------------------------";
	PCOUT << "-------------------------------------------------------";
	PCOUT << "-------------------------------------------------------\n";
	return;

} // end laplace

void biharmonic(int *n) {
	int nprocs, procid;
	MPI_Comm_rank(MPI_COMM_WORLD, &procid);
	MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

	/* Create Cartesian Communicator */
	int c_dims[2] = { 0 };
	MPI_Comm c_comm;
	accfft_create_comm(MPI_COMM_WORLD, c_dims, &c_comm);

	double f_time = 0 * MPI_Wtime(), i_time = 0, setup_time = 0;
	int alloc_max = 0;

	int isize[3], osize[3], istart[3], ostart[3];
	/* Get the local pencil size and the allocation size */
	alloc_max = accfft_local_size_dft_r2c_gpu(n, isize, istart, osize, ostart,
			c_comm);

	//data=(double*)accfft_alloc(isize[0]*isize[1]*isize[2]*sizeof(double));
	double * data_cpu = (double*) accfft_alloc(alloc_max);
	double* data;
	Complex* data_hat;
	cudaMalloc((void**) &data, alloc_max);
	cudaMalloc((void**) &data_hat, alloc_max);

	accfft_init();

	/* Create FFT plan */
	setup_time = -MPI_Wtime();
	accfft_plan_gpu * plan = accfft_plan_dft_3d_r2c_gpu(n, data,
			(double*) data_hat, c_comm, ACCFFT_MEASURE);
	setup_time += MPI_Wtime();

	/*  Initialize data */
	initialize(data_cpu, n, c_comm);
	cudaMemcpy(data, data_cpu, alloc_max, cudaMemcpyHostToDevice);

	MPI_Barrier(c_comm);

	double * biharmonic;
	cudaMalloc((void**) &biharmonic, alloc_max);
	double timings[5] = { 0 };

	double exec_time = -MPI_Wtime();
	accfft_biharmonic_gpu(biharmonic, data, plan, timings);
	exec_time += MPI_Wtime();
	/* Check err*/
	PCOUT << ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>" << std::endl;
	PCOUT << ">>>>Checking Biharmonic>>>>>>>" << std::endl;
	cudaMemcpy(data_cpu, biharmonic, alloc_max, cudaMemcpyDeviceToHost);
	check_err_biharmonic(data_cpu, n, c_comm);
	PCOUT << "<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<" << std::endl;
	PCOUT << "<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<\n" << std::endl;

	/* Compute some timings statistics */
	double g_setup_time, g_timings[5], g_exec_time;

	MPI_Reduce(timings, g_timings, 5, MPI_DOUBLE, MPI_MAX, 0, c_comm);
	MPI_Reduce(&setup_time, &g_setup_time, 1, MPI_DOUBLE, MPI_MAX, 0, c_comm);
	MPI_Reduce(&exec_time, &g_exec_time, 1, MPI_DOUBLE, MPI_MAX, 0, c_comm);

	PCOUT << "Timing for Grad Computation for size " << n[0] << "*" << n[1]
			<< "*" << n[2] << std::endl;
	PCOUT << "Setup \t\t" << g_setup_time << std::endl;
	PCOUT << "Evaluation \t" << g_exec_time << std::endl;

	accfft_free(data_cpu);
	cudaFree(data);
	cudaFree(data_hat);
	MPI_Barrier(c_comm);
	cudaFree(biharmonic);
	accfft_destroy_plan(plan);
	accfft_cleanup_gpu();
	MPI_Comm_free(&c_comm);
	PCOUT << "-------------------------------------------------------"
			<< std::endl;
	PCOUT << "-------------------------------------------------------"
			<< std::endl;
	PCOUT << "-------------------------------------------------------\n"
			<< std::endl;
	return;

} // end laplace

void divergence(int *n) {
	int nprocs, procid;
	MPI_Comm_rank(MPI_COMM_WORLD, &procid);
	MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

	/* Create Cartesian Communicator */
	int c_dims[2] = { 0 };
	MPI_Comm c_comm;
	accfft_create_comm(MPI_COMM_WORLD, c_dims, &c_comm);

	double f_time = 0 * MPI_Wtime(), i_time = 0, setup_time = 0;
	int alloc_max = 0;

	int isize[3], osize[3], istart[3], ostart[3];
	/* Get the local pencil size and the allocation size */
	alloc_max = accfft_local_size_dft_r2c_gpu(n, isize, istart, osize, ostart,
			c_comm);

	//data=(double*)accfft_alloc(isize[0]*isize[1]*isize[2]*sizeof(double));
	double * data_cpu = (double*) accfft_alloc(alloc_max);
	double* data;
	Complex* data_hat;
	cudaMalloc((void**) &data, alloc_max);
	cudaMalloc((void**) &data_hat, alloc_max);

	accfft_init();

	/* Create FFT plan */
	setup_time = -MPI_Wtime();
	accfft_plan_gpu * plan = accfft_plan_dft_3d_r2c_gpu(n, data,
			(double*) data_hat, c_comm, ACCFFT_MEASURE);
	setup_time += MPI_Wtime();

	/*  Initialize data */
	initialize(data_cpu, n, c_comm);
	cudaMemcpy(data, data_cpu, alloc_max, cudaMemcpyHostToDevice);

	MPI_Barrier(c_comm);

	double * gradx, *grady, *gradz, *divergence;
	cudaMalloc((void**) &gradx, alloc_max);
	cudaMalloc((void**) &grady, alloc_max);
	cudaMalloc((void**) &gradz, alloc_max);
	cudaMalloc((void**) &divergence, alloc_max);
	double timings[5] = { 0 };

	std::bitset < 3 > XYZ = 0;
	XYZ[0] = 1;
	XYZ[1] = 1;
	XYZ[2] = 1;
	double exec_time = -MPI_Wtime();
	accfft_grad_gpu(gradx, grady, gradz, data, plan, &XYZ, timings);
	accfft_divergence_gpu(divergence, gradx, grady, gradz, plan, timings);
	exec_time += MPI_Wtime();

	PCOUT << ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>" << std::endl;
	PCOUT << ">>>>>Checking Divergence>>>>>>" << std::endl;
	cudaMemcpy(data_cpu, divergence, alloc_max, cudaMemcpyDeviceToHost);
	check_err_laplace(data_cpu, n, c_comm);
	PCOUT << "<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<" << std::endl;
	PCOUT << "<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<\n" << std::endl;

	/* Compute some timings statistics */
	double g_setup_time, g_timings[5], g_exec_time;

	MPI_Reduce(timings, g_timings, 5, MPI_DOUBLE, MPI_MAX, 0, c_comm);
	MPI_Reduce(&setup_time, &g_setup_time, 1, MPI_DOUBLE, MPI_MAX, 0, c_comm);
	MPI_Reduce(&exec_time, &g_exec_time, 1, MPI_DOUBLE, MPI_MAX, 0, c_comm);

	PCOUT << "Timing for Grad Computation for size " << n[0] << "*" << n[1]
			<< "*" << n[2] << std::endl;
	PCOUT << "Setup \t\t" << g_setup_time << std::endl;
	PCOUT << "Evaluation \t" << g_exec_time << std::endl;

	accfft_free(data_cpu);
	cudaFree(data);
	cudaFree(data_hat);
	MPI_Barrier(c_comm);
	cudaFree(gradx);
	cudaFree(grady);
	cudaFree(gradz);
	accfft_destroy_plan(plan);
	accfft_cleanup_gpu();
	MPI_Comm_free(&c_comm);
	PCOUT << "-------------------------------------------------------"
			<< std::endl;
	PCOUT << "-------------------------------------------------------"
			<< std::endl;
	PCOUT << "-------------------------------------------------------\n"
			<< std::endl;
	return;

} // end divergence

int main(int argc, char **argv) {

	int NX, NY, NZ;
	MPI_Init(&argc, &argv);
	int nprocs, procid;
	MPI_Comm_rank(MPI_COMM_WORLD, &procid);
	MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

	/* Parsing Inputs  */
	if (argc == 1) {
		NX = 256;
		NY = 256;
		NZ = 256;
	} else {
		NX = atoi(argv[1]);
		NY = atoi(argv[2]);
		NZ = atoi(argv[3]);
	}
	int N[3] = { NX, NY, NZ };

	grad(N);

	laplace(N);

	divergence(N);

	biharmonic(N);
	MPI_Finalize();
	return 0;
} // end main

void initialize(double *a, int *n, MPI_Comm c_comm) {
	double pi = M_PI;
	int istart[3], isize[3], osize[3], ostart[3];
	accfft_local_size_dft_r2c_gpu(n, isize, istart, osize, ostart, c_comm);

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

void check_err_grad(double* a, int*n, MPI_Comm c_comm, int direction) {
	int nprocs, procid;
	MPI_Comm_rank(c_comm, &procid);
	MPI_Comm_size(c_comm, &nprocs);
	long long int size = n[0];
	size *= n[1];
	size *= n[2];
	size = 1;
	double pi = 4 * atan(1.0);

	int istart[3], isize[3], osize[3], ostart[3];
	accfft_local_size_dft_r2c_gpu(n, isize, istart, osize, ostart, c_comm);

	double err = 0, norm = 0;

	double X, Y, Z, numerical_r;
	long int ptr;
	int thid = omp_get_thread_num();
	for (int i = 0; i < isize[0]; i++) {
		for (int j = 0; j < isize[1]; j++) {
			for (int k = 0; k < isize[2]; k++) {
				X = 2 * pi / n[0] * (i + istart[0]);
				Y = 2 * pi / n[1] * (j + istart[1]);
				Z = 2 * pi / n[2] * k;
				ptr = i * isize[1] * n[2] + j * n[2] + k;
				numerical_r = a[ptr] / size;
				if (numerical_r != numerical_r)
					numerical_r = 0;
				err += std::abs(
						numerical_r - solution_grad(X, Y, Z, direction));
				norm += std::abs(solution_grad(X, Y, Z, direction));
			}
		}
	}

	double g_err = 0, g_norm = 0;
	MPI_Reduce(&err, &g_err, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
	MPI_Reduce(&norm, &g_norm, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
	PCOUT << "L1 Error of iFF(a)-a: " << g_err << std::endl;
	PCOUT << "Relative L1 Error of iFF(a)-a: " << g_err / g_norm << std::endl;
	if (g_err / g_norm < 1e-10)
		PCOUT << "Results are CORRECT!" << std::endl;
	else
		PCOUT << "\nResults are NOT CORRECT!\n";

	return;
} // end check_err

void check_err_laplace(double* a, int*n, MPI_Comm c_comm) {
	int nprocs, procid;
	MPI_Comm_rank(c_comm, &procid);
	MPI_Comm_size(c_comm, &nprocs);
	long long int size = n[0];
	size *= n[1];
	size *= n[2];
	size = 1;
	double pi = M_PI;

	int istart[3], isize[3], osize[3], ostart[3];
	accfft_local_size_dft_r2c_gpu(n, isize, istart, osize, ostart, c_comm);

	double err = 0, norm = 0;

	double X, Y, Z, numerical_r;
	long int ptr;
	int thid = omp_get_thread_num();
	for (int i = 0; i < isize[0]; i++) {
		for (int j = 0; j < isize[1]; j++) {
			for (int k = 0; k < isize[2]; k++) {
				X = 2 * pi / n[0] * (i + istart[0]);
				Y = 2 * pi / n[1] * (j + istart[1]);
				Z = 2 * pi / n[2] * k;
				ptr = i * isize[1] * n[2] + j * n[2] + k;
				numerical_r = a[ptr] / size;
				if (numerical_r != numerical_r)
					numerical_r = 0;
				err += std::abs(numerical_r - solution_laplace(X, Y, Z));
				norm += std::abs(solution_laplace(X, Y, Z));
			}
		}
	}

	double g_err = 0, g_norm = 0;
	MPI_Reduce(&err, &g_err, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
	MPI_Reduce(&norm, &g_norm, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
	PCOUT << "L1 Error of iFF(a)-a: " << g_err << std::endl;
	PCOUT << "Relative L1 Error of iFF(a)-a: " << g_err / g_norm << std::endl;
	if (g_err / g_norm < 1e-10)
		PCOUT << "Results are CORRECT!" << std::endl;
	else
		PCOUT << "\nResults are NOT CORRECT!\n";

	return;
} // end check_err_laplace

void check_err_biharmonic(double* a, int*n, MPI_Comm c_comm) {
	int nprocs, procid;
	MPI_Comm_rank(c_comm, &procid);
	MPI_Comm_size(c_comm, &nprocs);
	long long int size = n[0];
	size *= n[1];
	size *= n[2];
	size = 1;
	double pi = M_PI;

	int istart[3], isize[3], osize[3], ostart[3];
	accfft_local_size_dft_r2c_gpu(n, isize, istart, osize, ostart, c_comm);

	double err = 0, norm = 0;

	double X, Y, Z, numerical_r;
	long int ptr;
	int thid = omp_get_thread_num();
	for (int i = 0; i < isize[0]; i++) {
		for (int j = 0; j < isize[1]; j++) {
			for (int k = 0; k < isize[2]; k++) {
				X = 2 * pi / n[0] * (i + istart[0]);
				Y = 2 * pi / n[1] * (j + istart[1]);
				Z = 2 * pi / n[2] * k;
				ptr = i * isize[1] * n[2] + j * n[2] + k;
				numerical_r = a[ptr] / size;
				if (numerical_r != numerical_r)
					numerical_r = 0;
				err += std::abs(numerical_r - solution_biharmonic(X, Y, Z));
				norm += std::abs(solution_biharmonic(X, Y, Z));
			}
		}
	}

	double g_err = 0, g_norm = 0;
	MPI_Reduce(&err, &g_err, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
	MPI_Reduce(&norm, &g_norm, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
	PCOUT << "L1 Error of iFF(a)-a: " << g_err << std::endl;
	PCOUT << "Relative L1 Error of iFF(a)-a: " << g_err / g_norm << std::endl;
	if (g_err / g_norm < 1e-10)
		PCOUT << "Results are CORRECT!" << std::endl;
	else
		PCOUT << "\nResults are NOT CORRECT!\n";

	return;
} // end check_err_biharmonic
