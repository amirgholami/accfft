/*
 * File: step1f_gpu.cpp
 * License: Please see LICENSE file.
 * AccFFT: Massively Parallel FFT Library
 * Created by Amir Gholami on 11/23/2015
 * Email: contact@accfft.org
 */

#include <stdlib.h>
#include <math.h> // M_PI
#include <mpi.h>
#include <cuda_runtime_api.h>
#include <accfft_gpu.h> // single precision

void initialize(float *a, int*n, MPI_Comm c_comm);
void initialize_gpu(float *a, int*n, int* isize, int * istart);
void check_err(float* a, int*n, MPI_Comm c_comm);

void step1_gpu(int *n) {

	int nprocs, procid;
	MPI_Comm_rank(MPI_COMM_WORLD, &procid);
	MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

	/* Create Cartesian Communicator */
	int c_dims[2] = { 0 };
	MPI_Comm c_comm;
	accfft_create_comm(MPI_COMM_WORLD, c_dims, &c_comm);

	float *data, *data_cpu;
	Complexf *data_hat;
	double f_time = 0 * MPI_Wtime(), i_time = 0, setup_time = 0;
	int alloc_max = 0;

	int isize[3], osize[3], istart[3], ostart[3];
	/* Get the local pencil size and the allocation size */
	alloc_max = accfft_local_size_dft_r2c_gpu<float>(
                        n, isize, istart, osize, ostart, c_comm);

	data_cpu = (float*) malloc(isize[0] * isize[1] * isize[2] * sizeof(float));
	//data_hat=(Complexf*)accfft_alloc(alloc_max);
	cudaMalloc((void**) &data, isize[0] * isize[1] * isize[2] * sizeof(float));
	cudaMalloc((void**) &data_hat, alloc_max);

	//accfft_init(nthreads);

	/* Create FFT plan */
	setup_time = -MPI_Wtime();
	AccFFTs_gpu plan = AccFFTs_gpu(n, data,
			data_hat, c_comm, ACCFFT_MEASURE);
	setup_time += MPI_Wtime();

	/* Warm Up */
	plan.execute_r2c(data, data_hat);
	plan.execute_r2c(data, data_hat);
	plan.execute_r2c(data, data_hat);
	plan.execute_r2c(data, data_hat);
	plan.execute_r2c(data, data_hat);

	/*  Initialize data */
	initialize(data_cpu, n, c_comm);
	cudaMemcpy(data, data_cpu, isize[0] * isize[1] * isize[2] * sizeof(float),
			cudaMemcpyHostToDevice);
	// initialize_gpu(data,n,isize,istart); // GPU version of initialize function

	MPI_Barrier(c_comm);

	/* Perform forward FFT */
	f_time -= MPI_Wtime();
	plan.execute_r2c(data, data_hat);
	f_time += MPI_Wtime();

	MPI_Barrier(c_comm);

	float *data2_cpu, *data2;
	cudaMalloc((void**) &data2, isize[0] * isize[1] * isize[2] * sizeof(float));
	data2_cpu = (float*) malloc(isize[0] * isize[1] * isize[2] * sizeof(float));

	/* Perform backward FFT */
	i_time -= MPI_Wtime();
	plan.execute_c2r(data_hat, data2);
	i_time += MPI_Wtime();

	/* copy back results on CPU */
	cudaMemcpy(data2_cpu, data2, isize[0] * isize[1] * isize[2] * sizeof(float),
			cudaMemcpyDeviceToHost);

	/* Check Error */
	check_err(data2_cpu, n, c_comm);

	/* Compute some timings statistics */
	double g_f_time, g_i_time, g_setup_time;
	MPI_Reduce(&f_time, &g_f_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
	MPI_Reduce(&i_time, &g_i_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
	MPI_Reduce(&setup_time, &g_setup_time, 1, MPI_DOUBLE, MPI_MAX, 0,
			MPI_COMM_WORLD);

	PCOUT << "GPU Timing for FFT of size " << n[0] << "*" << n[1] << "*" << n[2]
			<< std::endl;
	PCOUT << "Setup \t" << g_setup_time << std::endl;
	PCOUT << "FFT \t" << g_f_time << std::endl;
	PCOUT << "IFFT \t" << g_i_time << std::endl;

	free(data_cpu);
	free(data2_cpu);
	cudaFree(data);
	cudaFree(data_hat);
	cudaFree(data2);
	accfft_cleanup_gpu();
	MPI_Comm_free(&c_comm);
	return;

} // end step1_gpu

inline float testcase(float X, float Y, float Z) {

	float sigma = 4;
	float pi = M_PI;
	float analytic;
	analytic = std::exp(-sigma * ((X - pi) * (X - pi) + (Y - pi) * (Y - pi)
							+ (Z - pi) * (Z - pi)));
	if (analytic != analytic)
		analytic = 0; /* Do you think the condition will be false always? */
	return analytic;
}
void initialize(float *a, int *n, MPI_Comm c_comm) {
	float pi = M_PI;
	int istart[3], isize[3], osize[3], ostart[3];
	accfft_local_size_dft_r2c_gpu<float>(n, isize, istart, osize, ostart, c_comm);

#pragma omp parallel
	{
		float X, Y, Z;
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

void check_err(float* a, int*n, MPI_Comm c_comm) {
	int nprocs, procid;
	MPI_Comm_rank(c_comm, &procid);
	MPI_Comm_size(c_comm, &nprocs);
	long long int size = n[0];
	size *= n[1];
	size *= n[2];
	float pi = M_PI;

	int istart[3], isize[3], osize[3], ostart[3];
	accfft_local_size_dft_r2c_gpu<float>(n, isize, istart, osize, ostart, c_comm);

	float err = 0, norm = 0;

	float X, Y, Z, numerical_r;
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
				err += std::abs(numerical_r - testcase(X, Y, Z));
				norm += std::abs(testcase(X, Y, Z));
			}
		}
	}

	float g_err = 0, g_norm = 0;
	MPI_Reduce(&err, &g_err, 1, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
	MPI_Reduce(&norm, &g_norm, 1, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
	PCOUT << "\nL1 Error of iFF(a)-a: " << g_err << std::endl;
	PCOUT << "Relative L1 Error of iFF(a)-a: " << g_err / g_norm << std::endl;
	if (g_err / g_norm < 1e-5)
		PCOUT << "\nResults are CORRECT! (upto single precision)\n\n";
	else
		PCOUT << "\nResults are NOT CORRECT!\n\n";

	return;
} // end check_err

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

	step1_gpu(N);

	MPI_Finalize();
	return 0;
} // end main
