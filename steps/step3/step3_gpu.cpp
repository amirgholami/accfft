/*
 * File: step3_gpu.cpp
 * License: Please see LICENSE file.
 * AccFFT: Massively Parallel FFT Library
 * Email: contact@accfft.org
 */
#include <stdlib.h>
#include <math.h>
#include <mpi.h>
#include <cuda_runtime_api.h>
//#include <accfft.h>
#include <accfft_gpu.h>
#define INPLACE // comment this line for outplace transform

void initialize(Complex *a, int*n, MPI_Comm c_comm);
void check_err(Complex* a, int*n, MPI_Comm c_comm);

void step3_gpu(int *n) {

	int nprocs, procid;
	MPI_Comm_rank(MPI_COMM_WORLD, &procid);
	MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

	/* Create Cartesian Communicator */
	int c_dims[2] = { 0 };
	MPI_Comm c_comm;
	accfft_create_comm(MPI_COMM_WORLD, c_dims, &c_comm);

	Complex *data, *data_cpu;
	Complex *data_hat;
	double f_time = 0 * MPI_Wtime(), i_time = 0, setup_time = 0;
	int alloc_max = 0;

	int isize[3], osize[3], istart[3], ostart[3];
	/* Get the local pencil size and the allocation size */
	alloc_max = accfft_local_size_dft_c2c_gpu<double>(
                        n, isize, istart, osize, ostart, c_comm);

#ifdef INPLACE
	data_cpu = (Complex*) malloc(alloc_max);
	cudaMalloc((void**) &data, alloc_max);
#else
	data_cpu=(Complex*)malloc(isize[0]*isize[1]*isize[2]*2*sizeof(double));
	cudaMalloc((void**) &data,isize[0]*isize[1]*isize[2]*2*sizeof(double));
	cudaMalloc((void**) &data_hat, alloc_max);
#endif

	//accfft_init(nthreads);
	setup_time = -MPI_Wtime();

	/* Create FFT plan */
#ifdef INPLACE
	AccFFTd_gpu plan = AccFFTd_gpu(n, data, data, c_comm,
			ACCFFT_MEASURE);
#else
	AccFFTd_gpu plan = AccFFTd_gpu(n,data,data_hat,c_comm,ACCFFT_MEASURE);
#endif
	setup_time += MPI_Wtime();

	/* Warmup Runs */
#ifdef INPLACE
	plan.execute_c2c(ACCFFT_FORWARD, data, data);
	plan.execute_c2c(ACCFFT_FORWARD, data, data);
#else
	plan.execute_c2c(ACCFFT_FORWARD,data,data_hat);
	plan.execute_c2c(ACCFFT_FORWARD,data,data_hat);
#endif

	/*  Initialize data */
	initialize(data_cpu, n, c_comm);
#ifdef INPLACE
	cudaMemcpy(data, data_cpu, alloc_max, cudaMemcpyHostToDevice);
#else
	cudaMemcpy(data, data_cpu,isize[0]*isize[1]*isize[2]*2*sizeof(double), cudaMemcpyHostToDevice);
#endif

	MPI_Barrier(c_comm);

	/* Perform forward FFT */
	f_time -= MPI_Wtime();
#ifdef INPLACE
	plan.execute_c2c(ACCFFT_FORWARD, data, data);
#else
	plan.execute_c2c(ACCFFT_FORWARD,data,data_hat);
#endif
	f_time += MPI_Wtime();

	MPI_Barrier(c_comm);

#ifndef INPLACE
	Complex *data2_cpu, *data2;
	cudaMalloc((void**) &data2, isize[0]*isize[1]*isize[2]*2*sizeof(double));
	data2_cpu=(Complex*) malloc(isize[0]*isize[1]*isize[2]*2*sizeof(double));
#endif

	/* Perform backward FFT */
	i_time -= MPI_Wtime();
#ifdef INPLACE
	plan.execute_c2c(ACCFFT_BACKWARD, data, data);
#else
	plan.execute_c2c(ACCFFT_BACKWARD,data_hat,data2);
#endif
	i_time += MPI_Wtime();

	/* copy back results on CPU and check error*/
#ifdef INPLACE
	cudaMemcpy(data_cpu, data, alloc_max, cudaMemcpyDeviceToHost);
	check_err(data_cpu, n, c_comm);
#else
	cudaMemcpy(data2_cpu, data2, isize[0]*isize[1]*isize[2]*2*sizeof(double), cudaMemcpyDeviceToHost);
	check_err(data2_cpu,n,c_comm);
#endif

	/* Compute some timings statistics */
	double g_f_time, g_i_time, g_setup_time;
	MPI_Reduce(&f_time, &g_f_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
	MPI_Reduce(&i_time, &g_i_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
	MPI_Reduce(&setup_time, &g_setup_time, 1, MPI_DOUBLE, MPI_MAX, 0,
			MPI_COMM_WORLD);

#ifdef INPLACE
	PCOUT << "GPU Timing for Inplace FFT of size " << n[0] << "*" << n[1] << "*"
			<< n[2] << std::endl;
#else
	PCOUT<<"GPU Timing for Outplace FFT of size "<<n[0]<<"*"<<n[1]<<"*"<<n[2]<<std::endl;
#endif
	PCOUT << "Setup \t" << g_setup_time << std::endl;
	PCOUT << "FFT \t" << g_f_time << std::endl;
	PCOUT << "IFFT \t" << g_i_time << std::endl;

	MPI_Barrier(c_comm);
	cudaDeviceSynchronize();
	free(data_cpu);
	cudaFree(data);
#ifndef INPLACE
	cudaFree(data_hat);
	free(data2_cpu);
	cudaFree(data2);
#endif
	accfft_cleanup_gpu();
	MPI_Comm_free(&c_comm);
	return;

} // end step3_gpu

inline double testcase(double X, double Y, double Z) {

	double sigma = 4;
	double pi = M_PI;
	double analytic;
	analytic = std::exp(-sigma * ((X - pi) * (X - pi) + (Y - pi) * (Y - pi)
							+ (Z - pi) * (Z - pi)));
	if (analytic != analytic)
		analytic = 0; /* Do you think the condition will be false always? */
	return analytic;
}

void initialize(Complex *a, int*n, MPI_Comm c_comm) {
	double pi = 4 * atan(1.0);
	int n_tuples = (n[2]);
	int istart[3], isize[3];
	int ostart[3], osize[3];
	accfft_local_size_dft_c2c_gpu<double>(
                        n, isize, istart, osize, ostart, c_comm);
#pragma omp parallel num_threads(16)
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
					ptr = i * isize[1] * n_tuples + j * n_tuples + k;
					a[ptr][0] = testcase(X, Y, Z); //(istart[0]+i)*n_tuples*n[1]+(istart[1]+j)*n_tuples+k+istart[2];//(istart[0]+i)*n[1]+istart[1]+j;//testcase(X,Y,Z);
					a[ptr][1] = testcase(X, Y, Z); //(istart[0]+i)*n_tuples*n[1]+(istart[1]+j)*n_tuples+k+istart[2];//(istart[0]+i)*n[1]+istart[1]+j;//testcase(X,Y,Z);
					//std::cout<<"("<<i<<","<<j<<","<<k<<")  "<<a[k+j*NZ+i*NY*NZ]<<std::endl;
				}
			}
		}
	}
	return;
}

void check_err(Complex* a, int*n, MPI_Comm c_comm) {
	int nprocs, procid;
	MPI_Comm_rank(c_comm, &procid);
	MPI_Comm_size(c_comm, &nprocs);
	long long int size = n[0];
	size *= n[1];
	size *= n[2];
	double pi = 4 * atan(1.0);

	int istart[3], isize[3], osize[3], ostart[3];
	accfft_local_size_dft_c2c_gpu<double>(
                        n, isize, istart, osize, ostart, c_comm);

	double err = 0, norm = 0;

	double X, Y, Z, numerical_r, numerical_c;
	long int ptr;
	for (int i = 0; i < isize[0]; i++) {
		for (int j = 0; j < isize[1]; j++) {
			for (int k = 0; k < isize[2]; k++) {
				X = 2 * pi / n[0] * (i + istart[0]);
				Y = 2 * pi / n[1] * (j + istart[1]);
				Z = 2 * pi / n[2] * k;
				ptr = i * isize[1] * n[2] + j * n[2] + k;
				numerical_r = a[ptr][0] / size;
				if (numerical_r != numerical_r)
					numerical_r = 0;
				numerical_c = a[ptr][1] / size;
				if (numerical_c != numerical_c)
					numerical_c = 0;
				err += std::abs(numerical_r - testcase(X, Y, Z))
						+ std::abs(numerical_c - testcase(X, Y, Z));
				norm += std::abs(testcase(X, Y, Z));
			}
		}
	}

	double g_err = 0, g_norm = 0;
	MPI_Reduce(&err, &g_err, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
	MPI_Reduce(&norm, &g_norm, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
	PCOUT << "\nL1 Error of iFF(a)-a: " << g_err << std::endl;
	PCOUT << "Relative L1 Error of iFF(a)-a: " << g_err / g_norm << std::endl;
	if (g_err / g_norm < 1e-10)
		PCOUT << "\nResults are CORRECT!\n\n";
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

	step3_gpu(N);

	MPI_Finalize();
	return 0;
} // end main
