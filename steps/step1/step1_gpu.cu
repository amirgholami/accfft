#include <stdlib.h>
#include <math.h> // M_PI
#include <mpi.h>
#include <accfft.h>
#include <accfft_gpu.h>

void initialize(double *a,int*n, MPI_Comm c_comm);


void step1_gpu(int *n) {

  int nprocs, procid;
  MPI_Comm_rank(MPI_COMM_WORLD, &procid);
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

  /* Create Cartesian Communicator */
  int c_dims[2];
  MPI_Comm c_comm;
  accfft_create_comm(MPI_COMM_WORLD,c_dims,&c_comm);

  double *data, *data_cpu;
  Complex *data_hat;
  double f_time=0*MPI_Wtime(),i_time=0, setup_time=0;
  int alloc_max=0;

  int isize[3],osize[3],istart[3],ostart[3];
  /* Get the local pencil size and the allocation size */
  alloc_max=accfft_local_size_dft_r2c_gpu(n,isize,istart,osize,ostart,c_comm);

  data_cpu=(double*)accfft_alloc(isize[0]*isize[1]*isize[2]*sizeof(double));
  //data_hat=(Complex*)accfft_alloc(alloc_max);
  cudaMalloc((void**) &data, isize[0]*isize[1]*isize[2]*sizeof(double));
  cudaMalloc((void**) &data_hat, alloc_max);

  //accfft_init(nthreads);
  setup_time=-MPI_Wtime();
  /* Create FFT plan */
  accfft_plan_gpu * plan=accfft_plan_dft_3d_r2c_gpu(n,data,(double*)data_hat,c_comm,ACCFFT_MEASURE);
  setup_time+=MPI_Wtime();

  /*  Initialize data */
  initialize(data_cpu,n,c_comm);
  cudaMemcpy(data, data_cpu, isize[0]*isize[1]*isize[2]*sizeof(double), cudaMemcpyHostToDevice);
  MPI_Barrier(c_comm);

  /* Perform forward FFT */
  f_time-=MPI_Wtime();
  accfft_execute_r2c_gpu(plan,data,data_hat);
  f_time+=MPI_Wtime();

  MPI_Barrier(c_comm);

  double *data2_cpu, *data2;
  cudaMalloc((void**) &data2, isize[0]*isize[1]*isize[2]*sizeof(double));
  data2_cpu=(double*)accfft_alloc(isize[0]*isize[1]*isize[2]*sizeof(double));

  /* Perform backward FFT */
  i_time-=MPI_Wtime();
  accfft_execute_c2r_gpu(plan,data_hat,data2);
  i_time+=MPI_Wtime();

  /* copy back results on CPU */
  cudaMemcpy(data2_cpu, data2, isize[0]*isize[1]*isize[2]*sizeof(double), cudaMemcpyDeviceToHost);  


  /* Check Error */
  double err=0,g_err=0;
  double norm=0,g_norm=0;
  for (int i=0;i<isize[0]*isize[1]*isize[2];++i){
    err+=data2_cpu[i]/n[0]/n[1]/n[2]-data_cpu[i];
    norm+=data2_cpu[i]/n[0]/n[1]/n[2];
  }
  MPI_Reduce(&err,&g_err,1, MPI_DOUBLE, MPI_MAX,0, MPI_COMM_WORLD);
  MPI_Reduce(&norm,&g_norm,1, MPI_DOUBLE, MPI_SUM,0, MPI_COMM_WORLD);

  PCOUT<<"\nComputing FFT/IFFT on GPU"<<std::endl;
  PCOUT<<"\n Error is "<<g_err<<std::endl;
  PCOUT<<"Relative Error is "<<g_err<<std::endl;


  /* Compute some timings statistics */
  double g_f_time, g_i_time, g_setup_time;
  MPI_Reduce(&f_time,&g_f_time,1, MPI_DOUBLE, MPI_MAX,0, MPI_COMM_WORLD);
  MPI_Reduce(&i_time,&g_i_time,1, MPI_DOUBLE, MPI_MAX,0, MPI_COMM_WORLD);
  MPI_Reduce(&setup_time,&g_setup_time,1, MPI_DOUBLE, MPI_MAX,0, MPI_COMM_WORLD);

  PCOUT<<"Timing for FFT of size "<<n[0]<<"*"<<n[1]<<"*"<<n[2]<<std::endl;
  PCOUT<<"Setup \t"<<g_setup_time<<std::endl;
  PCOUT<<"FFT \t"<<g_f_time<<std::endl;
  PCOUT<<"IFFT \t"<<g_i_time<<std::endl;

  accfft_free(data_cpu);
  accfft_free(data2_cpu);
  cudaFree(data);
  cudaFree(data_hat);
  cudaFree(data2);
  accfft_destroy_plan(plan);
  accfft_cleanup();
  MPI_Comm_free(&c_comm);
  return ;

} // end step1_gpu
