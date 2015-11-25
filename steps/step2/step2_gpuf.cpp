/*
 * File: step2_gpu.cpp
 * License: Please see LICENSE file.
 * AccFFT: Massively Parallel FFT Library
 * Created by Amir Gholami on 11/23/2015
 * Email: contact@accfft.org
 */
#include <stdlib.h>
#include <math.h>
#include <mpi.h>
#include <cuda_runtime_api.h>
//#include <accfft.h>
#include <accfft_gpuf.h>

void initialize_gpu(float *a,int*n, int* isize, int * istart);
void check_err(float* a,int*n,MPI_Comm c_comm);
float testcase(float X,float Y,float Z);


void step2_gpu(int *n) {

  int nprocs, procid;
  MPI_Comm_rank(MPI_COMM_WORLD, &procid);
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

  /* Create Cartesian Communicator */
  int c_dims[2];
  MPI_Comm c_comm;
  accfft_create_comm(MPI_COMM_WORLD,c_dims,&c_comm);

  float *data, *data_cpu;
  Complexf *data_hat;
  double f_time=0*MPI_Wtime(),i_time=0, setup_time=0;
  int alloc_max=0;

  int isize[3],osize[3],istart[3],ostart[3];
  /* Get the local pencil size and the allocation size */
  alloc_max=accfft_local_size_dft_r2c_gpuf(n,isize,istart,osize,ostart,c_comm);

  /* Note that both need to be allocated by alloc_max because of inplace transform*/
  data_cpu=(float*)malloc(alloc_max);
  cudaMalloc((void**) &data, alloc_max);

  //accfft_init(nthreads);

  /* Create FFT plan */
  setup_time=-MPI_Wtime();
  accfft_plan_gpuf * plan=accfft_plan_dft_3d_r2c_gpuf(n,data,data,c_comm,ACCFFT_MEASURE);
  setup_time+=MPI_Wtime();

  /* Warm Up */
  accfft_execute_r2c_gpuf(plan,data,(Complexf*)data);
  accfft_execute_r2c_gpuf(plan,data,(Complexf*)data);

  /* Initialize data */
  initialize_gpu(data,n,isize,istart);
  MPI_Barrier(c_comm);

  /* Perform forward FFT */
  f_time-=MPI_Wtime();
  accfft_execute_r2c_gpuf(plan,data,(Complexf*)data);
  f_time+=MPI_Wtime();
  MPI_Barrier(c_comm);

  /* Perform backward FFT */
  i_time-=MPI_Wtime();
  accfft_execute_c2r_gpuf(plan,(Complexf*)data,data);
  i_time+=MPI_Wtime();

  /* copy back results on CPU */
  cudaMemcpy(data_cpu, data,alloc_max, cudaMemcpyDeviceToHost);

  /* Check Error */
  check_err(data_cpu,n,c_comm);

  /* Compute some timings statistics */
  double g_f_time, g_i_time, g_setup_time;
  MPI_Reduce(&f_time,&g_f_time,1, MPI_DOUBLE, MPI_MAX,0, MPI_COMM_WORLD);
  MPI_Reduce(&i_time,&g_i_time,1, MPI_DOUBLE, MPI_MAX,0, MPI_COMM_WORLD);
  MPI_Reduce(&setup_time,&g_setup_time,1, MPI_DOUBLE, MPI_MAX,0, MPI_COMM_WORLD);

  PCOUT<<"GPU Timing for FFT of size "<<n[0]<<"*"<<n[1]<<"*"<<n[2]<<std::endl;
  PCOUT<<"Setup \t"<<g_setup_time<<std::endl;
  PCOUT<<"FFT \t"<<g_f_time<<std::endl;
  PCOUT<<"IFFT \t"<<g_i_time<<std::endl;

  free(data_cpu);
  cudaFree(data);
  accfft_destroy_plan_gpu(plan);
  accfft_cleanup_gpuf();
  MPI_Comm_free(&c_comm);
  return ;

} // end step2_gpu


void check_err(float* a,int*n,MPI_Comm c_comm){
  int nprocs, procid;
  MPI_Comm_rank(c_comm, &procid);
  MPI_Comm_size(c_comm,&nprocs);

  long long int size=n[0];
  size*=n[1]; size*=n[2];
  float pi=M_PI;

  // Note that n2_ is the padded version of n2 which is
  // the spatial size of the array. To access the spatial members
  // we must use n2_, because that is how it is written in memory!
  int n2_=(n[2]/2+1)*2;
  // Get the local pencil size and the allocation size
  int istart[3], isize[3], osize[3],ostart[3];
  accfft_local_size_dft_r2c_gpuf(n,isize,istart,osize,ostart,c_comm);

  float err=0,norm=0;
  {
    float X,Y,Z,numerical;
    long int ptr;
    for (int i=0; i<isize[0]; i++){
      for (int j=0; j<isize[1]; j++){
        for (int k=0; k<isize[2]; k++){
          X=2*pi/n[0]*(i+istart[0]);
          Y=2*pi/n[1]*(j+istart[1]);
          Z=2*pi/n[2]*k;
          ptr=i*isize[1]*n2_+j*n2_+k;
          numerical=a[ptr]/size; if(numerical!=numerical) numerical=0;
          err+=std::abs(numerical-testcase(X,Y,Z));
          norm+=std::abs(testcase(X,Y,Z));
          //std::cout<<"("<<i<<","<<j<<","<<k<<")  "<<numerical<<'\t'<<testcase(X,Y,Z)<<std::endl;
        }
      }
    }
  }

  float g_err=0,g_norm=0;
  MPI_Reduce(&err,&g_err,1, MPI_FLOAT, MPI_SUM,0, MPI_COMM_WORLD);
  MPI_Reduce(&norm,&g_norm,1, MPI_FLOAT, MPI_SUM,0, MPI_COMM_WORLD);
  PCOUT<<"\nL1 Error of iFF(a)-a: "<<g_err<<std::endl;
  PCOUT<<"Relative L1 Error of iFF(a)-a: "<<g_err/g_norm<<std::endl;
  if (g_err/g_norm< 1e-5)
    PCOUT<<"\nResults are CORRECT! (upto single precision)\n\n";
  else
    PCOUT<<"\nResults are NOT CORRECT!\n\n";

  return;
} // end check_err
int main(int argc, char **argv)
{
  int NX,NY,NZ;
  MPI_Init (&argc, &argv);
  int nprocs, procid;
  MPI_Comm_rank(MPI_COMM_WORLD, &procid);
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

  /* Parsing Inputs  */
  if(argc==1){
    NX=128;NY=128;NZ=128;
  }
  else{
    NX=atoi(argv[1]); NY=atoi(argv[2]); NZ=atoi(argv[3]);
  }
  int N[3]={NX,NY,NZ};

  step2_gpu(N);

  MPI_Finalize();
  return 0;
} // end main
