/*
 * File: step1.cpp
 * Project: AccFFT
 * Created by Amir Gholami on 12/23/2014
 * Contact: contact@accfft.org
 * Copyright (c) 2014-2015
 */

#include <mpi.h>
#include <accfft.h>
#include <iostream>
#include <cstdlib>

#include <cmath>
#include <math.h>
#include <cstring>
#include <sstream>
#include <fstream>
#include <iomanip>
#include <bitset>
#define NTH 10
#define VERBOSE 0
#define TIME_BACKWARD 0

int threads_ok;
typedef double Complex[2];

void initialize(double *a,int*n, MPI_Comm c_comm);
void check_err(double* a,int*n,MPI_Comm c_comm);
void step1(int *n, int nthreads);
inline double testcase(double X,double Y,double Z){

  double sigma= 4;
  double pi=4*atan(1.0);
  double analytic=0;
  analytic= (std::exp( -sigma * ( (X-pi)*(X-pi) + (Y-pi)*(Y-pi) +(Z-pi)*(Z-pi)  )));
  if(analytic!=analytic) analytic=0;
  //return sin(X);
  return analytic;
}

void step1(int *n, int nthreads) {
  int nprocs, procid;
  MPI_Comm_rank(MPI_COMM_WORLD, &procid);
  MPI_Comm_size(MPI_COMM_WORLD,&nprocs);

  /* Create Cartesian Communicator */
  int c_dims[2];
  MPI_Comm c_comm;
  accfft_create_comm(MPI_COMM_WORLD,c_dims,&c_comm);

  double f_time=0*MPI_Wtime(),i_time=0, setup_time=0;

  MPI_Barrier(MPI_COMM_WORLD);

  int alloc_max=0;

  double *data,*data_hat;

  int isize[3],osize[3],istart[3],ostart[3];
  /* Get the local pencil size and the allocation size */
  alloc_max=accfft_local_size_dft_r2c(n,isize,istart,osize,ostart,c_comm);

  data=(double*)accfft_alloc(alloc_max*sizeof(double));
  data_hat=(double*)accfft_alloc(alloc_max*sizeof(double));



  int coords[2],np[2],periods[2];
  MPI_Cart_get(c_comm,2,np,periods,coords);

  accfft_init(nthreads);
  setup_time=-MPI_Wtime();
  /* Create FFT plan */
  accfft_plan * plan=accfft_plan_dft_3d_r2c(n,data,data_hat,c_comm,NULL);
  setup_time+=MPI_Wtime();


  MPI_Barrier(c_comm);
  /*  Initialize data */
  initialize(data,n,c_comm);
  MPI_Barrier(c_comm);

  /* Perform forward FFT */
  f_time-=MPI_Wtime();
  accfft_execute(plan,-1,data,data_hat);
  f_time+=MPI_Wtime();

  MPI_Barrier(c_comm);


  /* Perform backward FFT */
  i_time-=MPI_Wtime();
  accfft_execute(plan,1,data_hat,data);
  i_time+=MPI_Wtime();

  /* Check Error */
  check_err(data,n,c_comm);

  /* Compute some timings statistics */
  double g_f_time, g_i_time, g_setup_time;
  MPI_Reduce(&f_time,&g_f_time,1, MPI_DOUBLE, MPI_MAX,0, MPI_COMM_WORLD);
  MPI_Reduce(&i_time,&g_i_time,1, MPI_DOUBLE, MPI_MAX,0, MPI_COMM_WORLD);
  MPI_Reduce(&setup_time,&g_setup_time,1, MPI_DOUBLE, MPI_MAX,0, MPI_COMM_WORLD);

  PCOUT<<"Timing for FFT of size "<<n[0]<<"*"<<n[1]<<"*"<<n[2]<<std::endl;
  PCOUT<<"Setup \t"<<g_setup_time<<std::endl;
  PCOUT<<"FFT \t"<<g_f_time<<std::endl;
  PCOUT<<"IFFT \t"<<g_i_time<<std::endl;

  accfft_free(data);
  accfft_free(data_hat);
  accfft_destroy_plan(plan);
  accfft_cleanup();
  MPI_Comm_free(&c_comm);
  return ;
}


int main(int argc, char **argv)
{

  int NX,NY,NZ;
  MPI_Init (&argc, &argv);
  int nprocs, procid;
  MPI_Comm_rank(MPI_COMM_WORLD, &procid);
  MPI_Comm_size(MPI_COMM_WORLD,&nprocs);

  /* Parsing Inputs  */
  if(argc==1){
    NX=128;NY=128;NZ=128;
  }
  else{
    NX=atoi(argv[1]); NY=atoi(argv[2]); NZ=atoi(argv[3]);
  }
  int N[3]={NX,NY,NZ};

  int nthreads=1;
  step1(N,nthreads);

  MPI_Finalize();
  return 0;
}



void initialize(double *a,int*n, MPI_Comm c_comm) {
  double pi=4*atan(1.0);
  int n_tuples=n[2];
  int istart[3], isize[3], osize[3],ostart[3];
  accfft_local_size_dft_r2c(n,isize,istart,osize,ostart,c_comm);

#pragma omp parallel
  {
    double X,Y,Z;
    long int ptr;
#pragma omp for
    for (int i=0; i<isize[0]; i++){
      for (int j=0; j<isize[1]; j++){
        for (int k=0; k<isize[2]; k++){
          X=2*pi/n[0]*(i+istart[0]);
          Y=2*pi/n[1]*(j+istart[1]);
          Z=2*pi/n[2]*k;
          ptr=i*isize[1]*n_tuples+j*n_tuples+k;
          a[ptr]=testcase(X,Y,Z);//(istart[0]+i)*n_tuples*n[1]+(istart[1]+j)*n_tuples+k+istart[2];//(istart[0]+i)*n[1]+istart[1]+j;//testcase(X,Y,Z);
          //std::cout<<"("<<i<<","<<j<<","<<k<<")  "<<a[k+j*NZ+i*NY*NZ]<<std::endl;
        }
      }
    }
  }
  return;
}
void check_err(double* a,int*n,MPI_Comm c_comm){
  int nprocs, procid;
  MPI_Comm_rank(c_comm, &procid);
  MPI_Comm_size(c_comm,&nprocs);
  long long int size=n[0];
  size*=n[1]; size*=n[2];
  double pi=4*atan(1.0);

  int n_tuples=n[2];
  int istart[3], isize[3], osize[3],ostart[3];
  accfft_local_size_dft_r2c(n,isize,istart,osize,ostart,c_comm);

  double err,norm;

  double X,Y,Z,numerical_r;
  long int ptr;
  int thid=omp_get_thread_num();
  for (int i=0; i<isize[0]; i++){
    for (int j=0; j<isize[1]; j++){
      for (int k=0; k<isize[2]; k++){
        X=2*pi/n[0]*(i+istart[0]);
        Y=2*pi/n[1]*(j+istart[1]);
        Z=2*pi/n[2]*k;
        ptr=i*isize[1]*n_tuples+j*n_tuples+k;
        numerical_r=a[ptr]/size; if(numerical_r!=numerical_r) numerical_r=0;
        err+=std::abs(numerical_r-testcase(X,Y,Z));
        norm+=std::abs(testcase(X,Y,Z));

        //PCOUT<<"("<<i<<","<<j<<","<<k<<")  "<<numerical<<'\t'<<testcase(X,Y,Z)<<std::endl;
      }
    }
  }

  double gerr=0,gnorm=0;
  MPI_Reduce(&err,&gerr,1, MPI_DOUBLE, MPI_SUM,0, MPI_COMM_WORLD);
  MPI_Reduce(&norm,&gnorm,1, MPI_DOUBLE, MPI_SUM,0, MPI_COMM_WORLD);
  PCOUT<<"The L1 error between iFF(a)-a is = "<<gerr<<std::endl;
  PCOUT<<"The Rel. L1 error between iFF(a)-a is = "<<gerr/gnorm<<std::endl;
  if(gerr/gnorm>1e-10){
    PCOUT<<"\033[1;31m ERROR!!! FFT not computed correctly!\033[0m"<<std::endl;
  }
  else{
    PCOUT<<"\033[1;36m FFT computed correctly!\033[0m"<<std::endl;
  }

}
