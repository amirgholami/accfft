/*
 * File: step2.cpp
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
#include <iomanip>  // cout width
#include <bitset>
#define NTH 4
#define VERBOSE 0

int threads_ok;
typedef double Complex[2];

inline double testcase(double X,double Y,double Z){
  double sigma= 4;
  double pi=4*atan(1.0);
  double analytic=0;
  analytic= (std::exp( -sigma * ( (X-pi)*(X-pi) + (Y-pi)*(Y-pi) +(Z-pi)*(Z-pi)  )));
  if(analytic!=analytic) analytic=0;
  return analytic;
}
void initialize(double *a,int*n, MPI_Comm c_comm);
void check_err(double* a,int*n,MPI_Comm c_comm);
void step1(int *n, int nthreads);

void initialize(double *a,int*n, MPI_Comm c_comm) {
  int nprocs, procid;
  MPI_Comm_rank(c_comm, &procid);
  MPI_Comm_size(c_comm,&nprocs);

  double pi=4*atan(1.0);
  int n_tuples=(n[2]/2+1)*2;
  // Get the local pencil size and the allocation size
  int istart[3], isize[3], osize[3],ostart[3];
  accfft_local_size_dft_r2c(n,isize,istart,osize,ostart,c_comm);
  //PCOUT<<"isize[0]="<<isize[0]<<" isize[1]="<<isize[1]<<" isize[2]="<<isize[2]<<std::endl;
  //PCOUT<<"istart[0]="<<istart[0]<<" istart[1]="<<istart[1]<<" istart[2]="<<istart[2]<<std::endl;

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
          a[ptr]=testcase(X,Y,Z);
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

  int n_tuples=(n[2]/2+1)*2;
  // Get the local pencil size and the allocation size
  int istart[3], isize[3], osize[3],ostart[3];
  accfft_local_size_dft_r2c(n,isize,istart,osize,ostart,c_comm);

    double err,norm;
    {
      double X,Y,Z,numerical;
      long int ptr;
      for (int i=0; i<isize[0]; i++){
        for (int j=0; j<isize[1]; j++){
          for (int k=0; k<isize[2]; k++){
            X=2*pi/n[0]*(i+istart[0]);
            Y=2*pi/n[1]*(j+istart[1]);
            Z=2*pi/n[2]*k;
            ptr=i*isize[1]*n_tuples+j*n_tuples+k;
            numerical=a[ptr]/size; if(numerical!=numerical) numerical=0;
            err+=std::abs(numerical-testcase(X,Y,Z));
            norm+=std::abs(testcase(X,Y,Z));
            //std::cout<<"("<<i<<","<<j<<","<<k<<")  "<<numerical<<'\t'<<testcase(X,Y,Z)<<std::endl;
          }
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

void step1(int *n, int nthreads) {
  int nprocs, procid;
  MPI_Comm_rank(MPI_COMM_WORLD, &procid);
  MPI_Comm_size(MPI_COMM_WORLD,&nprocs);

  /* Create Cartesian Communicator */
  int c_dims[2];
  MPI_Comm c_comm;
  accfft_create_comm(MPI_COMM_WORLD,c_dims,&c_comm);

  double fft_time=0*MPI_Wtime(),timings[5]={0}, dummy_timings[5]={0},setup_time=0;

  MPI_Barrier(MPI_COMM_WORLD);

  int alloc_local;
  int alloc_max=0;

  double *data;
  int isize[3],osize[3],istart[3],ostart[3];
  /* Get the local pencil size and the allocation size */
  alloc_max=accfft_local_size_dft_r2c(n,isize,istart,osize,ostart,c_comm);
  data=(double*)accfft_alloc(alloc_max);



  int coords[2],np[2],periods[2];
  MPI_Cart_get(c_comm,2,np,periods,coords);

  accfft_init(nthreads);
  setup_time=-MPI_Wtime();
  /* Create FFT plan */
  accfft_plan * plan=accfft_plan_dft_3d_r2c(n,data,data,c_comm,NULL);
  setup_time+=MPI_Wtime();


  MPI_Barrier(c_comm);

  /*  Initialize data */
  initialize(data,n,c_comm);
  MPI_Barrier(c_comm);

  /* Perform forward FFT */
  accfft_execute_r2c(plan,data,(Complex*)data,timings);
  MPI_Barrier(c_comm);

  fft_time=timings[4];

  /* Perform backward FFT */
  accfft_execute_c2r(plan,(Complex*)data,data);

  /* Check Error */
  check_err(data,n,c_comm);

  /* Compute some timings statistics */
  int nrep=1;
  double g_fft_time=0,g_timings[4],g_setup_time=0;
  MPI_Reduce(&fft_time,&g_fft_time,1, MPI_DOUBLE, MPI_MAX,0, MPI_COMM_WORLD);
  MPI_Reduce(timings,g_timings,4, MPI_DOUBLE, MPI_MAX,0, MPI_COMM_WORLD);
  MPI_Reduce(&setup_time,&g_setup_time,1, MPI_DOUBLE, MPI_MAX,0, MPI_COMM_WORLD);


  ptrdiff_t size=n[0];size*=n[1]; size*=n[2];
  double gflops=2.5*size*(log2(n[2]) +log2(n[0])+ log2(n[1]) )/(g_fft_time+g_timings[0])*nrep/1e9;

  std::cout.precision(4);
  PCOUT<<"np \t"<<"Grid \t"<<"Total"<<'\t'<<"FFT"<<'\t'<<"Shuffle"<<'\t'<<"Comm"<<'\t'<<"ReShuffle"<<'\t'<<"Setup"<<'\t'<<"threads"<<'\t'<<"T Method"<<'\t'<<" kway"<<'\t'<<"Async \t GFlops"<<std::endl;
  PCOUT<<nprocs<<'\t'<<c_dims[0]<<"*"<<c_dims[1]<<'\t'<<(g_fft_time+g_timings[0])/nrep<<'\t'<<g_fft_time/nrep<<'\t'<<g_timings[1]/nrep<<'\t'<<g_timings[2]/nrep<<'\t'<<g_timings[3]/nrep<<'\t'<<'\t'<<g_setup_time<<'\t'<<nthreads<<'\t'<<plan->T_plan_1->method<<'\t'<<plan->T_plan_1->kway<<'\t'<<plan->T_plan_1->kway_async<<'\t'<<gflops<<std::endl;

  accfft_free(data);
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

