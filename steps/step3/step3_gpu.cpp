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

void initialize(Complex *a,int*n, MPI_Comm c_comm);
void check_err(Complex* a,int*n,MPI_Comm c_comm);


void step3_gpu(int *n) {

  int nprocs, procid;
  MPI_Comm_rank(MPI_COMM_WORLD, &procid);
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

  /* Create Cartesian Communicator */
  int c_dims[2];
  c_dims[0]=1;c_dims[1]=5;
  MPI_Comm c_comm;
  accfft_create_comm(MPI_COMM_WORLD,c_dims,&c_comm);

  Complex *data, *data_cpu;
  Complex *data_hat;
  double f_time=0*MPI_Wtime(),i_time=0, setup_time=0;
  int alloc_max=0;

  int isize[3],osize[3],istart[3],ostart[3];
  /* Get the local pencil size and the allocation size */
  alloc_max=accfft_local_size_dft_c2c_gpu(n,isize,istart,osize,ostart,c_comm);

  data_cpu=(Complex*)malloc(isize[0]*isize[1]*isize[2]*2*sizeof(double));
  cudaMalloc((void**) &data,isize[0]*isize[1]*isize[2]*2*sizeof(double));
  cudaMalloc((void**) &data_hat, alloc_max);

  //accfft_init(nthreads);
  setup_time=-MPI_Wtime();
  /* Create FFT plan */
  accfft_plan_gpu * plan=accfft_plan_dft_3d_c2c_gpu(n,data,data_hat,c_comm,ACCFFT_MEASURE);
  setup_time+=MPI_Wtime();

  /*  Initialize data */
  initialize(data_cpu,n,c_comm);
  cudaMemcpy(data, data_cpu,isize[0]*isize[1]*isize[2]*2*sizeof(double), cudaMemcpyHostToDevice);

  MPI_Barrier(c_comm);

  /* Perform forward FFT */
  f_time-=MPI_Wtime();
  accfft_execute_c2c_gpu(plan,ACCFFT_FORWARD,data,data_hat);
  f_time+=MPI_Wtime();

  MPI_Barrier(c_comm);

  Complex *data2_cpu, *data2;
  cudaMalloc((void**) &data2, isize[0]*isize[1]*isize[2]*2*sizeof(double));
  data2_cpu=(Complex*) malloc(isize[0]*isize[1]*isize[2]*2*sizeof(double));

  /* Perform backward FFT */
  i_time-=MPI_Wtime();
  accfft_execute_c2c_gpu(plan,ACCFFT_BACKWARD,data_hat,data2);
  i_time+=MPI_Wtime();

  /* copy back results on CPU */
  cudaMemcpy(data2_cpu, data2, isize[0]*isize[1]*isize[2]*2*sizeof(double), cudaMemcpyDeviceToHost);


  /* Check Error */
  check_err(data2_cpu,n,c_comm);

  /* Compute some timings statistics */
  double g_f_time, g_i_time, g_setup_time;
  MPI_Reduce(&f_time,&g_f_time,1, MPI_DOUBLE, MPI_MAX,0, MPI_COMM_WORLD);
  MPI_Reduce(&i_time,&g_i_time,1, MPI_DOUBLE, MPI_MAX,0, MPI_COMM_WORLD);
  MPI_Reduce(&setup_time,&g_setup_time,1, MPI_DOUBLE, MPI_MAX,0, MPI_COMM_WORLD);

  PCOUT<<"GPU Timing for FFT of size "<<n[0]<<"*"<<n[1]<<"*"<<n[2]<<std::endl;
  PCOUT<<"Setup \t"<<g_setup_time<<std::endl;
  PCOUT<<"FFT \t"<<g_f_time<<std::endl;
  PCOUT<<"IFFT \t"<<g_i_time<<std::endl;

  MPI_Barrier(c_comm);
  cudaDeviceSynchronize();
  free(data_cpu);
  free(data2_cpu);
  cudaFree(data);
  cudaFree(data_hat);
  cudaFree(data2);
  accfft_destroy_plan_gpu(plan);
  accfft_cleanup_gpu();
  MPI_Comm_free(&c_comm);
  return ;

} // end step3_gpu

inline double testcase(double X,double Y,double Z){

  double sigma= 4;
  double pi=M_PI;
  double analytic;
  analytic= std::exp( -sigma * ( (X-pi)*(X-pi) + (Y-pi)*(Y-pi) + (Z-pi)*(Z-pi) ));
  if(analytic!=analytic) analytic=0; /* Do you think the condition will be false always? */
  return analytic;
}

void initialize(Complex *a,int*n, MPI_Comm c_comm) {
  double pi=4*atan(1.0);
  int n_tuples=(n[2]);
  int istart[3], isize[3];
  int ostart[3], osize[3];
  accfft_local_size_dft_c2c_gpu(n,isize,istart,osize,ostart,c_comm);
#pragma omp parallel num_threads(16)
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
          a[ptr][0]=testcase(X,Y,Z);//(istart[0]+i)*n_tuples*n[1]+(istart[1]+j)*n_tuples+k+istart[2];//(istart[0]+i)*n[1]+istart[1]+j;//testcase(X,Y,Z);
          a[ptr][1]=testcase(X,Y,Z);//(istart[0]+i)*n_tuples*n[1]+(istart[1]+j)*n_tuples+k+istart[2];//(istart[0]+i)*n[1]+istart[1]+j;//testcase(X,Y,Z);
          //std::cout<<"("<<i<<","<<j<<","<<k<<")  "<<a[k+j*NZ+i*NY*NZ]<<std::endl;
        }
      }
    }
  }
  return;
}
void check_err(Complex* a,int*n,MPI_Comm c_comm){
  ptrdiff_t size=n[0];
  size*=n[1]; size*=n[2];

  int NX=n[0], NY=n[1], NZ=n[2];
  double pi=4*atan(1.0);
  int nprocs, procid;
  int istart[3], isize[3];
  int ostart[3], osize[3];

  MPI_Comm_rank(c_comm, &procid);
  MPI_Comm_size(c_comm,&nprocs);
  accfft_local_size_dft_c2c_gpu(n,isize,istart,osize,ostart,c_comm);

  if(1){
    int num_th= omp_get_max_threads();
    double err[num_th],norm[num_th];
    for(int i=0;i<num_th;i++){
      err[i]=0.;
      norm[i]=0.;
    }

    int counter=0;
#pragma omp parallel num_threads(1)
    {
      double X,Y,Z,numerical_r,numerical_c;
      long int ptr;
      int thid=omp_get_thread_num();
#pragma omp for
      for (int i=0; i<isize[0]; i++){
        for (int j=0; j<isize[1]; j++){
          for (int k=0; k<isize[2]; k++){
            X=2*pi/n[0]*(i+istart[0]);
            Y=2*pi/n[1]*(j+istart[1]);
            Z=2*pi/n[2]*k;
            ptr=i*isize[1]*n[2]+j*n[2]+k;
            numerical_r=a[ptr][0]/size; if(numerical_r!=numerical_r) numerical_r=0;
            numerical_c=a[ptr][1]/size; if(numerical_c!=numerical_c) numerical_c=0;
            err[thid]+=std::abs(numerical_r-testcase(X,Y,Z))+std::abs(numerical_c-testcase(X,Y,Z));
            norm[thid]+=std::abs(testcase(X,Y,Z));
            //PCOUT<<a[k+j*(NZ)+i*NY*(NZ)][0]/size<<" \t "<<testcase(X,Y,Z)<<std::endl;
            if(err[thid]>1e-4 && counter<=10){
              PCOUT<<"("<<i+istart[0]<<","<<j+istart[1]<<","<<k<<")  "<<numerical_r<<","<<numerical_c<<'\t'<<testcase(X,Y,Z)<<std::endl;
              counter++;
            }
            //if(procid==0)  std::cout<<"("<<i<<","<<j<<","<<k<<")  "<<numerical_r<<","<<numerical_c<<'\t'<<testcase(X,Y,Z)<<std::endl;
          }}
      }
    }

    for(int i=1;i<num_th;i++){
      err[0]+=err[i];
      norm[0]+=norm[i];
    }

    double gerr=0,gnorm=0;
    MPI_Reduce(&err[0],&gerr,1, MPI_DOUBLE, MPI_SUM,0, MPI_COMM_WORLD);
    MPI_Reduce(&norm[0],&gnorm,1, MPI_DOUBLE, MPI_SUM,0, MPI_COMM_WORLD);
    PCOUT<<"The L1 error between iFF(a)-a is = "<<gerr<<std::endl;
    PCOUT<<"The Rel. L1 error between iFF(a)-a is = "<<gerr/gnorm<<std::endl;
    if(gerr/gnorm>1e-10){
      PCOUT<<"\033[1;31m ERROR!!! FFT not computed correctly!\033[0m"<<std::endl;
    }
    else{
      PCOUT<<"\033[1;36m FFT computed correctly!\033[0m"<<std::endl;
    }

  }
  if(0)
    for (int i=0; i<NX; i++)
      for (int j=0; j<NY; j++)
        for (int k=0; k<NZ; k++)
          PCOUT<<"("<<i<<","<<j<<","<<k<<")  "<<a[k+j*(NZ)+i*NY*(NZ)][0]<<" +i "<<a[k+j*(NZ)+i*NY*(NZ)][1]<<std::endl;
}

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

  step3_gpu(N);

  MPI_Finalize();
  return 0;
} // end main
