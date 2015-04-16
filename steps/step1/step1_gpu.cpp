#include <stdlib.h>
#include <math.h> // M_PI
#include <mpi.h>
#include <cuda_runtime_api.h>
//#include <accfft.h>
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

  data_cpu=(double*)malloc(isize[0]*isize[1]*isize[2]*sizeof(double));
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
  data2_cpu=(double*) malloc(isize[0]*isize[1]*isize[2]*sizeof(double));

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

  PCOUT<<"Relative Error is "<<g_err<<std::endl;


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
  free(data2_cpu);
  cudaFree(data);
  cudaFree(data_hat);
  cudaFree(data2);
  accfft_destroy_plan_gpu(plan);
  accfft_cleanup_gpu();
  MPI_Comm_free(&c_comm);
  return ;

} // end step1_gpu

inline double testcase(double X,double Y,double Z){

  double sigma= 4;
  double pi=M_PI;
  double analytic;
  analytic= std::exp( -sigma * ( (X-pi)*(X-pi) + (Y-pi)*(Y-pi) + (Z-pi)*(Z-pi) ));
  if(analytic!=analytic) analytic=0; /* Do you think the condition will be false always? */
  return analytic;
}
void initialize(double *a, int *n, MPI_Comm c_comm){
  double pi=M_PI;
  int n_tuples=n[2];
  int istart[3], isize[3], osize[3],ostart[3];
  accfft_local_size_dft_r2c_gpu(n,isize,istart,osize,ostart,c_comm);

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
} // end initialize

void check_err(double* a,int*n,MPI_Comm c_comm){
  int nprocs, procid;
  MPI_Comm_rank(c_comm, &procid);
  MPI_Comm_size(c_comm,&nprocs);
  long long int size=n[0];
  size*=n[1]; size*=n[2];
  double pi=4*atan(1.0);

  int n_tuples=n[2];
  int istart[3], isize[3], osize[3],ostart[3];
  accfft_local_size_dft_r2c_gpu(n,isize,istart,osize,ostart,c_comm);

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

  step1_gpu(N);

  MPI_Finalize();
  return 0;
} // end main
