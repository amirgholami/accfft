/**
 * @file
 * CPU functions of AccFFT
 */
/*
 *  Copyright (c) 2014-2015, Amir Gholami, George Biros
 *  All rights reserved.
 *  This file is part of AccFFT library.
 *
 *  AccFFT is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 2 of the License, or
 *  (at your option) any later version.
 *
 *  AccFFT is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with AccFFT.  If not, see <http://www.gnu.org/licenses/>.
 *
*/
#include <mpi.h>
#include <fftw3.h>
#include <omp.h>
#include <iostream>
#include <cmath>
#include <math.h>
#include "transpose.h"
#include <string.h>
#include "accfft.h"
#include "accfft_common.h"
#define VERBOSE 0
#define PCOUT if(procid==0) std::cout
typedef double Complex[2];

/**
 * Initializes the library.
 * @param nthreads The number of OpenMP threads to use for execution of local FFT.
 * @return 0 if successful
 */
int accfft_init(int nthreads){
  int threads_ok=1;
  if (threads_ok) threads_ok = fftw_init_threads();
  if (threads_ok) fftw_plan_with_nthreads(nthreads);

  return (!threads_ok);
}

/**
 * Cleanup all CPU resources
 */
void accfft_cleanup(){
  fftw_cleanup_threads();
  fftw_cleanup();
}

int dfft_get_local_size(int N0, int N1, int N2, int * isize, int * istart,MPI_Comm c_comm ){
  int nprocs, procid;
  MPI_Comm_rank(c_comm, &procid);

  int coords[2],np[2],periods[2];
  MPI_Cart_get(c_comm,2,np,periods,coords);
  isize[2]=N2;
  isize[0]=ceil(N0/(double)np[0]);
  isize[1]=ceil(N1/(double)np[1]);

  istart[0]=isize[0]*(coords[0]);
  istart[1]=isize[1]*(coords[1]);
  istart[2]=0;

  if((N0-isize[0]*coords[0])<isize[0]) {isize[0]=N0-isize[0]*coords[0]; isize[0]*=(int) isize[0]>0; istart[0]=N0-isize[0];}
  if((N1-isize[1]*coords[1])<isize[1]) {isize[1]=N1-isize[1]*coords[1]; isize[1]*=(int) isize[1]>0; istart[1]=N1-isize[1];}


  if(VERBOSE>=2){
    MPI_Barrier(c_comm);
    for(int r=0;r<np[0];r++)
      for(int c=0;c<np[1];c++){
        MPI_Barrier(c_comm);
        if((coords[0]==r) && (coords[1]==c))
          std::cout<<coords[0]<<","<<coords[1]<<" isize[0]= "<<isize[0]<<" isize[1]= "<<isize[1]<<" isize[2]= "<<isize[2]<<" istart[0]= "<<istart[0]<<" istart[1]= "<<istart[1]<<" istart[2]= "<<istart[2]<<std::endl;
        MPI_Barrier(c_comm);
      }
    MPI_Barrier(c_comm);
  }
  int alloc_local=isize[0]*isize[1]*isize[2]*sizeof(double);


  return alloc_local;
}


/**
 * Get the local sizes of the distributed global data for a R2C transform
 * @param n Integer array of size 3, corresponding to the global data size
 * @param isize The size of the data that is locally distributed to the calling process
 * @param istart The starting index of the data that locally resides on the calling process
 * @param osize The output size of the data that locally resides on the calling process,
 * after the R2C transform is finished
 * @param ostart The output starting index of the data that locally resides on the calling process,
 * after the R2C transform is finished
 * @param c_comm Cartesian communicator returned by \ref accfft_create_comm
 * @return
 */
int accfft_local_size_dft_r2c( int * n,int * isize, int * istart, int * osize, int *ostart,MPI_Comm c_comm){

  //1D & 2D Decomp
  int osize_0[3]={0}, ostart_0[3]={0};
  int osize_1[3]={0}, ostart_1[3]={0};
  int osize_2[3]={0}, ostart_2[3]={0};

  int alloc_local;
  int alloc_max=0,n_tuples;
  //inplace==true ? n_tuples=(n[2]/2+1)*2:  n_tuples=n[2]*2;
  n_tuples=(n[2]/2+1)*2;
  alloc_local=dfft_get_local_size(n[0],n[1],n_tuples,osize_0,ostart_0,c_comm);
  alloc_max=std::max(alloc_max, alloc_local);
  alloc_local=dfft_get_local_size(n[0],n_tuples/2,n[1],osize_1,ostart_1,c_comm);
  alloc_max=std::max(alloc_max, alloc_local*2);
  alloc_local=dfft_get_local_size(n[1],n_tuples/2,n[0],osize_2,ostart_2,c_comm);
  alloc_max=std::max(alloc_max, alloc_local*2);

  std::swap(osize_1[1],osize_1[2]);
  std::swap(ostart_1[1],ostart_1[2]);

  std::swap(ostart_2[1],ostart_2[2]);
  std::swap(ostart_2[0],ostart_2[1]);
  std::swap(osize_2[1],osize_2[2]);
  std::swap(osize_2[0],osize_2[1]);

  //isize[0]=osize_0[0];
  //isize[1]=osize_0[1];
  //isize[2]=n[2];//osize_0[2];
  dfft_get_local_size(n[0],n[1],n[2],isize,istart,c_comm);

  osize[0]=osize_2[0];
  osize[1]=osize_2[1];
  osize[2]=osize_2[2];

  ostart[0]=ostart_2[0];
  ostart[1]=ostart_2[1];
  ostart[2]=ostart_2[2];

  return alloc_max;

}


/**
 * Creates a 3D R2C parallel FFT plan.If data_out point to the same location as the input
 * data, then an inplace plan will be created. Otherwise the plan would be outplace.
 * @param n Integer array of size 3, corresponding to the global data size
 * @param data Input data in spatial domain
 * @param data_out Output data in frequency domain
 * @param c_comm Cartesian communicator returned by \ref accfft_create_comm
 * @param flags AccFFT flags, See \ref flags for more details.
 * @return
 */
accfft_plan*  accfft_plan_dft_3d_r2c(int * n, double * data, double * data_out, MPI_Comm c_comm,unsigned flags){
  accfft_plan *plan=new accfft_plan;
  int nprocs, procid;
  MPI_Comm_rank(c_comm, &procid);
  plan->procid=procid;
  MPI_Cart_get(c_comm,2,plan->np,plan->periods,plan->coord);
  plan->c_comm=c_comm;
  int *coord=plan->coord;
  MPI_Comm_split(c_comm,coord[0],coord[1],&plan->row_comm);
  MPI_Comm_split(c_comm,coord[1],coord[0],&plan->col_comm);

  plan->N[0]=n[0];plan->N[1]=n[1];plan->N[2]=n[2];

  plan->data=data;
  plan->data_out=data_out;
  if(data_out==data){
    plan->inplace=true;}
  else{plan->inplace=false;}

  unsigned fftw_flags;
  if(flags==ACCFFT_ESTIMATE)
    fftw_flags=FFTW_ESTIMATE;
  else
    fftw_flags=FFTW_MEASURE;

  if(plan->np[1]==1)
    plan->oneD=true;
  else
    plan->oneD=false;



  int *osize_0 =plan->osize_0, *ostart_0 =plan->ostart_0;
  int *osize_1 =plan->osize_1, *ostart_1 =plan->ostart_1;
  int *osize_2 =plan->osize_2, *ostart_2 =plan->ostart_2;
  int *osize_1i=plan->osize_1i,*ostart_1i=plan->ostart_1i;
  int *osize_2i=plan->osize_2i,*ostart_2i=plan->ostart_2i;

  int alloc_local;
  int alloc_max=0;
  int n_tuples_o,n_tuples_i;
  plan->inplace==true ? n_tuples_i=(n[2]/2+1)*2:  n_tuples_i=n[2];
  n_tuples_o=(n[2]/2+1)*2;

  int isize[3],osize[3],istart[3],ostart[3];
  alloc_max=accfft_local_size_dft_r2c(n,isize,istart,osize,ostart,c_comm);

  dfft_get_local_size(n[0],n[1],n_tuples_o/2,osize_0,ostart_0,c_comm);
  dfft_get_local_size(n[0],n_tuples_o/2,n[1],osize_1,ostart_1,c_comm);
  dfft_get_local_size(n[1],n_tuples_o/2,n[0],osize_2,ostart_2,c_comm);

  std::swap(osize_1[1],osize_1[2]);
  std::swap(ostart_1[1],ostart_1[2]);

  std::swap(ostart_2[1],ostart_2[2]);
  std::swap(ostart_2[0],ostart_2[1]);
  std::swap(osize_2[1],osize_2[2]);
  std::swap(osize_2[0],osize_2[1]);

  for(int i=0;i<3;i++){
    osize_1i[i]=osize_1[i];
    osize_2i[i]=osize_2[i];
    ostart_1i[i]=ostart_1[i];
    ostart_2i[i]=ostart_2[i];
  }


  // FFT Plans
  {
    plan->fplan_0= fftw_plan_many_dft_r2c(1, &n[2],osize_0[0]*osize_0[1], //int rank, const int *n, int howmany
        data, NULL,         //double *in, const int *inembed,
        1, n_tuples_i,      //int istride, int idist,
        (fftw_complex*)data_out, NULL,  //fftw_complex *out, const int *onembed,
        1, n_tuples_o/2,        // int ostride, int odist,
        fftw_flags);
    if(plan->fplan_0==NULL) std::cout<<"!!! fplan_0 not created in r2c plan!!!"<<std::endl;

    plan->iplan_0= fftw_plan_many_dft_c2r(1, &n[2],osize_0[0]*osize_0[1], //int rank, const int *n, int howmany
        (fftw_complex*)data_out, NULL,          //double *in, const int *inembed,
        1, n_tuples_o/2,      //int istride, int idist,
        data, NULL, //fftw_complex *out, const int *onembed,
        1, n_tuples_i,        // int ostride, int odist,
        fftw_flags);
    if(plan->iplan_0==NULL) std::cout<<"!!! iplan_0 not created in r2c plan!!!"<<std::endl;

    // ----

    fftw_iodim dims, howmany_dims[2];
    dims.n=osize_1[1];
    dims.is=osize_1[2];
    dims.os=osize_1[2];

    howmany_dims[0].n=osize_1[2];
    howmany_dims[0].is=1;
    howmany_dims[0].os=1;

    howmany_dims[1].n=osize_1[0];
    howmany_dims[1].is=osize_1[1]*osize_1[2];
    howmany_dims[1].os=osize_1[1]*osize_1[2];

    plan->fplan_1=fftw_plan_guru_dft(
        1, &dims,
        2,howmany_dims,
        (fftw_complex*)data_out, (fftw_complex*)data_out,
        -1,fftw_flags);
    if(plan->fplan_1==NULL) std::cout<<"!!! fplan1 not created in r2c plan!!!"<<std::endl;
    plan->iplan_1=fftw_plan_guru_dft(
        1, &dims,
        2,howmany_dims,
        (fftw_complex*)data_out, (fftw_complex*)data_out,
        1,fftw_flags);
    if(plan->iplan_1==NULL) std::cout<<"!!! iplan1 not created in r2c plan !!!"<<std::endl;

    // ----

    plan->fplan_2= fftw_plan_many_dft(1, &n[0],osize_2[2]*osize_2[1], //int rank, const int *n, int howmany
        (fftw_complex*)data_out, NULL,          //double *in, const int *inembed,
        osize_2[2]*osize_2[1],1,      //int istride, int idist,
        (fftw_complex*)data_out, NULL,  //fftw_complex *out, const int *onembed,
        osize_2[2]*osize_2[1],1,        // int ostride, int odist,
        FFTW_FORWARD,fftw_flags);
    if(plan->fplan_2==NULL) std::cout<<"!!! fplan2 not created in r2c plan !!!"<<std::endl;

    plan->iplan_2= fftw_plan_many_dft(1, &n[0],osize_2[2]*osize_2[1], //int rank, const int *n, int howmany
        (fftw_complex*)data_out, NULL,          //double *in, const int *inembed,
        osize_2[2]*osize_2[1],1,      //int istride, int idist,
        (fftw_complex*)data_out, NULL,  //fftw_complex *out, const int *onembed,
        osize_2[2]*osize_2[1],1,        // int ostride, int odist,
        FFTW_BACKWARD,fftw_flags);
    if(plan->iplan_2==NULL) std::cout<<"!!! iplan2 not created in r2c plan !!!"<<std::endl;
  }

  // 1D Decomposition
  if (plan->oneD){
    plan->Mem_mgr   = new Mem_Mgr<double>(n[0],n[1],n_tuples_o,c_comm);
    plan->T_plan_2  = new T_Plan<double>(n[0],n[1],n_tuples_o, plan->Mem_mgr, c_comm);
    plan->T_plan_2i = new T_Plan<double>(n[1],n[0],n_tuples_o,plan->Mem_mgr, c_comm);
    plan->T_plan_1  =NULL;
    plan->T_plan_1i =NULL;

    plan->T_plan_2->alloc_local=alloc_max;
    plan->T_plan_2i->alloc_local=alloc_max;

    if(flags==ACCFFT_MEASURE){
      plan->T_plan_2->which_fast_method(plan->T_plan_2,data_out);
    }
    else{
      plan->T_plan_2->method=2;
      plan->T_plan_2->kway=2;
    }
    plan->T_plan_2i->method=plan->T_plan_2->method;
    plan->T_plan_2i->kway=plan->T_plan_2->kway;
    plan->data=data;


    plan->data=data;

  } // end 1D Decomp r2c

  // 2D Decomposition
  if (!plan->oneD){
    plan->Mem_mgr   = new Mem_Mgr<double>(n[1],n_tuples_o/2,2,plan->row_comm,osize_0[0],alloc_max);
    plan->T_plan_1  = new T_Plan<double>(n[1],n_tuples_o/2,2, plan->Mem_mgr, plan->row_comm,osize_0[0]);
    plan->T_plan_2  = new T_Plan<double>(n[0],n[1],osize_2[2]*2, plan->Mem_mgr, plan->col_comm);
    plan->T_plan_2i = new T_Plan<double>(n[1],n[0],osize_2i[2]*2, plan->Mem_mgr, plan->col_comm);
    plan->T_plan_1i = new T_Plan<double>(n_tuples_o/2,n[1],2, plan->Mem_mgr, plan->row_comm,osize_1i[0]);

    plan->T_plan_1->alloc_local=plan->alloc_max;
    plan->T_plan_2->alloc_local=plan->alloc_max;
    plan->T_plan_2i->alloc_local=plan->alloc_max;
    plan->T_plan_1i->alloc_local=plan->alloc_max;


    if(flags==ACCFFT_MEASURE){
      if(coord[0]==0){
        plan->T_plan_1->which_fast_method(plan->T_plan_1,data_out,osize_0[0]);
      }
    }
    else{
      if(coord[0]==0){
        plan->T_plan_1->method=2;
        plan->T_plan_1->kway=2;
      }
    }

    MPI_Bcast(&plan->T_plan_1->method,1, MPI_INT,0, c_comm );
    MPI_Bcast(&plan->T_plan_1->kway,1, MPI_INT,0, c_comm );

    plan->T_plan_1->method =plan->T_plan_1->method;
    plan->T_plan_2->method =plan->T_plan_1->method;
    plan->T_plan_2i->method=plan->T_plan_1->method;
    plan->T_plan_1i->method=plan->T_plan_1->method;

    plan->T_plan_1->kway =plan->T_plan_1->kway;
    plan->T_plan_2->kway =plan->T_plan_1->kway;
    plan->T_plan_2i->kway=plan->T_plan_1->kway;
    plan->T_plan_1i->kway=plan->T_plan_1->kway;

    plan->data=data;
  } // end 2D r2c

  plan->r2c_plan_baked=true;

  return plan;

} // end accfft_plan_dft_3d_r2c

void accfft_execute(accfft_plan* plan, int direction,double * data,double * data_out, double * timer,std::bitset<3> XYZ){

  if(data==NULL)
    data=plan->data;
  if(data_out==NULL)
    data_out=plan->data_out;
  int * coords=plan->coord;
  int procid=plan->procid;
  double fft_time=0;
  double * timings;
  if(timer==NULL){
    timings=new double[5];
    memset(timings,0,sizeof(double)*5);
  }
  else{
    timings=timer;
  }


  // 1D Decomposition
  int *osize_0 =plan->osize_0, *ostart_0 =plan->ostart_0;
  int *osize_1 =plan->osize_1, *ostart_1 =plan->ostart_1;
  int *osize_2 =plan->osize_2, *ostart_2 =plan->ostart_2;
  int *osize_1i=plan->osize_1i,*ostart_1i=plan->ostart_1i;
  int *osize_2i=plan->osize_2i,*ostart_2i=plan->ostart_2i;

  if(direction==-1){
    /**************************************************************/
    /*******************  N0/P0 x N1/P1 x N2 **********************/
    /**************************************************************/
    // FFT in Z direction
    fft_time-=MPI_Wtime();
    if(XYZ[0])
      fftw_execute_dft_r2c(plan->fplan_0,(double*)data,(fftw_complex*)data_out);
    fft_time+=MPI_Wtime();

    // Perform N0/P0 transpose
    if(!plan->oneD){
      plan->T_plan_1->execute(plan->T_plan_1,data_out,timings,2,osize_0[0],coords[0]);
    }
    /**************************************************************/
    /*******************  N0/P0 x N1 x N2/P1 **********************/
    /**************************************************************/
    fft_time-=MPI_Wtime();
    if(XYZ[1])
      fftw_execute_dft(plan->fplan_1,(fftw_complex*)data_out,(fftw_complex*)data_out);
    fft_time+=MPI_Wtime();


    if(plan->oneD){
      plan->T_plan_2->execute(plan->T_plan_2,data_out,timings,2);
    }
    else{
      plan->T_plan_2->execute(plan->T_plan_2,data_out,timings,2,1,coords[1]);
    }
    /**************************************************************/
    /*******************  N0 x N1/P0 x N2/P1 **********************/
    /**************************************************************/
    fft_time-=MPI_Wtime();
    if(XYZ[2])
      fftw_execute_dft(plan->fplan_2,(fftw_complex*)data_out,(fftw_complex*)data_out);
    fft_time+=MPI_Wtime();
  }
  else if (direction==1){
    fft_time-=MPI_Wtime();
    if(XYZ[2])
      fftw_execute_dft(plan->iplan_2,(fftw_complex*)data,(fftw_complex*)data);
    fft_time+=MPI_Wtime();


    if(plan->oneD){
      plan->T_plan_2i->execute(plan->T_plan_2i,data,timings,1);
    }
    else{
      plan->T_plan_2i->execute(plan->T_plan_2i,data,timings,1,1,coords[1]);
    }

    MPI_Barrier(plan->c_comm);
    /**************************************************************/
    /*******************  N0/P0 x N1 x N2/P1 **********************/
    /**************************************************************/
    fft_time-=MPI_Wtime();
    if(XYZ[1])
      fftw_execute_dft(plan->iplan_1,(fftw_complex*)data,(fftw_complex*)data);
    fft_time+=MPI_Wtime();


    if(!plan->oneD){
      plan->T_plan_1i->execute(plan->T_plan_1i,data,timings,1,osize_1i[0],coords[0]);
    }
    MPI_Barrier(plan->c_comm);
    /**************************************************************/
    /*******************  N0/P0 x N1/P1 x N2 **********************/
    /**************************************************************/
    // IFFT in Z direction
    fft_time-=MPI_Wtime();
    if(XYZ[0])
      fftw_execute_dft_c2r(plan->iplan_0,(fftw_complex*)data,(double*)data_out);
    fft_time+=MPI_Wtime();
    MPI_Barrier(plan->c_comm);

  }

  timings[4]=fft_time;
  if(timer==NULL){
    delete [] timings;
  }
  MPI_Barrier(plan->c_comm);

  return;
}
/**
 * Get the local sizes of the distributed global data for a C2C transform
 * @param n Integer array of size 3, corresponding to the global data size
 * @param isize The size of the data that is locally distributed to the calling process
 * @param istart The starting index of the data that locally resides on the calling process
 * @param osize The output size of the data that locally resides on the calling process,
 * after the C2C transform is finished
 * @param ostart The output starting index of the data that locally resides on the calling process,
 * after the R2C transform is finished
 * @param c_comm Cartesian communicator returned by \ref accfft_create_comm
 * @return
 */
int accfft_local_size_dft_c2c( int * n,int * isize, int * istart, int * osize, int *ostart,MPI_Comm c_comm){

  int osize_0[3]={0}, ostart_0[3]={0};
  int osize_1[3]={0}, ostart_1[3]={0};
  int osize_2[3]={0}, ostart_2[3]={0};
  int osize_1i[3]={0}, ostart_1i[3]={0};
  int osize_2i[3]={0}, ostart_2i[3]={0};

  int alloc_local;
  int alloc_max=0,n_tuples=n[2]*2;
  alloc_local=dfft_get_local_size(n[0],n[1],n[2],osize_0,ostart_0,c_comm);
  alloc_max=std::max(alloc_max, alloc_local);
  alloc_local=dfft_get_local_size(n[0],n[2],n[1],osize_1,ostart_1,c_comm);
  alloc_max=std::max(alloc_max, alloc_local);
  alloc_local=dfft_get_local_size(n[1],n[2],n[0],osize_2,ostart_2,c_comm);
  alloc_max=std::max(alloc_max, alloc_local);
  alloc_max*=2; // because of c2c

  std::swap(osize_1[1],osize_1[2]);
  std::swap(ostart_1[1],ostart_1[2]);

  std::swap(ostart_2[1],ostart_2[2]);
  std::swap(ostart_2[0],ostart_2[1]);
  std::swap(osize_2[1],osize_2[2]);
  std::swap(osize_2[0],osize_2[1]);

  for(int i=0;i<3;i++){
    osize_1i[i]=osize_1[i];
    osize_2i[i]=osize_2[i];
    ostart_1i[i]=ostart_1[i];
    ostart_2i[i]=ostart_2[i];
  }

  //isize[0]=osize_0[0];
  //isize[1]=osize_0[1];
  //isize[2]=n[2];//osize_0[2];
  dfft_get_local_size(n[0],n[1],n[2],isize,istart,c_comm);

  osize[0]=osize_2[0];
  osize[1]=osize_2[1];
  osize[2]=osize_2[2];

  ostart[0]=ostart_2[0];
  ostart[1]=ostart_2[1];
  ostart[2]=ostart_2[2];

  return alloc_max;

}

/**
 * Creates a 3D C2C parallel FFT plan. If data_out point to the same location as the input
 * data, then an inplace plan will be created. Otherwise the plan would be outplace.
 * @param n Integer array of size 3, corresponding to the global data size
 * @param data Input data in spatial domain
 * @param data_out Output data in frequency domain
 * @param c_comm Cartesian communicator returned by \ref accfft_create_comm
 * @param flags AccFFT flags, See \ref flags for more details.
 * @return
 */
accfft_plan*  accfft_plan_dft_3d_c2c(int * n, Complex * data, Complex * data_out, MPI_Comm c_comm,unsigned flags){

  accfft_plan *plan=new accfft_plan;
  int nprocs, procid;
  MPI_Comm_rank(c_comm, &procid);
  plan->procid=procid;
  MPI_Cart_get(c_comm,2,plan->np,plan->periods,plan->coord);
  plan->c_comm=c_comm;
  int *coord=plan->coord;
  MPI_Comm_split(c_comm,coord[0],coord[1],&plan->row_comm);
  MPI_Comm_split(c_comm,coord[1],coord[0],&plan->col_comm);
  plan->N[0]=n[0];plan->N[1]=n[1];plan->N[2]=n[2];

  plan->data_c=data;
  plan->data_out_c=data_out;
  if(data_out==data){
    plan->inplace=true;}
  else{plan->inplace=false;}

  unsigned fftw_flags;
  if(flags==ACCFFT_ESTIMATE)
    fftw_flags=FFTW_ESTIMATE;
  else
    fftw_flags=FFTW_MEASURE;


  if(plan->np[1]==1)
    plan->oneD=true;
  else
    plan->oneD=false;



  // FFT Plans
  int *osize_0 =plan->osize_0, *ostart_0 =plan->ostart_0;
  int *osize_1 =plan->osize_1, *ostart_1 =plan->ostart_1;
  int *osize_2 =plan->osize_2, *ostart_2 =plan->ostart_2;
  int *osize_1i=plan->osize_1i,*ostart_1i=plan->ostart_1i;
  int *osize_2i=plan->osize_2i,*ostart_2i=plan->ostart_2i;

  int alloc_local;
  int alloc_max=0,n_tuples=(n[2]/2+1)*2;

  int isize[3],osize[3],istart[3],ostart[3];
  alloc_max=accfft_local_size_dft_c2c(n,isize,istart,osize,ostart,c_comm);
  plan->alloc_max=alloc_max;

  dfft_get_local_size(n[0],n[1],n[2],osize_0,ostart_0,c_comm);
  dfft_get_local_size(n[0],n[2],n[1],osize_1,ostart_1,c_comm);
  dfft_get_local_size(n[1],n[2],n[0],osize_2,ostart_2,c_comm);

  std::swap(osize_1[1],osize_1[2]);
  std::swap(ostart_1[1],ostart_1[2]);

  std::swap(ostart_2[1],ostart_2[2]);
  std::swap(ostart_2[0],ostart_2[1]);
  std::swap(osize_2[1],osize_2[2]);
  std::swap(osize_2[0],osize_2[1]);

  for(int i=0;i<3;i++){
    osize_1i[i]=osize_1[i];
    osize_2i[i]=osize_2[i];
    ostart_1i[i]=ostart_1[i];
    ostart_2i[i]=ostart_2[i];
  }

  {
    // fplan_0
    plan->fplan_0= fftw_plan_many_dft(1, &n[2],osize_0[0]*osize_0[1], //int rank, const int *n, int howmany
        data, NULL,         //double *in, const int *inembed,
        1, n[2],      //int istride, int idist,
        data_out, NULL, //fftw_complex *out, const int *onembed,
        1, n[2],        // int ostride, int odist,
        FFTW_FORWARD,fftw_flags);
    if(plan->fplan_0==NULL) std::cout<<"!!! fplan0 not created in c2c plan!!!"<<std::endl;
    plan->iplan_0= fftw_plan_many_dft(1, &n[2],osize_0[0]*osize_0[1], //int rank, const int *n, int howmany
        data_out, NULL,         //double *in, const int *inembed,
        1, n[2],      //int istride, int idist,
        data, NULL, //fftw_complex *out, const int *onembed,
        1, n[2],        // int ostride, int odist,
        FFTW_BACKWARD,fftw_flags);
    if(plan->iplan_0==NULL) std::cout<<"!!! iplan0 not created in c2c plan!!!"<<std::endl;


    // fplan_1
    fftw_iodim dims, howmany_dims[2];
    dims.n=osize_1[1];
    dims.is=osize_1[2];
    dims.os=osize_1[2];

    howmany_dims[0].n=osize_1[2];
    howmany_dims[0].is=1;
    howmany_dims[0].os=1;

    howmany_dims[1].n=osize_1[0];
    howmany_dims[1].is=osize_1[1]*osize_1[2];
    howmany_dims[1].os=osize_1[1]*osize_1[2];

    plan->fplan_1=fftw_plan_guru_dft(
        1, &dims,
        2,howmany_dims,
        data_out, data_out,
        -1,fftw_flags);
    if(plan->fplan_1==NULL) std::cout<<"!!! fplan1 not created in c2c plan !!!"<<std::endl;
    plan->iplan_1=fftw_plan_guru_dft(
        1, &dims,
        2,howmany_dims,
        (fftw_complex*)data_out, (fftw_complex*)data_out,
        1,fftw_flags);
    if(plan->iplan_1==NULL) std::cout<<"!!! iplan1 not created in c2c plan !!!"<<std::endl;


    // fplan_2
    plan->fplan_2= fftw_plan_many_dft(1, &n[0],osize_2[1]*osize_2[2], //int rank, const int *n, int howmany
        data_out, NULL,         //double *in, const int *inembed,
        osize_2[1]*osize_2[2],1 ,     //int istride, int idist,
        data_out, NULL, //fftw_complex *out, const int *onembed,
        osize_2[1]*osize_2[2],1 ,       // int ostride, int odist,
        FFTW_FORWARD,fftw_flags);
    if(plan->fplan_2==NULL) std::cout<<"!!! fplan2 not created in c2c plan!!!"<<std::endl;
    plan->iplan_2= fftw_plan_many_dft(1, &n[0],osize_2[1]*osize_2[2], //int rank, const int *n, int howmany
        data_out, NULL,         //double *in, const int *inembed,
        osize_2[1]*osize_2[2],1 ,     //int istride, int idist,
        data_out, NULL, //fftw_complex *out, const int *onembed,
        osize_2[1]*osize_2[2],1 ,       // int ostride, int odist,
        FFTW_BACKWARD,fftw_flags);
    if(plan->iplan_2==NULL) std::cout<<"!!! fplan2 not created in c2c plan!!!"<<std::endl;

  }

  // 1D Decomposition
  if (plan->oneD){
    int NX=n[0],NY=n[1],NZ=n[2];
    plan->Mem_mgr   = new Mem_Mgr<double>(NX,NY,(NZ)*2,c_comm);
    plan->T_plan_2  = new T_Plan<double>(NX,NY,(NZ)*2, plan->Mem_mgr,c_comm);
    plan->T_plan_2i = new T_Plan<double>(NY,NX,NZ*2, plan->Mem_mgr,c_comm);

    plan->T_plan_1=NULL;
    plan->T_plan_1i=NULL;

    plan->alloc_max=alloc_max;
    plan->T_plan_2->alloc_local=alloc_max;
    plan->T_plan_2i->alloc_local=alloc_max;


    if(flags==ACCFFT_MEASURE){
      plan->T_plan_2->which_fast_method(plan->T_plan_2,(double*)data_out);
    }
    else{
      plan->T_plan_2->method=2;
      plan->T_plan_2->kway=2;
    }
    plan->T_plan_2i->method=plan->T_plan_2->method;
    plan->T_plan_2i->kway=plan->T_plan_2->kway;

  } // end 1D decomp c2c

  // 2D Decomposition
  if (!plan->oneD){
    // the reaseon for n_tuples/2 is to avoid splitting of imag and real parts of complex numbers

    plan->Mem_mgr   =new Mem_Mgr<double>(n[1],n[2],2,plan->row_comm,osize_0[0],alloc_max);
    plan->T_plan_1  =new T_Plan<double>(n[1],n[2],2, plan->Mem_mgr, plan->row_comm,osize_0[0]);
    plan->T_plan_2  =new T_Plan<double>(n[0],n[1],2*osize_2[2], plan->Mem_mgr, plan->col_comm);
    plan->T_plan_2i =new T_Plan<double>(n[1],n[0],2*osize_2i[2], plan->Mem_mgr, plan->col_comm);
    plan->T_plan_1i =new T_Plan<double>(n[2],n[1],2, plan->Mem_mgr, plan->row_comm,osize_1i[0]);

    plan->T_plan_1->alloc_local=plan->alloc_max;
    plan->T_plan_2->alloc_local=plan->alloc_max;
    plan->T_plan_2i->alloc_local=plan->alloc_max;
    plan->T_plan_1i->alloc_local=plan->alloc_max;


    if(flags==ACCFFT_MEASURE){
      if(coord[0]==0){
        plan->T_plan_1->which_fast_method(plan->T_plan_1,(double*)data_out,osize_0[0]);
      }
    }
    else{
      if(coord[0]==0){
        plan->T_plan_1->method=2;
        plan->T_plan_1->kway=2;
      }
    }

    MPI_Bcast(&plan->T_plan_1->method,1, MPI_INT,0, c_comm );
    MPI_Bcast(&plan->T_plan_1->kway,1, MPI_INT,0, c_comm );

    plan->T_plan_1->method =plan->T_plan_1->method;
    plan->T_plan_2->method =plan->T_plan_1->method;
    plan->T_plan_2i->method=plan->T_plan_1->method;
    plan->T_plan_1i->method=plan->T_plan_1->method;
    plan->T_plan_1->kway =plan->T_plan_1->kway;
    plan->T_plan_2->kway =plan->T_plan_1->kway;
    plan->T_plan_2i->kway=plan->T_plan_1->kway;
    plan->T_plan_1i->kway=plan->T_plan_1->kway;
  } // end 2D Decomp c2c

  plan->c2c_plan_baked=true;
  return plan;

} // end accfft_plan_dft_3d_c2c

/**
 * Execute R2C plan. This function is blocking and only returns after the transform is completed.
 * @note For inplace transforms, data_out should point to the same memory address as data, AND
 * the plan must have been created as inplace.
 * @param plan FFT plan created by \ref accfft_plan_dft_3d_r2c.
 * @param data Input data in spatial domain.
 * @param data_out Output data in frequency domain.
 * @param timer See \ref timer for more details.
 * @param XYZ a bit set field that determines which directions FFT should be executed
 */
void accfft_execute_r2c(accfft_plan* plan, double * data,Complex * data_out, double * timer,std::bitset<3> XYZ){
  if(plan->r2c_plan_baked){
    accfft_execute(plan,-1,data,(double*)data_out,timer);
  }
  else{
    if(plan->procid==0) std::cout<<"Error. r2c plan has not been made correctly. Please first create the plan before calling execute functions."<<std::endl;
  }
  return;
}


/**
 * Execute C2R plan. This function is blocking and only returns after the transform is completed.
 * @note For inplace transform, data_out should point to the same memory address as data, AND
 * the plan must have been created as inplace.
 * @param plan FFT plan created by \ref accfft_plan_dft_3d_r2c.
 * @param data Input data in frequency domain.
 * @param data_out Output data in frequency domain.
 * @param timer See \ref timer for more details.
 * @param XYZ a bit set field that determines which directions FFT should be executed
 */
void accfft_execute_c2r(accfft_plan* plan, Complex * data,double * data_out, double * timer,std::bitset<3> XYZ){
  if(plan->r2c_plan_baked){
    accfft_execute(plan,1,(double*)data,data_out,timer);
  }
  else{
    if(plan->procid==0) std::cout<<"Error. r2c plan has not been made correctly. Please first create the plan before calling execute functions."<<std::endl;
  }
  return;
}


/**
 * Execute C2C plan. This function is blocking and only returns after the transform is completed.
 * @note For inplace transforms, data_out should point to the same memory address as data, AND
 * the plan must have been created as inplace.
 * @param plan FFT plan created by \ref accfft_plan_dft_3d_r2c.
 * @param data Input data in frequency domain.
 * @param data_out Output data in frequency domain.
 * @param timer See \ref timer for more details.
 * @param XYZ a bit set field that determines which directions FFT should be executed
 */
void accfft_execute_c2c(accfft_plan* plan, int direction,Complex * data, Complex * data_out, double * timer,std::bitset<3> XYZ){
  if(!plan->c2c_plan_baked){
    if(plan->procid==0) std::cout<<"Error. r2c plan has not been made correctly. Please first create the plan before calling execute functions."<<std::endl;
    return;
  }

  if(data==NULL)
    data=plan->data_c;
  if(data_out==NULL)
    data_out=plan->data_out_c;
  int * coords=plan->coord;
  int procid=plan->procid;
  double fft_time=0;
  double * timings;
  if(timer==NULL){
    timings=new double[5];
    memset(timings,0,sizeof(double)*5);
  }
  else{
    timings=timer;
  }



  int *osize_0 =plan->osize_0, *ostart_0 =plan->ostart_0;
  int *osize_1 =plan->osize_1, *ostart_1 =plan->ostart_1;
  int *osize_2 =plan->osize_2, *ostart_2 =plan->ostart_2;
  int *osize_1i=plan->osize_1i,*ostart_1i=plan->ostart_1i;
  int *osize_2i=plan->osize_2i,*ostart_2i=plan->ostart_2i;

  if(direction==-1){
    /**************************************************************/
    /*******************  N0/P0 x N1/P1 x N2 **********************/
    /**************************************************************/
    // FFT in Z direction
    fft_time-=MPI_Wtime();
    if(XYZ[0])
      fftw_execute_dft(plan->fplan_0,data,data_out);
    fft_time+=MPI_Wtime();



    if(!plan->oneD){
      plan->T_plan_1->execute(plan->T_plan_1,(double*)data_out,timings,2,osize_0[0],coords[0]);
    }
    /**************************************************************/
    /*******************  N0/P0 x N1 x N2/P1 **********************/
    /**************************************************************/
    fft_time-=MPI_Wtime();
    if(XYZ[1])
      fftw_execute_dft(plan->fplan_1,(fftw_complex*)data_out,(fftw_complex*)data_out);
    fft_time+=MPI_Wtime();


    if(!plan->oneD){
      plan->T_plan_2->execute(plan->T_plan_2,(double*)data_out,timings,2,1,coords[1]);
    }
    else{
      plan->T_plan_2->execute(plan->T_plan_2,(double*)data_out,timings,2);
    }
    /**************************************************************/
    /*******************  N0 x N1/P0 x N2/P1 **********************/
    /**************************************************************/
    fft_time-=MPI_Wtime();
    if(XYZ[2])
      fftw_execute_dft(plan->fplan_2,(fftw_complex*)data_out,(fftw_complex*)data_out);
    fft_time+=MPI_Wtime();
  }
  else if (direction==1){
    fft_time-=MPI_Wtime();
    if(XYZ[2])
      fftw_execute_dft(plan->iplan_2,(fftw_complex*)data,(fftw_complex*)data);
    fft_time+=MPI_Wtime();


    if(!plan->oneD){
      plan->T_plan_2i->execute(plan->T_plan_2i,(double*)data,timings,1,1,coords[1]);
    }
    else{
      plan->T_plan_2i->execute(plan->T_plan_2i,(double*)data,timings,1);
    }
    /**************************************************************/
    /*******************  N0/P0 x N1 x N2/P1 **********************/
    /**************************************************************/
    fft_time-=MPI_Wtime();
    if(XYZ[1])
      fftw_execute_dft(plan->iplan_1,(fftw_complex*)data,(fftw_complex*)data);
    fft_time+=MPI_Wtime();

    if(!plan->oneD){
      plan->T_plan_1i->execute(plan->T_plan_1i,(double*)data,timings,1,osize_1i[0],coords[0]);
    }
    MPI_Barrier(plan->c_comm);
    /**************************************************************/
    /*******************  N0/P0 x N1/P1 x N2 **********************/
    /**************************************************************/

    // IFFT in Z direction
    fft_time-=MPI_Wtime();
    if(XYZ[0])
      fftw_execute_dft(plan->iplan_0,data,data_out);
    fft_time+=MPI_Wtime();

  }

  timings[4]=fft_time;
  if(timer==NULL){
    delete [] timings;
  }
  MPI_Barrier(plan->c_comm);

  return;
}

/**
 * Destroy AccFFT CPU plan.
 * @param plan Input plan to be destroyed.
 */
void accfft_destroy_plan(accfft_plan * plan){

  if(plan!=NULL){
    if(plan->T_plan_1!=NULL)  delete(plan->T_plan_1);
    if(plan->T_plan_1i!=NULL) delete(plan->T_plan_1i);
    if(plan->T_plan_2!=NULL)  delete(plan->T_plan_2);
    if(plan->T_plan_2i!=NULL) delete(plan->T_plan_2i);
    if(plan->Mem_mgr!=NULL)   delete(plan->Mem_mgr);
    if(plan->fplan_0!=NULL) fftw_destroy_plan(plan->fplan_0);
    if(plan->fplan_1!=NULL) fftw_destroy_plan(plan->fplan_1);
    if(plan->fplan_2!=NULL) fftw_destroy_plan(plan->fplan_2);
    if(plan->iplan_0!=NULL) fftw_destroy_plan(plan->iplan_0);
    if(plan->iplan_1!=NULL) fftw_destroy_plan(plan->iplan_1);
    if(plan->iplan_2!=NULL) fftw_destroy_plan(plan->iplan_2);

    MPI_Comm_free(&plan->row_comm);
    MPI_Comm_free(&plan->col_comm);
    delete plan;
  }
}
