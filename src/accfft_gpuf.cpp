/**
 * @file
 * Single precision GPU functions of AccFFT
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

#include "accfft_gpuf.h"
#include <mpi.h>
#include <omp.h>
#include <iostream>
#include <cmath>
#include <math.h>
#include "transpose_cuda.h"
#include <cuda_runtime_api.h>
#include <string.h>
#include <cuda.h>
#include <cufft.h>
#include "accfft_common.h"
#define VERBOSE 0
#define PCOUT if(procid==0) std::cout

/**
 * Cleanup all CPU resources
 */
void accfft_cleanup_gpuf(){
  // empty for now
}

int dfft_get_local_size_gpuf(int N0, int N1, int N2, int * isize, int * istart,MPI_Comm c_comm ){
  int  procid;
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
  int alloc_local=isize[0]*isize[1]*isize[2]*sizeof(float);


  return alloc_local;
}

/**
 * Get the local sizes of the distributed global data for a GPU R2C transform
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
int accfft_local_size_dft_r2c_gpuf( int * n,int * isize, int * istart, int * osize, int *ostart,MPI_Comm c_comm, bool inplace){

  //1D & 2D Decomp
  int osize_0[3]={0}, ostart_0[3]={0};
  int osize_1[3]={0}, ostart_1[3]={0};
  int osize_2[3]={0}, ostart_2[3]={0};

  int alloc_local;
  int alloc_max=0,n_tuples;
  //inplace==true ? n_tuples=(n[2]/2+1)*2:  n_tuples=n[2];
  n_tuples=(n[2]/2+1)*2;//SNAFU
  alloc_local=dfft_get_local_size_gpuf(n[0],n[1],n_tuples,osize_0,ostart_0,c_comm);
  alloc_max=std::max(alloc_max, alloc_local);
  alloc_local=dfft_get_local_size_gpuf(n[0],n_tuples/2,n[1],osize_1,ostart_1,c_comm);
  alloc_max=std::max(alloc_max, alloc_local*2);
  alloc_local=dfft_get_local_size_gpuf(n[1],n_tuples/2,n[0],osize_2,ostart_2,c_comm);
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
  dfft_get_local_size_gpuf(n[0],n[1],n[2],isize,istart,c_comm);

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
accfft_plan_gpuf*  accfft_plan_dft_3d_r2c_gpuf(int * n, float * data_d,float * data_out_d, MPI_Comm c_comm,unsigned flags){
  accfft_plan_gpuf *plan=new accfft_plan_gpuf;
  int procid;
  MPI_Comm_rank(c_comm, &procid);
  plan->procid=procid;
  MPI_Cart_get(c_comm,2,plan->np,plan->periods,plan->coord);
  plan->c_comm=c_comm;
  int *coord=plan->coord;
  MPI_Comm_split(c_comm,coord[0],coord[1],&plan->row_comm);
  MPI_Comm_split(c_comm,coord[1],coord[0],&plan->col_comm);
  plan->N[0]=n[0];plan->N[1]=n[1];plan->N[2]=n[2];
  plan->data=data_d;
  plan->data_out=data_out_d;

  if(plan->np[1]==1)
    plan->oneD=true;
  else
    plan->oneD=false;


  if(data_out_d==data_d){
    plan->inplace=true;}
  else{plan->inplace=false;}

  int *osize_0 =plan->osize_0, *ostart_0 =plan->ostart_0;
  int *osize_1 =plan->osize_1, *ostart_1 =plan->ostart_1;
  int *osize_2 =plan->osize_2, *ostart_2 =plan->ostart_2;
  int *osize_1i=plan->osize_1i,*ostart_1i=plan->ostart_1i;
  int *osize_2i=plan->osize_2i,*ostart_2i=plan->ostart_2i;

  int alloc_max=0;
  int n_tuples_i, n_tuples_o;
  //plan->inplace==true ? n_tuples=(n[2]/2+1)*2: n_tuples=n[2]*2;
  plan->inplace==true ? n_tuples_i=(n[2]/2+1)*2:  n_tuples_i=n[2];
  n_tuples_o=(n[2]/2+1)*2;

  int isize[3],osize[3],istart[3],ostart[3];
  alloc_max=accfft_local_size_dft_r2c_gpuf(n,isize,istart,osize,ostart,c_comm,plan->inplace);
  plan->alloc_max=alloc_max;

  dfft_get_local_size_gpuf(n[0],n[1],n_tuples_o,osize_0,ostart_0,c_comm);
  dfft_get_local_size_gpuf(n[0],n_tuples_o/2,n[1],osize_1,ostart_1,c_comm);
  dfft_get_local_size_gpuf(n[1],n_tuples_o/2,n[0],osize_2,ostart_2,c_comm);

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

  // fplan_0
  int NX=n[0], NY=n[1], NZ=n[2];
  cufftResult_t cufft_error;
  {
    int f_inembed[1]={n_tuples_i};
    int f_onembed[1]={n_tuples_o/2};
    int idist=(n_tuples_i);
    int odist=n_tuples_o/2;
    int istride=1;
    int ostride=1;
    int batch=osize_0[0]*osize_0[1];//NX;

    if(batch!=0)
    {
      cufft_error=cufftPlanMany(&plan->fplan_0, 1, &n[2],
          f_inembed, istride, idist, // *inembed, istride, idist
          f_onembed, ostride, odist, // *onembed, ostride, odist
          CUFFT_R2C, batch);
      if(cufft_error!= CUFFT_SUCCESS)
      {
        fprintf(stderr, "CUFFT error: fplan_0 creation failed %d \n",cufft_error); return NULL;
      }
      //cufftSetCompatibilityMode(fplan,CUFFT_COMPATIBILITY_FFTW_PADDING); if (cudaGetLastError() != cudaSuccess){fprintf(stderr, "Cuda error:Failed at fplan cuda compatibility\n"); return;}
    }
    if(batch!=0)
    {
      cufft_error=cufftPlanMany(&plan->iplan_0, 1, &n[2],
          f_onembed, ostride, odist, // *onembed, ostride, odist
          f_inembed, istride, idist, // *inembed, istride, idist
          CUFFT_C2R, batch);
      if(cufft_error!= CUFFT_SUCCESS)
      {
        fprintf(stderr, "CUFFT error: iplan_0 creation failed %d \n",cufft_error); return NULL;
      }
      //cufftSetCompatibilityMode(fplan,CUFFT_COMPATIBILITY_FFTW_PADDING); if (cudaGetLastError() != cudaSuccess){fprintf(stderr, "Cuda error:Failed at fplan cuda compatibility\n"); return;}
    }
  }
  // fplan_1
  {
    int f_inembed[1]={NY};
    int f_onembed[1]={NY};
    int idist=1;
    int odist=1;
    int istride=osize_1[2];
    int ostride=osize_1[2];
    int batch=osize_1[2];

    if(batch!=0)
    {
      cufft_error=cufftPlanMany(&plan->fplan_1, 1, &n[1],
          f_inembed, istride, idist, // *inembed, istride, idist
          f_onembed, ostride, odist, // *onembed, ostride, odist
          CUFFT_C2C, batch);
      if(cufft_error!= CUFFT_SUCCESS)
      {
        fprintf(stderr, "CUFFT error: fplan_1 creation failed %d \n",cufft_error); return NULL;
      }
      //cufftSetCompatibilityMode(fplan,CUFFT_COMPATIBILITY_FFTW_PADDING); if (cudaGetLastError() != cudaSuccess){fprintf(stderr, "Cuda error:Failed at fplan cuda compatibility\n"); return;}
    }
  }
  // fplan_2
  {
    int f_inembed[1]={NX};
    int f_onembed[1]={NX};
    int idist=1;
    int odist=1;
    int istride=osize_2[1]*osize_2[2];
    int ostride=osize_2[1]*osize_2[2];
    int batch=osize_2[1]*osize_2[2];;

    if(batch!=0)
    {
      cufft_error=cufftPlanMany(&plan->fplan_2, 1, &n[0],
          f_inembed, istride, idist, // *inembed, istride, idist
          f_onembed, ostride, odist, // *onembed, ostride, odist
          CUFFT_C2C, batch);
      if(cufft_error!= CUFFT_SUCCESS)
      {
        fprintf(stderr, "CUFFT error: fplan_2 creation failed %d \n",cufft_error); return NULL;
      }
      //cufftSetCompatibilityMode(fplan,CUFFT_COMPATIBILITY_FFTW_PADDING); if (cudaGetLastError() != cudaSuccess){fprintf(stderr, "Cuda error:Failed at fplan cuda compatibility\n"); return;}
    }
  }


  // 1D Decomposition
  if(plan->oneD){
    int N0=n[0], N1=n[1], N2=n[2];

    plan->Mem_mgr  = new Mem_Mgr_gpu<float>(N0,N1,n_tuples_o,c_comm);
    plan->T_plan_2 = new T_Plan_gpu <float>(N0,N1,n_tuples_o, plan->Mem_mgr, c_comm);
    plan->T_plan_2i= new T_Plan_gpu <float>(N1,N0,n_tuples_o,plan->Mem_mgr, c_comm);
    plan->T_plan_1=NULL;
    plan->T_plan_1i=NULL;

    plan->alloc_max=alloc_max;
    plan->T_plan_2->alloc_local=alloc_max;
    plan->T_plan_2i->alloc_local=alloc_max;


    if(flags==ACCFFT_MEASURE){
      plan->T_plan_2->which_fast_method_gpu(plan->T_plan_2,data_out_d);
    }
    else{
      plan->T_plan_2->method=2;
      plan->T_plan_2->kway=2;
    }
    checkCuda_accfft (cudaDeviceSynchronize());
    MPI_Barrier(plan->c_comm);

    plan->T_plan_2->method =plan->T_plan_2->method;
    plan->T_plan_2i->method=plan->T_plan_2->method;

    plan->T_plan_2->kway =plan->T_plan_2->kway;
    plan->T_plan_2i->kway=plan->T_plan_2->kway;


  }

  // 2D Decomposition
  if (!plan->oneD){
    // the reaseon for n_tuples/2 is to avoid splitting of imag and real parts of complex numbers
    plan->Mem_mgr  =new Mem_Mgr_gpu<float>(n[1],n_tuples_o/2,2,plan->row_comm,osize_0[0],alloc_max);
    plan->T_plan_1 = new T_Plan_gpu<float>(n[1],n_tuples_o/2,2, plan->Mem_mgr, plan->row_comm,osize_0[0]);
    plan->T_plan_2 = new T_Plan_gpu<float>(n[0],n[1],osize_2[2]*2,plan->Mem_mgr, plan->col_comm);
    plan->T_plan_2i= new T_Plan_gpu<float>(n[1],n[0],osize_2i[2]*2, plan->Mem_mgr, plan->col_comm);
    plan->T_plan_1i= new T_Plan_gpu<float>(n_tuples_o/2,n[1],2, plan->Mem_mgr, plan->row_comm,osize_1i[0]);


    plan->T_plan_1->alloc_local=plan->alloc_max;
    plan->T_plan_2->alloc_local=plan->alloc_max;
    plan->T_plan_2i->alloc_local=plan->alloc_max;
    plan->T_plan_1i->alloc_local=plan->alloc_max;



    if(flags==ACCFFT_MEASURE){
      if(coord[0]==0){
        plan->T_plan_1->which_fast_method_gpu(plan->T_plan_1,data_out_d,osize_0[0]);
      }
    }
    else{
      plan->T_plan_1->method=2;
      plan->T_plan_1->kway=2;
    }

    MPI_Bcast(&plan->T_plan_1->method,1, MPI_INT,0, c_comm );
    MPI_Bcast(&plan->T_plan_1->kway,1, MPI_INT,0, c_comm );

    checkCuda_accfft (cudaDeviceSynchronize());
    MPI_Barrier(plan->c_comm);
    plan->T_plan_1->method =plan->T_plan_1->method;
    plan->T_plan_2->method =plan->T_plan_1->method;
    plan->T_plan_2i->method=plan->T_plan_1->method;
    plan->T_plan_1i->method=plan->T_plan_1->method;
    plan->T_plan_1->kway =plan->T_plan_1->kway;
    plan->T_plan_2->kway =plan->T_plan_1->kway;
    plan->T_plan_2i->kway=plan->T_plan_1->kway;
    plan->T_plan_1i->kway=plan->T_plan_1->kway;

    plan->iplan_1=-1;
    plan->iplan_2=-1;

  }

  plan->r2c_plan_baked=true;
  return plan;

}


void accfft_execute_gpuf(accfft_plan_gpuf* plan, int direction,float * data_d, float * data_out_d, double * timer,std::bitset<3> xyz){

  if(data_d==NULL)
    data_d=plan->data;
  if(data_out_d==NULL)
    data_out_d=plan->data_out;

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

  cudaEvent_t memcpy_startEvent, memcpy_stopEvent;
  cudaEvent_t fft_startEvent, fft_stopEvent;
  checkCuda_accfft( cudaEventCreate(&memcpy_startEvent) );
  checkCuda_accfft( cudaEventCreate(&memcpy_stopEvent) );
  checkCuda_accfft( cudaEventCreate(&fft_startEvent) );
  checkCuda_accfft( cudaEventCreate(&fft_stopEvent) );
  int NY=plan->N[1];
  float dummy_time=0;


  int *osize_0 =plan->osize_0;// *ostart_0 =plan->ostart_0;
  int *osize_1 =plan->osize_1;// *ostart_1 =plan->ostart_1;
  //int *osize_2 =plan->osize_2, *ostart_2 =plan->ostart_2;
  int *osize_1i=plan->osize_1i;//*ostart_1i=plan->ostart_1i;
  //int *osize_2i=plan->osize_2i,*ostart_2i=plan->ostart_2i;

  if(direction==-1){
    /**************************************************************/
    /*******************  N0/P0 x N1/P1 x N2 **********************/
    /**************************************************************/
    // FFT in Z direction
    if(xyz[0]){
      checkCuda_accfft( cudaEventRecord(fft_startEvent,0) );
      checkCuda_accfft (cufftExecR2C(plan->fplan_0,(cufftReal*)data_d, (cufftComplex*)data_out_d));
      checkCuda_accfft( cudaEventRecord(fft_stopEvent,0) );
      checkCuda_accfft( cudaEventSynchronize(fft_stopEvent) ); // wait until fft is executed
      checkCuda_accfft( cudaEventElapsedTime(&dummy_time, fft_startEvent, fft_stopEvent) );
      fft_time+=dummy_time/1000;
    }
    else
      data_out_d=data_d;

    // Perform N0/P0 transpose


    if(!plan->oneD){
      plan->T_plan_1->execute_gpu(plan->T_plan_1,data_out_d,timings,2,osize_0[0],coords[0]);
    }
    /**************************************************************/
    /*******************  N0/P0 x N1 x N2/P1 **********************/
    /**************************************************************/
    if(xyz[1]){
      checkCuda_accfft( cudaEventRecord(fft_startEvent,0) );
      for (int i=0;i<osize_1[0];++i){
        checkCuda_accfft (cufftExecC2C(plan->fplan_1,(cufftComplex*)&data_out_d[2*i*osize_1[1]*osize_1[2]], (cufftComplex*)&data_out_d[2*i*osize_1[1]*osize_1[2]],CUFFT_FORWARD));
      }
      checkCuda_accfft( cudaEventRecord(fft_stopEvent,0) );
      checkCuda_accfft( cudaEventSynchronize(fft_stopEvent) ); // wait until fft is executed
      checkCuda_accfft( cudaEventElapsedTime(&dummy_time, fft_startEvent, fft_stopEvent) );
      fft_time+=dummy_time/1000;
      MPI_Barrier(plan->c_comm);
    }

    if(plan->oneD){
      plan->T_plan_2->execute_gpu(plan->T_plan_2,data_out_d,timings,2);
    }
    else{
      plan->T_plan_2->execute_gpu(plan->T_plan_2,data_out_d,timings,2,1,coords[1]);
    }
    /**************************************************************/
    /*******************  N0 x N1/P0 x N2/P1 **********************/
    /**************************************************************/
    if(xyz[2]){
      checkCuda_accfft( cudaEventRecord(fft_startEvent,0) );
      checkCuda_accfft (cufftExecC2C(plan->fplan_2,(cufftComplex*)data_out_d, (cufftComplex*)data_out_d,CUFFT_FORWARD));
      checkCuda_accfft( cudaEventRecord(fft_stopEvent,0) );
      checkCuda_accfft( cudaEventSynchronize(fft_stopEvent) ); // wait until fft is executed
      checkCuda_accfft( cudaEventElapsedTime(&dummy_time, fft_startEvent, fft_stopEvent) );
      fft_time+=dummy_time/1000;
    }
  }
  else if (direction==1){
    if(xyz[2]){
      checkCuda_accfft( cudaEventRecord(fft_startEvent,0) );
      checkCuda_accfft (cufftExecC2C(plan->fplan_2,(cufftComplex*)data_d, (cufftComplex*)data_d,CUFFT_INVERSE));
      checkCuda_accfft( cudaEventRecord(fft_stopEvent,0) );
      checkCuda_accfft( cudaEventSynchronize(fft_stopEvent) ); // wait until fft is executed
      checkCuda_accfft( cudaEventElapsedTime(&dummy_time, fft_startEvent, fft_stopEvent) );
      fft_time+=dummy_time/1000;
      MPI_Barrier(plan->c_comm);
    }

    if(plan->oneD){
      plan->T_plan_2i->execute_gpu(plan->T_plan_2i,(float*)data_d,timings,1);
    }
    else{
      plan->T_plan_2i->execute_gpu(plan->T_plan_2i,(float*)data_d,timings,1,1,coords[1]);
    }
    /**************************************************************/
    /*******************  N0/P0 x N1 x N2/P1 **********************/
    /**************************************************************/
    if(xyz[1]){
      checkCuda_accfft( cudaEventRecord(fft_startEvent,0) );
      for (int i=0;i<osize_1i[0];++i){
        checkCuda_accfft (cufftExecC2C(plan->fplan_1,(cufftComplex*)&data_d[2*i*NY*osize_1i[2]], (cufftComplex*)&data_d[2*i*NY*osize_1i[2]],CUFFT_INVERSE));
      }
      checkCuda_accfft( cudaEventRecord(fft_stopEvent,0) );
      checkCuda_accfft( cudaEventSynchronize(fft_stopEvent) ); // wait until fft is executed
      checkCuda_accfft( cudaEventElapsedTime(&dummy_time, fft_startEvent, fft_stopEvent) );
      fft_time+=dummy_time/1000;
      MPI_Barrier(plan->c_comm);
    }



    if(!plan->oneD){
      plan->T_plan_1i->execute_gpu(plan->T_plan_1i,(float*)data_d,timings,1,osize_1i[0],coords[0]);
    }
    MPI_Barrier(plan->c_comm);
    /**************************************************************/
    /*******************  N0/P0 x N1/P1 x N2 **********************/
    /**************************************************************/

    // IFFT in Z direction
    if(xyz[0]){
      checkCuda_accfft( cudaEventRecord(fft_startEvent,0) );
      checkCuda_accfft (cufftExecC2R(plan->iplan_0,(cufftComplex*)data_d,(cufftReal*)data_out_d));
      checkCuda_accfft( cudaEventRecord(fft_stopEvent,0) );
      checkCuda_accfft( cudaEventSynchronize(fft_stopEvent) ); // wait until fft is executed
      checkCuda_accfft( cudaEventElapsedTime(&dummy_time, fft_startEvent, fft_stopEvent) );
      fft_time+=dummy_time/1000;
    }
    else
      data_out_d=data_d;

  }

  timings[4]=fft_time;
  if(timer==NULL){
    delete [] timings;
  }
  MPI_Barrier(plan->c_comm);

  return;
}


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
void accfft_execute_r2c_gpuf(accfft_plan_gpuf* plan, float * data,Complexf * data_out, double * timer,std::bitset<3> xyz){
  if(plan->r2c_plan_baked){
    accfft_execute_gpuf(plan,-1,data,(float*)data_out,timer,xyz);
  }
  else{
    if(plan->procid==0) std::cout<<"Error. r2c plan has not been made correctly. Please first create the plan before calling execute functions."<<std::endl;
  }

  return;
}


/**
 * Execute C2R plan. This function is blocking and only returns after the transform is completed.
 * @note For inplace transforms, data_out should point to the same memory address as data, AND
 * the plan must have been created as inplace.
 * @param plan FFT plan created by \ref accfft_plan_dft_3d_r2c.
 * @param data Input data in frequency domain.
 * @param data_out Output data in frequency domain.
 * @param timer See \ref timer for more details.
 * @param XYZ a bit set field that determines which directions FFT should be executed
 */
void accfft_execute_c2r_gpuf(accfft_plan_gpuf* plan, Complexf * data,float * data_out, double* timer,std::bitset<3> xyz){
  if(plan->r2c_plan_baked){
    accfft_execute_gpuf(plan,1,(float*)data,data_out,timer,xyz);
  }
  else{
    if(plan->procid==0) std::cout<<"Error. r2c plan has not been made correctly. Please first create the plan before calling execute functions."<<std::endl;
  }

  return;
}


/**
 * Get the local sizes of the distributed global data for a GPU C2C transform
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
int accfft_local_size_dft_c2c_gpuf( int * n,int * isize, int * istart, int * osize, int *ostart,MPI_Comm c_comm){

  int osize_0[3]={0}, ostart_0[3]={0};
  int osize_1[3]={0}, ostart_1[3]={0};
  int osize_2[3]={0}, ostart_2[3]={0};
  //int osize_1i[3]={0}, ostart_1i[3]={0};
  //int osize_2i[3]={0}, ostart_2i[3]={0};

  int alloc_local;
  int alloc_max=0;//,n_tuples=n[2]*2;
  alloc_local=dfft_get_local_size_gpuf(n[0],n[1],n[2],osize_0,ostart_0,c_comm);
  alloc_max=std::max(alloc_max, alloc_local);
  alloc_local=dfft_get_local_size_gpuf(n[0],n[2],n[1],osize_1,ostart_1,c_comm);
  alloc_max=std::max(alloc_max, alloc_local);
  alloc_local=dfft_get_local_size_gpuf(n[1],n[2],n[0],osize_2,ostart_2,c_comm);
  alloc_max=std::max(alloc_max, alloc_local);
  alloc_max*=2; // because of c2c

  std::swap(osize_1[1],osize_1[2]);
  std::swap(ostart_1[1],ostart_1[2]);

  std::swap(ostart_2[1],ostart_2[2]);
  std::swap(ostart_2[0],ostart_2[1]);
  std::swap(osize_2[1],osize_2[2]);
  std::swap(osize_2[0],osize_2[1]);

  //isize[0]=osize_0[0];
  //isize[1]=osize_0[1];
  //isize[2]=n[2];//osize_0[2];
  dfft_get_local_size_gpuf(n[0],n[1],n[2],isize,istart,c_comm);

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
accfft_plan_gpuf*  accfft_plan_dft_3d_c2c_gpuf(int * n, Complexf * data_d, Complexf * data_out_d, MPI_Comm c_comm,unsigned flags){
  accfft_plan_gpuf *plan=new accfft_plan_gpuf;
  int nprocs, procid;
  MPI_Comm_rank(c_comm, &procid);
  plan->procid=procid;
  MPI_Cart_get(c_comm,2,plan->np,plan->periods,plan->coord);
  plan->c_comm=c_comm;
  int *coord=plan->coord;
  MPI_Comm_split(c_comm,coord[0],coord[1],&plan->row_comm);
  MPI_Comm_split(c_comm,coord[1],coord[0],&plan->col_comm);
  plan->N[0]=n[0];plan->N[1]=n[1];plan->N[2]=n[2];
  int NX=n[0], NY=n[1], NZ=n[2];
  cufftResult_t cufft_error;

  plan->data_c=data_d;
  plan->data_out_c=data_out_d;
  if(data_out_d==data_d){
    plan->inplace=true;}
  else{plan->inplace=false;}

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
  int alloc_max=0,n_tuples=n[2]*2;

  int isize[3],osize[3],istart[3],ostart[3];
  alloc_max=accfft_local_size_dft_c2c_gpuf(n,isize,istart,osize,ostart,c_comm);
  plan->alloc_max=alloc_max;

  dfft_get_local_size_gpuf(n[0],n[1],n[2],osize_0,ostart_0,c_comm);
  dfft_get_local_size_gpuf(n[0],n[2],n[1],osize_1,ostart_1,c_comm);
  dfft_get_local_size_gpuf(n[1],n[2],n[0],osize_2,ostart_2,c_comm);


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


  // fplan_0
  {
    int f_inembed[1]={NZ};
    int f_onembed[1]={NZ};
    int idist=(NZ);
    int odist=(NZ);
    int istride=1;
    int ostride=1;
    int batch=osize_0[0]*osize_0[1];//NX;

    if(batch!=0)
    {
      cufft_error=cufftPlanMany(&plan->fplan_0, 1, &n[2],
          f_inembed, istride, idist, // *inembed, istride, idist
          f_onembed, ostride, odist, // *onembed, ostride, odist
          CUFFT_C2C, batch);
      if(cufft_error!= CUFFT_SUCCESS)
      {
        fprintf(stderr, "CUFFT error: fplan_0 creation failed %d \n",cufft_error); return NULL;
      }
      //cufftSetCompatibilityMode(fplan,CUFFT_COMPATIBILITY_FFTW_PADDING); if (cudaGetLastError() != cudaSuccess){fprintf(stderr, "Cuda error:Failed at fplan cuda compatibility\n"); return;}
    }
  }
  // fplan_1
  {
    int f_inembed[1]={NY};
    int f_onembed[1]={NY};
    int idist=1;
    int odist=1;
    int istride=osize_1[2];
    int ostride=osize_1[2];
    int batch=osize_1[2];

    if(batch!=0)
    {
      cufft_error=cufftPlanMany(&plan->fplan_1, 1, &n[1],
          f_inembed, istride, idist, // *inembed, istride, idist
          f_onembed, ostride, odist, // *onembed, ostride, odist
          CUFFT_C2C, batch);
      if(cufft_error!= CUFFT_SUCCESS)
      {
        fprintf(stderr, "CUFFT error: fplan_1 creation failed %d \n",cufft_error); return NULL;
      }
      //cufftSetCompatibilityMode(fplan,CUFFT_COMPATIBILITY_FFTW_PADDING); if (cudaGetLastError() != cudaSuccess){fprintf(stderr, "Cuda error:Failed at fplan cuda compatibility\n"); return;}
    }
  }
  // fplan_2
  {
    int f_inembed[1]={NX};
    int f_onembed[1]={NX};
    int idist=1;
    int odist=1;
    int istride=osize_2[1]*osize_2[2];
    int ostride=osize_2[1]*osize_2[2];
    int batch=osize_2[1]*osize_2[2];;

    if(batch!=0)
    {
      cufft_error=cufftPlanMany(&plan->fplan_2, 1, &n[0],
          f_inembed, istride, idist, // *inembed, istride, idist
          f_onembed, ostride, odist, // *onembed, ostride, odist
          CUFFT_C2C, batch);
      if(cufft_error!= CUFFT_SUCCESS)
      {
        fprintf(stderr, "CUFFT error: fplan_2 creation failed %d \n",cufft_error); return NULL;
      }
      //cufftSetCompatibilityMode(fplan,CUFFT_COMPATIBILITY_FFTW_PADDING); if (cudaGetLastError() != cudaSuccess){fprintf(stderr, "Cuda error:Failed at fplan cuda compatibility\n"); return;}
    }
  }

  // 1D Decomposition
  if (plan->oneD){
    int NX=n[0],NY=n[1],NZ=n[2];


    plan->alloc_max=alloc_max;

    plan->Mem_mgr= new Mem_Mgr_gpu <float>(NX,NY,(NZ)*2,c_comm);
    plan->T_plan_2= new T_Plan_gpu <float>(NX,NY,(NZ)*2, plan->Mem_mgr,c_comm);
    plan->T_plan_2i= new T_Plan_gpu<float>(NY,NX,NZ*2, plan->Mem_mgr,c_comm);

    plan->T_plan_2->alloc_local=alloc_max;
    plan->T_plan_2i->alloc_local=alloc_max;
    plan->T_plan_1=NULL;
    plan->T_plan_1i=NULL;




    if(flags==ACCFFT_MEASURE){
      plan->T_plan_2->which_fast_method_gpu(plan->T_plan_2,(float*)data_out_d);
    }
    else{
      plan->T_plan_2->method=2;
      plan->T_plan_2->kway=2;
    }
    checkCuda_accfft (cudaDeviceSynchronize());
    MPI_Barrier(plan->c_comm);

    plan->T_plan_2->method =plan->T_plan_2->method;
    plan->T_plan_2i->method=plan->T_plan_2->method;

    plan->T_plan_2->kway =plan->T_plan_2->kway;
    plan->T_plan_2i->kway=plan->T_plan_2->kway;


  }

  // 2D Decomposition
  if (!plan->oneD){
    // the reaseon for n_tuples/2 is to avoid splitting of imag and real parts of complex numbers
    plan->Mem_mgr=  new Mem_Mgr_gpu<float>(n[1],n[2],2,plan->row_comm,osize_0[0],alloc_max);
    plan->T_plan_1= new  T_Plan_gpu<float>(n[1],n[2],2, plan->Mem_mgr, plan->row_comm,osize_0[0]);
    plan->T_plan_2= new  T_Plan_gpu<float>(n[0],n[1],2*osize_2[2], plan->Mem_mgr, plan->col_comm);
    plan->T_plan_2i= new T_Plan_gpu<float>(n[1],n[0],2*osize_2i[2], plan->Mem_mgr, plan->col_comm);
    plan->T_plan_1i= new T_Plan_gpu<float>(n[2],n[1],2, plan->Mem_mgr, plan->row_comm,osize_1i[0]);

    plan->T_plan_1->alloc_local=plan->alloc_max;
    plan->T_plan_2->alloc_local=plan->alloc_max;
    plan->T_plan_2i->alloc_local=plan->alloc_max;
    plan->T_plan_1i->alloc_local=plan->alloc_max;


    plan->iplan_0=NULL;
    plan->iplan_1=NULL;
    plan->iplan_2=NULL;

    int coords[2],np[2],periods[2];
    MPI_Cart_get(c_comm,2,np,periods,coords);

    if(flags==ACCFFT_MEASURE){
      if(coords[0]==0){
        plan->T_plan_1->which_fast_method_gpu(plan->T_plan_1,(float*)data_out_d,osize_0[0]);
      }
    }
    else{
      plan->T_plan_1->method=2;
      plan->T_plan_1->kway=2;
    }

    MPI_Bcast(&plan->T_plan_1->method,1, MPI_INT,0, c_comm );
    MPI_Bcast(&plan->T_plan_1->kway,1, MPI_INT,0, c_comm );
    checkCuda_accfft (cudaDeviceSynchronize());
    MPI_Barrier(plan->c_comm);


    plan->T_plan_1->method =plan->T_plan_1->method;
    plan->T_plan_2->method =plan->T_plan_1->method;
    plan->T_plan_2i->method=plan->T_plan_1->method;
    plan->T_plan_1i->method=plan->T_plan_1->method;

    plan->T_plan_1->kway =plan->T_plan_1->kway;
    plan->T_plan_2->kway =plan->T_plan_1->kway;
    plan->T_plan_2i->kway=plan->T_plan_1->kway;
    plan->T_plan_1i->kway=plan->T_plan_1->kway;


  }

  plan->c2c_plan_baked=true;
  return plan;
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
void accfft_execute_c2c_gpuf(accfft_plan_gpuf* plan, int direction,Complexf * data_d, Complexf * data_out_d, double * timer,std::bitset<3> xyz){

  if(!plan->c2c_plan_baked){
    if(plan->procid==0) std::cout<<"Error. r2c plan has not been made correctly. Please first create the plan before calling execute functions."<<std::endl;
    return;
  }

  if(data_d==NULL)
    data_d=plan->data_c;
  if(data_out_d==NULL)
    data_out_d=plan->data_out_c;
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

  int NY=plan->N[1];
  cudaEvent_t memcpy_startEvent, memcpy_stopEvent;
  cudaEvent_t fft_startEvent, fft_stopEvent;
  checkCuda_accfft( cudaEventCreate(&memcpy_startEvent) );
  checkCuda_accfft( cudaEventCreate(&memcpy_stopEvent) );
  checkCuda_accfft( cudaEventCreate(&fft_startEvent) );
  checkCuda_accfft( cudaEventCreate(&fft_stopEvent) );
  cufftResult_t cufft_error;
  float dummy_time=0;

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
    if(xyz[0]){
      checkCuda_accfft( cudaEventRecord(fft_startEvent,0) );
      checkCuda_accfft(cufftExecC2C(plan->fplan_0,(cufftComplex*)data_d, (cufftComplex*)data_out_d,CUFFT_FORWARD));
      checkCuda_accfft( cudaEventRecord(fft_stopEvent,0) );
      checkCuda_accfft( cudaEventSynchronize(fft_stopEvent) ); // wait until fft is executed
      checkCuda_accfft( cudaEventElapsedTime(&dummy_time, fft_startEvent, fft_stopEvent) );
      fft_time+=dummy_time/1000;
    }
    else
      data_out_d=data_d;

    if(!plan->oneD){
      plan->T_plan_1->execute_gpu(plan->T_plan_1,(float*)data_out_d,timings,2,osize_0[0],coords[0]);
    }
    checkCuda_accfft (cudaDeviceSynchronize());
    MPI_Barrier(plan->c_comm);
    /**************************************************************/
    /*******************  N0/P0 x N1 x N2/P1 **********************/
    /**************************************************************/
    if(xyz[1]){
      checkCuda_accfft( cudaEventRecord(fft_startEvent,0) );
      for (int i=0;i<osize_1[0];++i){
        checkCuda_accfft(cufftExecC2C(plan->fplan_1,(cufftComplex*)&data_out_d[i*osize_1[1]*osize_1[2]], (cufftComplex*)&data_out_d[i*osize_1[1]*osize_1[2]],CUFFT_FORWARD));
      }
      checkCuda_accfft( cudaEventRecord(fft_stopEvent,0) );
      checkCuda_accfft( cudaEventSynchronize(fft_stopEvent) ); // wait until fft is executed
      checkCuda_accfft( cudaEventElapsedTime(&dummy_time, fft_startEvent, fft_stopEvent) );
      fft_time+=dummy_time/1000;
    }

    MPI_Barrier(plan->c_comm);



    if(plan->oneD){
      plan->T_plan_2->execute_gpu(plan->T_plan_2,(float*)data_out_d,timings,2);
    }
    else{
      plan->T_plan_2->execute_gpu(plan->T_plan_2,(float*)data_out_d,timings,2,1,coords[1]);
    }
    checkCuda_accfft (cudaDeviceSynchronize());
    MPI_Barrier(plan->c_comm);
    /**************************************************************/
    /*******************  N0 x N1/P0 x N2/P1 **********************/
    /**************************************************************/
    if(xyz[2]){
      checkCuda_accfft( cudaEventRecord(fft_startEvent,0) );
      checkCuda_accfft(cufftExecC2C(plan->fplan_2,(cufftComplex*)data_out_d, (cufftComplex*)data_out_d,CUFFT_FORWARD));
      checkCuda_accfft( cudaEventRecord(fft_stopEvent,0) );
      checkCuda_accfft( cudaEventSynchronize(fft_stopEvent) ); // wait until fft is executed
      checkCuda_accfft( cudaEventElapsedTime(&dummy_time, fft_startEvent, fft_stopEvent) );
      fft_time+=dummy_time/1000;
    }

  }
  else if (direction==1){
    if(xyz[2]){
      checkCuda_accfft( cudaEventRecord(fft_startEvent,0) );
      checkCuda_accfft (cufftExecC2C(plan->fplan_2,(cufftComplex*)data_d, (cufftComplex*)data_d,CUFFT_INVERSE));
      checkCuda_accfft (cudaDeviceSynchronize());
      checkCuda_accfft( cudaEventRecord(fft_stopEvent,0) );
      checkCuda_accfft( cudaEventSynchronize(fft_stopEvent) ); // wait until fft is executed
      checkCuda_accfft( cudaEventElapsedTime(&dummy_time, fft_startEvent, fft_stopEvent) );
      fft_time+=dummy_time/1000;
      MPI_Barrier(plan->c_comm);
    }


    if(plan->oneD){
      plan->T_plan_2i->execute_gpu(plan->T_plan_2i,(float*)data_d,timings,1);
    }
    else{
      plan->T_plan_2i->execute_gpu(plan->T_plan_2i,(float*)data_d,timings,1,1,coords[1]);
    }
    checkCuda_accfft (cudaDeviceSynchronize());
    MPI_Barrier(plan->c_comm);
    /**************************************************************/
    /*******************  N0/P0 x N1 x N2/P1 **********************/
    /**************************************************************/
    if(xyz[1]){
      checkCuda_accfft( cudaEventRecord(fft_startEvent,0) );
      for (int i=0;i<osize_1i[0];++i){
        checkCuda_accfft (cufftExecC2C(plan->fplan_1,(cufftComplex*)&data_d[i*NY*osize_1i[2]], (cufftComplex*)&data_d[i*NY*osize_1i[2]],CUFFT_INVERSE));
      }
      checkCuda_accfft( cudaEventRecord(fft_stopEvent,0) );
      checkCuda_accfft( cudaEventSynchronize(fft_stopEvent) ); // wait until fft is executed
      checkCuda_accfft( cudaEventElapsedTime(&dummy_time, fft_startEvent, fft_stopEvent) );
      fft_time+=dummy_time/1000;
    }
    MPI_Barrier(plan->c_comm);

    if(!plan->oneD){
      plan->T_plan_1i->execute_gpu(plan->T_plan_1i,(float*)data_d,timings,1,osize_1i[0],coords[0]);
    }
    checkCuda_accfft (cudaDeviceSynchronize());
    MPI_Barrier(plan->c_comm);
    /**************************************************************/
    /*******************  N0/P0 x N1/P1 x N2 **********************/
    /**************************************************************/

    if(xyz[0]){
      checkCuda_accfft( cudaEventRecord(fft_startEvent,0) );
      checkCuda_accfft (cufftExecC2C(plan->fplan_0,(cufftComplex*)data_d,(cufftComplex*)data_out_d,CUFFT_INVERSE));
      checkCuda_accfft( cudaEventRecord(fft_stopEvent,0) );
      checkCuda_accfft( cudaEventSynchronize(fft_stopEvent) ); // wait until fft is executed
      checkCuda_accfft( cudaEventElapsedTime(&dummy_time, fft_startEvent, fft_stopEvent) );
      fft_time+=dummy_time/1000;
    }
    else
      data_out_d=data_d;

  }

  timings[4]=fft_time;
  if(timer==NULL){
    delete [] timings;
  }
  MPI_Barrier(plan->c_comm);

  return;
}


/**
 * Destroy AccFFT CPU plan. This function calls \ref accfft_destroy_plan_gpu.
 * @param plan Input plan to be destroyed.
 */
void accfft_destroy_plan(accfft_plan_gpuf * plan){
  return (accfft_destroy_plan_gpu(plan));
}

/**
 * Destroy AccFFT GPU plan.
 * @param plan Input plan to be destroyed.
 */
void accfft_destroy_plan_gpu(accfft_plan_gpuf * plan){

  if(plan->T_plan_1!=NULL)delete(plan->T_plan_1);
  if(plan->T_plan_1i!=NULL)delete(plan->T_plan_1i);
  if(plan->T_plan_2!=NULL)delete(plan->T_plan_2);
  if(plan->T_plan_2i!=NULL)delete(plan->T_plan_2i);
  if(plan->Mem_mgr!=NULL)delete(plan->Mem_mgr);

  if(plan->fplan_0!=-1)cufftDestroy(plan->fplan_0);
  if(plan->fplan_1!=-1)cufftDestroy(plan->fplan_1);
  if(plan->fplan_2!=-1)cufftDestroy(plan->fplan_2);

  if(plan->iplan_0!=-1)cufftDestroy(plan->iplan_0);
  if(plan->iplan_1!=-1)cufftDestroy(plan->iplan_1);
  if(plan->iplan_2!=-1)cufftDestroy(plan->iplan_2);

  MPI_Comm_free(&plan->row_comm);
  MPI_Comm_free(&plan->col_comm);
  return;
}


template <typename T,typename Tc>
void accfft_execute_r2c_gpu_t(accfft_plan_gpuf* plan, T* data,Tc* data_out, double * timer,std::bitset<3> XYZ){
  accfft_execute_r2c_gpuf(plan,data,data_out,timer,XYZ);
  return;
}
template <typename Tc, typename T>
void accfft_execute_c2r_gpu_t(accfft_plan_gpuf* plan, Tc* data,T* data_out, double * timer,std::bitset<3> XYZ){
  accfft_execute_c2r_gpuf(plan,data,data_out,timer,XYZ);
  return;
}
template void accfft_execute_r2c_gpu_t<float,Complexf>(accfft_plan_gpuf* plan, float*    data,Complexf* data_out, double * timer,std::bitset<3> XYZ);
template void accfft_execute_c2r_gpu_t<Complexf,float>(accfft_plan_gpuf* plan, Complexf* data,float*    data_out, double * timer,std::bitset<3> XYZ);

template <typename T>
int accfft_local_size_dft_r2c_gpu_t( int * n,int * isize, int * istart, int * osize, int *ostart,MPI_Comm c_comm){
  return accfft_local_size_dft_r2c_gpuf(n,isize,istart,osize,ostart,c_comm);
}
template int accfft_local_size_dft_r2c_gpu_t<float>( int * n,int * isize, int * istart, int * osize, int *ostart,MPI_Comm c_comm);
template int accfft_local_size_dft_r2c_gpu_t<Complexf>( int * n,int * isize, int * istart, int * osize, int *ostart,MPI_Comm c_comm);

