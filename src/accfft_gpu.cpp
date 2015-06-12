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

#include "accfft_gpu.h"
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
typedef double Complex[2];

void accfft_cleanup_gpu(){
  // empty for now
}

int dfft_get_local_size_gpu(int N0, int N1, int N2, int * isize, int * istart,MPI_Comm c_comm ){
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
  int alloc_local=isize[0]*isize[1]*isize[2]*sizeof(double);


  return alloc_local;
}

int accfft_local_size_dft_r2c_gpu( int * n,int * isize, int * istart, int * osize, int *ostart,MPI_Comm c_comm, bool inplace){

  //1D & 2D Decomp
  int osize_0[3]={0}, ostart_0[3]={0};
  int osize_1[3]={0}, ostart_1[3]={0};
  int osize_2[3]={0}, ostart_2[3]={0};

  int alloc_local;
  int alloc_max=0,n_tuples;
  //inplace==true ? n_tuples=(n[2]/2+1)*2:  n_tuples=n[2];
  n_tuples=(n[2]/2+1)*2;//SNAFU
  alloc_local=dfft_get_local_size_gpu(n[0],n[1],n_tuples,osize_0,ostart_0,c_comm);
  alloc_max=std::max(alloc_max, alloc_local);
  alloc_local=dfft_get_local_size_gpu(n[0],n_tuples/2,n[1],osize_1,ostart_1,c_comm);
  alloc_max=std::max(alloc_max, alloc_local*2);
  alloc_local=dfft_get_local_size_gpu(n[1],n_tuples/2,n[0],osize_2,ostart_2,c_comm);
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
  dfft_get_local_size_gpu(n[0],n[1],n[2],isize,istart,c_comm);

  osize[0]=osize_2[0];
  osize[1]=osize_2[1];
  osize[2]=osize_2[2];

  ostart[0]=ostart_2[0];
  ostart[1]=ostart_2[1];
  ostart[2]=ostart_2[2];

  return alloc_max;

}

accfft_plan_gpu*  accfft_plan_dft_3d_r2c_gpu(int * n, double * data_d, double * data_out_d, MPI_Comm c_comm,unsigned flags){
  accfft_plan_gpu *plan=new accfft_plan_gpu;
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

  if(data_out_d==data_d){
    plan->inplace=true;}
  else{plan->inplace=false;}

  // 1D Decomposition
  if(plan->np[1]==1){
    int N0=n[0], N1=n[1], N2=n[2];
    int n_tuples_o,n_tuples_i;
//    plan->inplace==true ? n_tuples=(N2/2+1)*2:  n_tuples=N2*2;
    plan->inplace==true ? n_tuples_i=(N2/2+1)*2:  n_tuples_i=N2;
    n_tuples_o=(N2/2+1)*2;
    plan->Mem_mgr= new Mem_Mgr_gpu(N0,N1,n_tuples_o,c_comm);
    plan->T_plan_1= new T_Plan_gpu(N0,N1,n_tuples_o, plan->Mem_mgr, c_comm);
    plan->T_plan_1i= new T_Plan_gpu(N1,N0,n_tuples_o,plan->Mem_mgr, c_comm);

    int isize[3],osize[3],istart[3],ostart[3];
    int alloc_max=accfft_local_size_dft_r2c_gpu(n,isize,istart,osize,ostart,c_comm,plan->inplace);
    plan->alloc_max=alloc_max;
    plan->T_plan_1->alloc_local=alloc_max;
    plan->T_plan_1i->alloc_local=alloc_max;


    /* Forward Transform Plan. */
    {
      int NX=n[0], NY=n[1], NZ=n[2];
      int f_inembed[2]={NY,n_tuples_i};
      int f_onembed[2]={NY,n_tuples_o/2};
      int idist=NY*n_tuples_i;
      int odist=NY*n_tuples_o/2;
      int istride=1;
      int ostride=1;
      int batch=plan->T_plan_1->local_n0;//NX;

      cufftResult_t cufft_error;
      if(batch!=0)
      {
        cufft_error=cufftPlanMany(&plan->fplan_0, 2, &n[1],
            f_inembed, istride, idist, // *inembed, istride, idist
            f_onembed, ostride, odist, // *onembed, ostride, odist
            CUFFT_D2Z, batch);
        if(cufft_error!= CUFFT_SUCCESS){
          fprintf(stderr, "CUFFT error: fplan creation failed %d \n",cufft_error); return NULL;
        }
        //cufftSetCompatibilityMode(fplan,CUFFT_COMPATIBILITY_FFTW_PADDING); if (cudaGetLastError() != cudaSuccess){fprintf(stderr, "Cuda error:Failed at fplan cuda compatibility\n"); return;}
      }


      int local_n0=plan->T_plan_1->local_n0;
      int local_n1=plan->T_plan_1->local_n1;
      int f_inembed2[1]={NX};
      int f_onembed2[1]={NX};
      int idist2=1;
      int odist2=1;
      int istride2=local_n1*n_tuples_o/2;
      int ostride2=local_n1*n_tuples_o/2;
      if(plan->T_plan_1->local_n1*n_tuples_o/2!=0)
      {
        cufft_error=cufftPlanMany(&plan->fplan_1, 1, &n[0],
            f_inembed2, istride2, idist2, // *inembed, istride, idist
            f_onembed2, ostride2, odist2, // *onembed, ostride, odist
            CUFFT_Z2Z,plan->T_plan_1->local_n1*n_tuples_o/2);
        if(cufft_error!= CUFFT_SUCCESS){
          fprintf(stderr, "CUFFT error: fplan2 creation failed %d\n",cufft_error); return NULL;
        }
        //cufftSetCompatibilityMode(fplan2,CUFFT_COMPATIBILITY_FFTW_PADDING); if (cudaGetLastError() != cudaSuccess){fprintf(stderr, "Cuda error:Failed at fplan2 cuda compatibility\n"); return;}
      }

      /* Backward Transform Plan */
      if(batch!=0)
      {
        cufft_error=cufftPlanMany(&plan->iplan_0, 2, &n[1],
            f_onembed, ostride,odist , // *inembed, istride, idist
            f_inembed, istride,idist, // *onembed, ostride, odist
            CUFFT_Z2D, batch);
        if(cufft_error!= CUFFT_SUCCESS){
          fprintf(stderr, "CUFFT error: iplan creation failed %d\n",cufft_error); return NULL;
        }
        //cufftSetCompatibilityMode(iplan,CUFFT_COMPATIBILITY_FFTW_PADDING); if (cudaGetLastError() != cudaSuccess){fprintf(stderr, "Cuda error:Failed at cuda compatibility\n"); return;}
      }
    }


    static int method_static=0;
    static int kway_static_2=0;
    if(method_static==0){
      plan->T_plan_1->which_fast_method_gpu(plan->T_plan_1,data_out_d);
      method_static=plan->T_plan_1->method;
      kway_static_2=plan->T_plan_1->kway;//-4;
    }
    else{
      plan->T_plan_1->method=method_static;
      plan->T_plan_1->kway=kway_static_2;
    }

    checkCuda_accfft (cudaDeviceSynchronize());
    MPI_Barrier(plan->c_comm);

    plan->T_plan_1->method=plan->T_plan_1->method;
    plan->T_plan_1->kway=kway_static_2;
    plan->T_plan_1i->method=plan->T_plan_1->method;
    plan->T_plan_1i->kway=kway_static_2;

    // Make unused parts of plan NULL
    plan->T_plan_2=NULL;
    plan->T_plan_2i=NULL;
    plan->fplan_2=NULL;
    plan->iplan_2=NULL;

  }

  // 2D Decomposition
  if (plan->np[1]!=1){

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
    alloc_max=accfft_local_size_dft_r2c_gpu(n,isize,istart,osize,ostart,c_comm,plan->inplace);
    plan->alloc_max=alloc_max;

    dfft_get_local_size_gpu(n[0],n[1],n_tuples_o,osize_0,ostart_0,c_comm);
    dfft_get_local_size_gpu(n[0],n_tuples_o/2,n[1],osize_1,ostart_1,c_comm);
    dfft_get_local_size_gpu(n[1],n_tuples_o/2,n[0],osize_2,ostart_2,c_comm);

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

    // the reaseon for n_tuples/2 is to avoid splitting of imag and real parts of complex numbers
    plan->Mem_mgr= new Mem_Mgr_gpu(n[1],n_tuples_o/2,2,plan->row_comm,osize_0[0],alloc_max);
    plan->T_plan_1= new T_Plan_gpu(n[1],n_tuples_o/2,2, plan->Mem_mgr, plan->row_comm,osize_0[0]);
    plan->T_plan_2= new T_Plan_gpu(n[0],n[1],osize_2[2]*2,plan->Mem_mgr, plan->col_comm);
    plan->T_plan_2i= new T_Plan_gpu(n[1],n[0],osize_2i[2]*2, plan->Mem_mgr, plan->col_comm);
    plan->T_plan_1i= new T_Plan_gpu(n_tuples_o/2,n[1],2, plan->Mem_mgr, plan->row_comm,osize_1i[0]);


    plan->T_plan_1->alloc_local=plan->alloc_max;
    plan->T_plan_2->alloc_local=plan->alloc_max;
    plan->T_plan_2i->alloc_local=plan->alloc_max;
    plan->T_plan_1i->alloc_local=plan->alloc_max;

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
            CUFFT_D2Z, batch);
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
            CUFFT_Z2D, batch);
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
            CUFFT_Z2Z, batch);
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
            CUFFT_Z2Z, batch);
        if(cufft_error!= CUFFT_SUCCESS)
        {
          fprintf(stderr, "CUFFT error: fplan_2 creation failed %d \n",cufft_error); return NULL;
        }
        //cufftSetCompatibilityMode(fplan,CUFFT_COMPATIBILITY_FFTW_PADDING); if (cudaGetLastError() != cudaSuccess){fprintf(stderr, "Cuda error:Failed at fplan cuda compatibility\n"); return;}
      }
    }



    static int method_static_2=0;
    static int kway_static_2=0;
    if(method_static_2==0){
      if(coord[0]==0){
        plan->T_plan_1->which_fast_method_gpu(plan->T_plan_1,data_out_d);
        method_static_2=plan->T_plan_1->method;//-4;
        kway_static_2=plan->T_plan_1->kway;//-4;
      }
      MPI_Bcast(&method_static_2,1, MPI_INT,0, c_comm );
      MPI_Bcast(&kway_static_2,1, MPI_INT,0, c_comm );
      MPI_Barrier(c_comm);
    }
    checkCuda_accfft (cudaDeviceSynchronize());
    MPI_Barrier(plan->c_comm);
    plan->T_plan_1->method=method_static_2;
    plan->T_plan_2->method=method_static_2;
    plan->T_plan_2i->method=method_static_2;
    plan->T_plan_1i->method=method_static_2;
    plan->T_plan_1->kway=kway_static_2;
    plan->T_plan_2->kway=kway_static_2;
    plan->T_plan_2i->kway=kway_static_2;
    plan->T_plan_1i->kway=kway_static_2;

    plan->iplan_1=-1;
    plan->iplan_2=-1;

  }
  return plan;

}

void accfft_execute_gpu(accfft_plan_gpu* plan, int direction,double * data_d, double * data_out_d, double * timer){

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

  // 1D Decomposition
  if(plan->np[1]==1){
    if(direction==-1){

      /**************************************************************/
      /*******************  N0/P0 x N1 x N2 *************************/
      /**************************************************************/
      checkCuda_accfft( cudaEventRecord(fft_startEvent,0) );
      checkCuda_accfft(cufftExecD2Z(plan->fplan_0, (cufftDoubleReal*)data_d, (cufftDoubleComplex*)data_out_d));
      checkCuda_accfft( cudaEventRecord(fft_stopEvent,0) );
      checkCuda_accfft( cudaEventSynchronize(fft_stopEvent) ); // wait until fft is executed
      checkCuda_accfft( cudaEventElapsedTime(&dummy_time, fft_startEvent, fft_stopEvent) );
      fft_time+=dummy_time/1000;

      MPI_Barrier(plan->c_comm);
      plan->T_plan_1->execute_gpu(plan->T_plan_1,data_out_d,timings,2);
      /**************************************************************/
      /*******************  N1 x N0/P0 x N2 *************************/
      /**************************************************************/
      checkCuda_accfft( cudaEventRecord(fft_startEvent,0) );
      checkCuda_accfft(cufftExecZ2Z(plan->fplan_1,(cufftDoubleComplex*)data_out_d, (cufftDoubleComplex*)data_out_d,CUFFT_FORWARD));
      checkCuda_accfft( cudaEventRecord(fft_stopEvent,0) );
      checkCuda_accfft( cudaEventSynchronize(fft_stopEvent) ); // wait until fft is executed
      checkCuda_accfft( cudaEventElapsedTime(&dummy_time, fft_startEvent, fft_stopEvent) );
      fft_time+=dummy_time/1000;

    }

    if(direction==1){
      /* Now Perform the inverse transform  */
      checkCuda_accfft( cudaEventRecord(fft_startEvent,0) );
      checkCuda_accfft(cufftExecZ2Z(plan->fplan_1,(cufftDoubleComplex*)data_d,(cufftDoubleComplex*)data_d,CUFFT_INVERSE));
      checkCuda_accfft( cudaEventRecord(fft_stopEvent,0) );
      checkCuda_accfft( cudaEventSynchronize(fft_stopEvent) ); // wait until fft is executed
      checkCuda_accfft( cudaEventElapsedTime(&dummy_time, fft_startEvent, fft_stopEvent) );
      fft_time+=dummy_time/1000;

      plan->T_plan_1i->execute_gpu(plan->T_plan_1i,data_d,timings,1);
      /**************************************************************/
      /*******************  N0/P0 x N1 x N2 *************************/
      /**************************************************************/
      checkCuda_accfft( cudaEventRecord(fft_startEvent,0) );
      checkCuda_accfft(cufftExecZ2D(plan->iplan_0, (cufftDoubleComplex*)data_d,(cufftDoubleReal*)data_out_d));
      checkCuda_accfft( cudaEventRecord(fft_stopEvent,0) );
      checkCuda_accfft( cudaEventSynchronize(fft_stopEvent) ); // wait until fft is executed
      checkCuda_accfft( cudaEventElapsedTime(&dummy_time, fft_startEvent, fft_stopEvent) );
      fft_time+=dummy_time/1000;

    }
  }

  // 2D Decomposition
  if(plan->np[1]!=1){
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
      checkCuda_accfft( cudaEventRecord(fft_startEvent,0) );
      checkCuda_accfft (cufftExecD2Z(plan->fplan_0,(cufftDoubleReal*)data_d, (cufftDoubleComplex*)data_out_d));
      checkCuda_accfft( cudaEventRecord(fft_stopEvent,0) );
      checkCuda_accfft( cudaEventSynchronize(fft_stopEvent) ); // wait until fft is executed
      checkCuda_accfft( cudaEventElapsedTime(&dummy_time, fft_startEvent, fft_stopEvent) );
      fft_time+=dummy_time/1000;

      // Perform N0/P0 transpose


      plan->T_plan_1->execute_gpu(plan->T_plan_1,data_out_d,timings,2,osize_0[0],coords[0]);
      /**************************************************************/
      /*******************  N0/P0 x N1 x N2/P1 **********************/
      /**************************************************************/
      checkCuda_accfft( cudaEventRecord(fft_startEvent,0) );
      for (int i=0;i<osize_1[0];++i){
        checkCuda_accfft (cufftExecZ2Z(plan->fplan_1,(cufftDoubleComplex*)&data_out_d[2*i*osize_1[1]*osize_1[2]], (cufftDoubleComplex*)&data_out_d[2*i*osize_1[1]*osize_1[2]],CUFFT_FORWARD));
      }
      checkCuda_accfft( cudaEventRecord(fft_stopEvent,0) );
      checkCuda_accfft( cudaEventSynchronize(fft_stopEvent) ); // wait until fft is executed
      checkCuda_accfft( cudaEventElapsedTime(&dummy_time, fft_startEvent, fft_stopEvent) );
      fft_time+=dummy_time/1000;
      MPI_Barrier(plan->c_comm);

      plan->T_plan_2->execute_gpu(plan->T_plan_2,data_out_d,timings,2,1,coords[1]);
      /**************************************************************/
      /*******************  N0 x N1/P0 x N2/P1 **********************/
      /**************************************************************/
      checkCuda_accfft( cudaEventRecord(fft_startEvent,0) );
      checkCuda_accfft (cufftExecZ2Z(plan->fplan_2,(cufftDoubleComplex*)data_out_d, (cufftDoubleComplex*)data_out_d,CUFFT_FORWARD));
      checkCuda_accfft( cudaEventRecord(fft_stopEvent,0) );
      checkCuda_accfft( cudaEventSynchronize(fft_stopEvent) ); // wait until fft is executed
      checkCuda_accfft( cudaEventElapsedTime(&dummy_time, fft_startEvent, fft_stopEvent) );
      fft_time+=dummy_time/1000;
    }
    else if (direction==1){
      checkCuda_accfft( cudaEventRecord(fft_startEvent,0) );
      checkCuda_accfft (cufftExecZ2Z(plan->fplan_2,(cufftDoubleComplex*)data_d, (cufftDoubleComplex*)data_d,CUFFT_INVERSE));
      checkCuda_accfft( cudaEventRecord(fft_stopEvent,0) );
      checkCuda_accfft( cudaEventSynchronize(fft_stopEvent) ); // wait until fft is executed
      checkCuda_accfft( cudaEventElapsedTime(&dummy_time, fft_startEvent, fft_stopEvent) );
      fft_time+=dummy_time/1000;

      MPI_Barrier(plan->c_comm);


      plan->T_plan_2i->execute_gpu(plan->T_plan_2i,data_d,timings,1,1,coords[1]);
      /**************************************************************/
      /*******************  N0/P0 x N1 x N2/P1 **********************/
      /**************************************************************/
      checkCuda_accfft( cudaEventRecord(fft_startEvent,0) );
      for (int i=0;i<osize_1i[0];++i){
        checkCuda_accfft (cufftExecZ2Z(plan->fplan_1,(cufftDoubleComplex*)&data_d[2*i*NY*osize_1i[2]], (cufftDoubleComplex*)&data_d[2*i*NY*osize_1i[2]],CUFFT_INVERSE));
      }
      checkCuda_accfft( cudaEventRecord(fft_stopEvent,0) );
      checkCuda_accfft( cudaEventSynchronize(fft_stopEvent) ); // wait until fft is executed
      checkCuda_accfft( cudaEventElapsedTime(&dummy_time, fft_startEvent, fft_stopEvent) );
      fft_time+=dummy_time/1000;
      MPI_Barrier(plan->c_comm);



      plan->T_plan_1i->execute_gpu(plan->T_plan_1i,data_d,timings,1,osize_1i[0],coords[0]);
      MPI_Barrier(plan->c_comm);
      /**************************************************************/
      /*******************  N0/P0 x N1/P1 x N2 **********************/
      /**************************************************************/

      // IFFT in Z direction
      checkCuda_accfft( cudaEventRecord(fft_startEvent,0) );
      checkCuda_accfft (cufftExecZ2D(plan->iplan_0,(cufftDoubleComplex*)data_d,(cufftDoubleReal*)data_out_d));
      checkCuda_accfft( cudaEventRecord(fft_stopEvent,0) );
      checkCuda_accfft( cudaEventSynchronize(fft_stopEvent) ); // wait until fft is executed
      checkCuda_accfft( cudaEventElapsedTime(&dummy_time, fft_startEvent, fft_stopEvent) );
      fft_time+=dummy_time/1000;

    }
  }
  timings[4]=fft_time;
  if(timer==NULL){
    delete [] timings;
  }
  MPI_Barrier(plan->c_comm);

  return;
}
void accfft_execute_r2c_gpu(accfft_plan_gpu* plan, double * data,Complex * data_out, double * timer){
  accfft_execute_gpu(plan,-1,data,(double*)data_out,timer);

  return;
}
void accfft_execute_c2r_gpu(accfft_plan_gpu* plan, Complex * data,double * data_out, double * timer){
  accfft_execute_gpu(plan,1,(double*)data,data_out,timer);

  return;
}

int accfft_local_size_dft_c2c_gpu( int * n,int * isize, int * istart, int * osize, int *ostart,MPI_Comm c_comm){

  int osize_0[3]={0}, ostart_0[3]={0};
  int osize_1[3]={0}, ostart_1[3]={0};
  int osize_2[3]={0}, ostart_2[3]={0};
  //int osize_1i[3]={0}, ostart_1i[3]={0};
  //int osize_2i[3]={0}, ostart_2i[3]={0};

  int alloc_local;
  int alloc_max=0;//,n_tuples=n[2]*2;
  alloc_local=dfft_get_local_size_gpu(n[0],n[1],n[2],osize_0,ostart_0,c_comm);
  alloc_max=std::max(alloc_max, alloc_local);
  alloc_local=dfft_get_local_size_gpu(n[0],n[2],n[1],osize_1,ostart_1,c_comm);
  alloc_max=std::max(alloc_max, alloc_local);
  alloc_local=dfft_get_local_size_gpu(n[1],n[2],n[0],osize_2,ostart_2,c_comm);
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
  dfft_get_local_size_gpu(n[0],n[1],n[2],isize,istart,c_comm);

  osize[0]=osize_2[0];
  osize[1]=osize_2[1];
  osize[2]=osize_2[2];

  ostart[0]=ostart_2[0];
  ostart[1]=ostart_2[1];
  ostart[2]=ostart_2[2];

  return alloc_max;

}

accfft_plan_gpu*  accfft_plan_dft_3d_c2c_gpu(int * n, Complex * data_d, Complex * data_out_d, MPI_Comm c_comm,unsigned flags){
  accfft_plan_gpu *plan=new accfft_plan_gpu;
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

  // 1D Decomposition
  if (plan->np[1]==1){
    int NX=n[0],NY=n[1],NZ=n[2];


    int isize[3],osize[3],istart[3],ostart[3];
    int alloc_max=accfft_local_size_dft_c2c_gpu(n,isize,istart,osize,ostart,c_comm);
    plan->alloc_max=alloc_max;

    plan->Mem_mgr= new Mem_Mgr_gpu(NX,NY,(NZ)*2,c_comm);
    plan->T_plan_1= new T_Plan_gpu(NX,NY,(NZ)*2, plan->Mem_mgr,c_comm);
    plan->T_plan_1i= new T_Plan_gpu(NY,NX,NZ*2, plan->Mem_mgr,c_comm);

    plan->T_plan_1->alloc_local=alloc_max;
    plan->T_plan_1i->alloc_local=alloc_max;

    ptrdiff_t local_n0=plan->T_plan_1->local_n0;
    ptrdiff_t local_n1=plan->T_plan_1->local_n1;
    int N0=NX, N1=NY, N2=NZ;

    /* Forward Transform Plan. */
    int n[3] = {NX, NY, NZ};
    int f_inembed[2]={NY,(NZ)};
    int f_onembed[2]={NY,NZ};
    int idist=NY*(NZ);
    int odist=NY*(NZ);
    int istride=1;
    int ostride=1;
    int batch=plan->T_plan_1->local_n0;//NX;

    cufftResult_t cufft_error;
    if(batch!=0)
    {
      cufft_error=cufftPlanMany(&plan->fplan_0, 2, &n[1],
          f_inembed, istride, idist, // *inembed, istride, idist
          f_onembed, ostride, odist, // *onembed, ostride, odist
          CUFFT_Z2Z, batch);
      if(cufft_error!= CUFFT_SUCCESS)
      {
        fprintf(stderr, "CUFFT error: fplan creation failed %d \n",cufft_error); return NULL;
      }
      //cufftSetCompatibilityMode(fplan,CUFFT_COMPATIBILITY_FFTW_PADDING); if (cudaGetLastError() != cudaSuccess){fprintf(stderr, "Cuda error:Failed at fplan cuda compatibility\n"); return;}
    }
    int f_inembed2[1]={NX};
    int f_onembed2[1]={NX};
    int idist2=1;
    int odist2=1;
    int istride2=local_n1*(NZ);
    int ostride2=local_n1*(NZ);
    MPI_Barrier(c_comm);
    if(local_n1*NZ!=0){
    cufft_error=cufftPlanMany(&plan->fplan_1, 1, &n[0],
        f_inembed2, istride2, idist2, // *inembed, istride, idist
        f_onembed2, ostride2, odist2, // *onembed, ostride, odist
        CUFFT_Z2Z, local_n1*(NZ));
    if(cufft_error!= CUFFT_SUCCESS){
      fprintf(stderr, "CUFFT error: fplan2 creation failed %d\n",cufft_error); return NULL;
    }
    //cufftSetCompatibilityMode(fplan2,CUFFT_COMPATIBILITY_FFTW_PADDING); if (cudaGetLastError() != cudaSuccess){fprintf(stderr, "Cuda error:Failed at fplan2 cuda compatibility\n"); return;}
    }

    plan->T_plan_1->which_fast_method_gpu(plan->T_plan_1,(double*)data_out_d);
    plan->T_plan_1i->method=plan->T_plan_1->method;
    plan->T_plan_1i->kway=plan->T_plan_1->kway;

    // Make unused parts of plan NULL
    plan->T_plan_2=NULL;
    plan->T_plan_2i=NULL;
    plan->fplan_2=NULL;
    plan->iplan_2=NULL;
  }

  // 2D Decomposition
  if (plan->np[1]!=1){

    int *osize_0 =plan->osize_0, *ostart_0 =plan->ostart_0;
    int *osize_1 =plan->osize_1, *ostart_1 =plan->ostart_1;
    int *osize_2 =plan->osize_2, *ostart_2 =plan->ostart_2;
    int *osize_1i=plan->osize_1i,*ostart_1i=plan->ostart_1i;
    int *osize_2i=plan->osize_2i,*ostart_2i=plan->ostart_2i;

    int alloc_local;
    int alloc_max=0,n_tuples=n[2]*2;

    int isize[3],osize[3],istart[3],ostart[3];
    alloc_max=accfft_local_size_dft_c2c_gpu(n,isize,istart,osize,ostart,c_comm);
    plan->alloc_max=alloc_max;

    dfft_get_local_size_gpu(n[0],n[1],n[2],osize_0,ostart_0,c_comm);
    dfft_get_local_size_gpu(n[0],n[2],n[1],osize_1,ostart_1,c_comm);
    dfft_get_local_size_gpu(n[1],n[2],n[0],osize_2,ostart_2,c_comm);


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



    // the reaseon for n_tuples/2 is to avoid splitting of imag and real parts of complex numbers
    plan->Mem_mgr=  new Mem_Mgr_gpu(n[1],n[2],2,plan->row_comm,osize_0[0],alloc_max);
    plan->T_plan_1= new  T_Plan_gpu(n[1],n[2],2, plan->Mem_mgr, plan->row_comm,osize_0[0]);
    plan->T_plan_2= new  T_Plan_gpu(n[0],n[1],2*osize_2[2], plan->Mem_mgr, plan->col_comm);
    plan->T_plan_2i= new T_Plan_gpu(n[1],n[0],2*osize_2i[2], plan->Mem_mgr, plan->col_comm);
    plan->T_plan_1i= new T_Plan_gpu(n[2],n[1],2, plan->Mem_mgr, plan->row_comm,osize_1i[0]);

    plan->T_plan_1->alloc_local=plan->alloc_max;
    plan->T_plan_2->alloc_local=plan->alloc_max;
    plan->T_plan_2i->alloc_local=plan->alloc_max;
    plan->T_plan_1i->alloc_local=plan->alloc_max;


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
            CUFFT_Z2Z, batch);
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
            CUFFT_Z2Z, batch);
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
            CUFFT_Z2Z, batch);
        if(cufft_error!= CUFFT_SUCCESS)
        {
          fprintf(stderr, "CUFFT error: fplan_2 creation failed %d \n",cufft_error); return NULL;
        }
        //cufftSetCompatibilityMode(fplan,CUFFT_COMPATIBILITY_FFTW_PADDING); if (cudaGetLastError() != cudaSuccess){fprintf(stderr, "Cuda error:Failed at fplan cuda compatibility\n"); return;}
      }
    }

    plan->iplan_0=NULL;
    plan->iplan_1=NULL;
    plan->iplan_2=NULL;

    int coords[2],np[2],periods[2];
    MPI_Cart_get(c_comm,2,np,periods,coords);
    int transpose_method=0;
    int kway_method=0;
    if(coords[0]==0){
      plan->T_plan_1->which_fast_method_gpu(plan->T_plan_1,(double*)data_out_d);
      transpose_method=plan->T_plan_1->method;
      kway_method=plan->T_plan_1->kway;
    }
    checkCuda_accfft (cudaDeviceSynchronize());
    MPI_Barrier(plan->c_comm);


    MPI_Bcast(&transpose_method,1, MPI_INT,0, c_comm);
    MPI_Bcast(&kway_method,1, MPI_INT,0, c_comm);
    MPI_Barrier(c_comm);
    plan->T_plan_1->method=transpose_method;
    plan->T_plan_2->method= transpose_method;
    plan->T_plan_2i->method=transpose_method;
    plan->T_plan_1i->method=transpose_method;

    plan->T_plan_1->kway=kway_method;
    plan->T_plan_2->kway= kway_method;
    plan->T_plan_2i->kway=kway_method;
    plan->T_plan_1i->kway=kway_method;


  }
  return plan;
}

void accfft_execute_c2c_gpu(accfft_plan_gpu* plan, int direction,Complex * data_d, Complex * data_out_d, double * timer){

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

  // 1D Decomposition
  if(plan->np[1]==1){
    if(direction==-1){

      /**************************************************************/
      /*******************  N0/P0 x N1 x N2 *************************/
      /**************************************************************/
      checkCuda_accfft( cudaEventRecord(fft_startEvent,0) );
      checkCuda_accfft(cufftExecZ2Z(plan->fplan_0,(cufftDoubleComplex*)data_d, (cufftDoubleComplex*)data_out_d,CUFFT_FORWARD));
      checkCuda_accfft( cudaEventRecord(fft_stopEvent,0) );
      checkCuda_accfft( cudaEventSynchronize(fft_stopEvent) ); // wait until fft is executed
      checkCuda_accfft( cudaEventElapsedTime(&dummy_time, fft_startEvent, fft_stopEvent) );
      fft_time+=dummy_time/1000;



      MPI_Barrier(plan->c_comm);
      plan->T_plan_1->execute_gpu(plan->T_plan_1,(double*)data_out_d,timings,2);
      /**************************************************************/
      /*******************  N1 x N0/P0 x N2 *************************/
      /**************************************************************/
      checkCuda_accfft( cudaEventRecord(fft_startEvent,0) );
      checkCuda_accfft(cufftExecZ2Z(plan->fplan_1,(cufftDoubleComplex*)data_out_d, (cufftDoubleComplex*)data_out_d,CUFFT_FORWARD));
      checkCuda_accfft( cudaEventRecord(fft_stopEvent,0) );
      checkCuda_accfft( cudaEventSynchronize(fft_stopEvent) ); // wait until fft is executed
      checkCuda_accfft( cudaEventElapsedTime(&dummy_time, fft_startEvent, fft_stopEvent) );
      fft_time+=dummy_time/1000;
      MPI_Barrier(plan->c_comm);

    }

    if(direction==1){
      /* Now Perform the inverse transform  */
      checkCuda_accfft( cudaEventRecord(fft_startEvent,0) );
      checkCuda_accfft(cufftExecZ2Z(plan->fplan_1,(cufftDoubleComplex*)data_d,(cufftDoubleComplex*)data_d,CUFFT_INVERSE));
      checkCuda_accfft( cudaEventRecord(fft_stopEvent,0) );
      checkCuda_accfft( cudaEventSynchronize(fft_stopEvent) ); // wait until fft is executed
      checkCuda_accfft( cudaEventElapsedTime(&dummy_time, fft_startEvent, fft_stopEvent) );
      fft_time+=dummy_time/1000;

      plan->T_plan_1i->execute_gpu(plan->T_plan_1i,(double*)data_d,timings,1);
      /**************************************************************/
      /*******************  N0/P0 x N1 x N2 *************************/
      /**************************************************************/

      checkCuda_accfft( cudaEventRecord(fft_startEvent,0) );
      checkCuda_accfft(cufftExecZ2Z(plan->fplan_0,(cufftDoubleComplex*)data_d,(cufftDoubleComplex*)data_out_d,CUFFT_INVERSE));
      checkCuda_accfft( cudaEventRecord(fft_stopEvent,0) );
      checkCuda_accfft( cudaEventSynchronize(fft_stopEvent) ); // wait until fft is executed
      checkCuda_accfft( cudaEventElapsedTime(&dummy_time, fft_startEvent, fft_stopEvent) );
      fft_time+=dummy_time/1000;

    }
  }

  // 2D Decomposition
  if(plan->np[1]!=1){
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
      checkCuda_accfft( cudaEventRecord(fft_startEvent,0) );
      checkCuda_accfft(cufftExecZ2Z(plan->fplan_0,(cufftDoubleComplex*)data_d, (cufftDoubleComplex*)data_out_d,CUFFT_FORWARD));
      checkCuda_accfft( cudaEventRecord(fft_stopEvent,0) );
      checkCuda_accfft( cudaEventSynchronize(fft_stopEvent) ); // wait until fft is executed
      checkCuda_accfft( cudaEventElapsedTime(&dummy_time, fft_startEvent, fft_stopEvent) );
      fft_time+=dummy_time/1000;

      plan->T_plan_1->execute_gpu(plan->T_plan_1,(double*)data_out_d,timings,2,osize_0[0],coords[0]);
      checkCuda_accfft (cudaDeviceSynchronize());
      MPI_Barrier(plan->c_comm);
      /**************************************************************/
      /*******************  N0/P0 x N1 x N2/P1 **********************/
      /**************************************************************/
      checkCuda_accfft( cudaEventRecord(fft_startEvent,0) );
      for (int i=0;i<osize_1[0];++i){
        checkCuda_accfft(cufftExecZ2Z(plan->fplan_1,(cufftDoubleComplex*)&data_out_d[i*osize_1[1]*osize_1[2]], (cufftDoubleComplex*)&data_out_d[i*osize_1[1]*osize_1[2]],CUFFT_FORWARD));
      }
      checkCuda_accfft( cudaEventRecord(fft_stopEvent,0) );
      checkCuda_accfft( cudaEventSynchronize(fft_stopEvent) ); // wait until fft is executed
      checkCuda_accfft( cudaEventElapsedTime(&dummy_time, fft_startEvent, fft_stopEvent) );
      fft_time+=dummy_time/1000;
      MPI_Barrier(plan->c_comm);



      plan->T_plan_2->execute_gpu(plan->T_plan_2,(double*)data_out_d,timings,2,1,coords[1]);
      checkCuda_accfft (cudaDeviceSynchronize());
      MPI_Barrier(plan->c_comm);
      /**************************************************************/
      /*******************  N0 x N1/P0 x N2/P1 **********************/
      /**************************************************************/
      checkCuda_accfft( cudaEventRecord(fft_startEvent,0) );
      checkCuda_accfft(cufftExecZ2Z(plan->fplan_2,(cufftDoubleComplex*)data_out_d, (cufftDoubleComplex*)data_out_d,CUFFT_FORWARD));
      checkCuda_accfft( cudaEventRecord(fft_stopEvent,0) );
      checkCuda_accfft( cudaEventSynchronize(fft_stopEvent) ); // wait until fft is executed
      checkCuda_accfft( cudaEventElapsedTime(&dummy_time, fft_startEvent, fft_stopEvent) );
      fft_time+=dummy_time/1000;

    }
    else if (direction==1){
      checkCuda_accfft( cudaEventRecord(fft_startEvent,0) );
      checkCuda_accfft (cufftExecZ2Z(plan->fplan_2,(cufftDoubleComplex*)data_d, (cufftDoubleComplex*)data_d,CUFFT_INVERSE));
      checkCuda_accfft (cudaDeviceSynchronize());
      checkCuda_accfft( cudaEventRecord(fft_stopEvent,0) );
      checkCuda_accfft( cudaEventSynchronize(fft_stopEvent) ); // wait until fft is executed
      checkCuda_accfft( cudaEventElapsedTime(&dummy_time, fft_startEvent, fft_stopEvent) );
      fft_time+=dummy_time/1000;
      MPI_Barrier(plan->c_comm);


      plan->T_plan_2i->execute_gpu(plan->T_plan_2i,(double*)data_d,timings,1,1,coords[1]);
      checkCuda_accfft (cudaDeviceSynchronize());
      MPI_Barrier(plan->c_comm);
      /**************************************************************/
      /*******************  N0/P0 x N1 x N2/P1 **********************/
      /**************************************************************/
      checkCuda_accfft( cudaEventRecord(fft_startEvent,0) );
      for (int i=0;i<osize_1i[0];++i){
        checkCuda_accfft (cufftExecZ2Z(plan->fplan_1,(cufftDoubleComplex*)&data_d[i*NY*osize_1i[2]], (cufftDoubleComplex*)&data_d[i*NY*osize_1i[2]],CUFFT_INVERSE));
      }
      checkCuda_accfft( cudaEventRecord(fft_stopEvent,0) );
      checkCuda_accfft( cudaEventSynchronize(fft_stopEvent) ); // wait until fft is executed
      checkCuda_accfft( cudaEventElapsedTime(&dummy_time, fft_startEvent, fft_stopEvent) );
      fft_time+=dummy_time/1000;
      MPI_Barrier(plan->c_comm);

      plan->T_plan_1i->execute_gpu(plan->T_plan_1i,(double*)data_d,timings,1,osize_1i[0],coords[0]);
      checkCuda_accfft (cudaDeviceSynchronize());
      MPI_Barrier(plan->c_comm);
      /**************************************************************/
      /*******************  N0/P0 x N1/P1 x N2 **********************/
      /**************************************************************/

      // IFFT in Z direction
      checkCuda_accfft( cudaEventRecord(fft_startEvent,0) );
      checkCuda_accfft (cufftExecZ2Z(plan->fplan_0,(cufftDoubleComplex*)data_d,(cufftDoubleComplex*)data_out_d,CUFFT_INVERSE));
      checkCuda_accfft( cudaEventRecord(fft_stopEvent,0) );
      checkCuda_accfft( cudaEventSynchronize(fft_stopEvent) ); // wait until fft is executed
      checkCuda_accfft( cudaEventElapsedTime(&dummy_time, fft_startEvent, fft_stopEvent) );
      fft_time+=dummy_time/1000;

    }

  }
  timings[4]=fft_time;
  if(timer==NULL){
    delete [] timings;
  }
  MPI_Barrier(plan->c_comm);

  return;
}


void accfft_destroy_plan(accfft_plan_gpu * plan){
  return (accfft_destroy_plan_gpu(plan));
}
void accfft_destroy_plan_gpu(accfft_plan_gpu * plan){

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
