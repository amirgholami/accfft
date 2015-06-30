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
#include <iostream>
#include <transpose_cuda.h>

#include <stdio.h>
#include <bitset>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <assert.h>
#define PCOUT if(procid==0) std::cout
#include "parUtils.h"
#include <math.h>
#include <cmath>
#include <stdlib.h>
#include <functional>
#include <cmath>
#define VERBOSE 0


int log2(int a){
return log(a)/log(2);
}
static bool IsPowerOfTwo(ulong x)
{
      return (x & (x - 1)) == 0;
}

static int intpow(int a,int b){
  return ((int)std::pow((double)a,b));
}
#ifdef ENABLE_GPU
Mem_Mgr_gpu::Mem_Mgr_gpu(int N0, int N1,int tuples, MPI_Comm Comm, int howmany, int specified_alloc_local){

  N[0]=N0;
  N[1]=N1;
  n_tuples=tuples;
  int procid, nprocs;
  MPI_Comm_rank(Comm, &procid);
  MPI_Comm_size(Comm,&nprocs);


  // Determine local_n0/n1 of each processor
  if(specified_alloc_local==0){
    {
      ptrdiff_t * local_n0_proc=(ptrdiff_t*) malloc(sizeof(ptrdiff_t)*nprocs);
      ptrdiff_t * local_n1_proc=(ptrdiff_t*) malloc(sizeof(ptrdiff_t)*nprocs);
#pragma omp parallel for
      for (int proc=0;proc<nprocs;++proc){
        local_n0_proc[proc]=ceil(N[0]/(double)nprocs);
        local_n1_proc[proc]=ceil(N[1]/(double)nprocs);


        if((N[0]-local_n0_proc[proc]*proc)<local_n0_proc[proc]) {local_n0_proc[proc]=N[0]-local_n0_proc[proc]*proc; local_n0_proc[proc]*=(int) local_n0_proc[proc]>0;}
        if((N[1]-local_n1_proc[proc]*proc)<local_n1_proc[proc]) {local_n1_proc[proc]=N[1]-local_n1_proc[proc]*proc;local_n1_proc[proc]*=(int) local_n1_proc[proc]>0;}


      }

      local_n0=local_n0_proc[procid];
      local_n1=local_n1_proc[procid];
      free(local_n0_proc);
      free(local_n1_proc);
    }


    // Determine alloc local based on maximum size of input and output distribution
    alloc_local=local_n0*N[1]*n_tuples*sizeof(double);
    if(alloc_local<local_n1*N[0]*n_tuples*sizeof(double))
      alloc_local=local_n1*N[0]*n_tuples*sizeof(double);

    alloc_local*=howmany;
  }
  else{
    alloc_local=specified_alloc_local;
  }
  if( alloc_local<=1.05*intpow(2,30) )
    PINNED=1;
  else
    PINNED=0;
#ifdef ENABLE_GPU
 // PCOUT<<"ENABLE GPU"<<std::endl;
  if(PINNED==1){
    cudaError_t cuda_err1, cuda_err2;
    double pinned_time=-MPI_Wtime();
    buffer=NULL; buffer_2=NULL;
    cuda_err1=cudaMallocHost((void**)&buffer,alloc_local);
    cuda_err2=cudaMallocHost((void**)&buffer_2,alloc_local);
    if(cuda_err1!=cudaSuccess || cuda_err2!=cudaSuccess){
      std::cout<<"!!!!!!!!!! Failed to cudaMallocHost in MemMgr"<<std::endl;
    }
    pinned_time+=MPI_Wtime();
   // PCOUT<<"PINNED Alloc time= "<<pinned_time<<std::endl;
  }
  else{
    posix_memalign((void **)&buffer_2,64, alloc_local);
    posix_memalign((void **)&buffer,64, alloc_local);
  }
  cudaMalloc((void **)&buffer_d, alloc_local);
#else
  posix_memalign((void **)&buffer,64, alloc_local);
  posix_memalign((void **)&buffer_2,64, alloc_local);
#endif
  memset( buffer,0, alloc_local );
  memset( buffer_2,0, alloc_local );

}
Mem_Mgr_gpu::~Mem_Mgr_gpu(){

#ifdef ENABLE_GPU
    cudaError_t cuda_err1=cudaSuccess, cuda_err2=cudaSuccess,cuda_err3=cudaSuccess;
  if(PINNED==1){
    if(buffer!=NULL)  cuda_err1=cudaFreeHost(buffer);
    if(buffer!=NULL)  cuda_err2=cudaFreeHost(buffer_2);
    if(cuda_err1!=cudaSuccess || cuda_err2!=cudaSuccess){
      std::cout<<"!!!!!!!!!! Failed to cudaFreeHost in MemMgr; err1= "<<cuda_err1<<" err2= "<<cuda_err2<<std::endl;
    }
  }
  else{
    free(buffer);
    free(buffer_2);
  }
  cuda_err3=cudaFree(buffer_d);
    if(cuda_err3!=cudaSuccess){
      std::cout<<"!!!!!!!!!! Failed to cudaFree in MemMgr; err3= "<<cuda_err3<<std::endl;
    }
#else
  free(buffer);
  free(buffer_2);
#endif
}
T_Plan_gpu::T_Plan_gpu(int N0, int N1,int tuples, Mem_Mgr_gpu * Mem_mgr, MPI_Comm Comm, int howmany){

  N[0]=N0;
  N[1]=N1;
  n_tuples=tuples;
  MPI_Comm_rank(Comm, &procid);
  MPI_Comm_size(Comm,&nprocs);

  local_n0_proc=(ptrdiff_t*) malloc(sizeof(ptrdiff_t)*nprocs);
  local_n1_proc=(ptrdiff_t*) malloc(sizeof(ptrdiff_t)*nprocs);
  local_0_start_proc=(ptrdiff_t*) malloc(sizeof(ptrdiff_t)*nprocs);
  local_1_start_proc=(ptrdiff_t*) malloc(sizeof(ptrdiff_t)*nprocs);

  // Determine local_n0/n1 of each processor

  local_0_start_proc[0]=0;local_1_start_proc[0]=0;
  for (int proc=0;proc<nprocs;++proc){
    local_n0_proc[proc]=ceil(N[0]/(double)nprocs);
    local_n1_proc[proc]=ceil(N[1]/(double)nprocs);


    if((N[0]-local_n0_proc[proc]*proc)<local_n0_proc[proc]) {local_n0_proc[proc]=N[0]-local_n0_proc[proc]*proc; local_n0_proc[proc]*=(int) local_n0_proc[proc]>0;}
    if((N[1]-local_n1_proc[proc]*proc)<local_n1_proc[proc]) {local_n1_proc[proc]=N[1]-local_n1_proc[proc]*proc;local_n1_proc[proc]*=(int) local_n1_proc[proc]>0;}

    if(proc!=0){
      local_0_start_proc[proc]=local_0_start_proc[proc-1]+local_n0_proc[proc-1];
      local_1_start_proc[proc]=local_1_start_proc[proc-1]+local_n1_proc[proc-1];
    }

  }

  local_n0=local_n0_proc[procid];
  local_n1=local_n1_proc[procid];
  local_0_start=local_0_start_proc[procid];
  local_1_start=local_1_start_proc[procid];


  alloc_local=Mem_mgr->alloc_local;
  // Determine effective processors taking part in each transpose phase
  nprocs_0=0; nprocs_1=0;
  for (int proc=0;proc<nprocs;++proc){
    if(local_n0_proc[proc]!=0)
      nprocs_0+=1;
    if(local_n1_proc[proc]!=0)
      nprocs_1+=1;
  }



  // Set send recv counts for communication part
  scount_proc=(int*) malloc(sizeof(int)*nprocs);
  rcount_proc=(int*) malloc(sizeof(int)*nprocs);
  soffset_proc=(int*) malloc(sizeof(int)*nprocs);
  roffset_proc=(int*) malloc(sizeof(int)*nprocs);

  scount_proc_f=(int*) malloc(sizeof(int)*nprocs);
  rcount_proc_f=(int*) malloc(sizeof(int)*nprocs);
  soffset_proc_f=(int*) malloc(sizeof(int)*nprocs);
  roffset_proc_f=(int*) malloc(sizeof(int)*nprocs);

  scount_proc_w=(int*) malloc(sizeof(int)*nprocs);
  rcount_proc_w=(int*) malloc(sizeof(int)*nprocs);
  soffset_proc_w=(int*) malloc(sizeof(int)*nprocs);
  roffset_proc_w=(int*) malloc(sizeof(int)*nprocs);

  //scount_proc_v8=(int*) malloc(sizeof(int)*nprocs);
  //rcount_proc_v8=(int*) malloc(sizeof(int)*nprocs);
  //soffset_proc_v8=(int*) malloc(sizeof(int)*nprocs);
  //roffset_proc_v8=(int*) malloc(sizeof(int)*nprocs);
  //memset(scount_proc_v8,0,sizeof(int)*nprocs);
  //memset(rcount_proc_v8,0,sizeof(int)*nprocs);
  //memset(soffset_proc_v8,0,sizeof(int)*nprocs);
  //memset(roffset_proc_v8,0,sizeof(int)*nprocs);

  last_recv_count=0; // Will store the n_tuples of the last received data. In general ~=n_tuples
  if(nprocs_1>nprocs_0)
    for (int proc=0;proc<nprocs;++proc){

      scount_proc[proc]=local_n1_proc[proc]*local_n0*n_tuples;

      if(scount_proc[proc]!=0)
        rcount_proc[proc]=local_n1_proc[proc]*local_n0_proc[proc]*n_tuples;//scount_proc[proc]; 
      else
        rcount_proc[proc]=local_n1*local_n0_proc[proc]*n_tuples;//local_n0_proc[proc]*n_tuples; //

      soffset_proc[proc]=0;
      roffset_proc[proc]=0;
      if(proc==0){
        soffset_proc[proc]=0;
        roffset_proc[proc]=0;
      }
      else{
        soffset_proc[proc]=soffset_proc[proc-1]+scount_proc[proc-1];
        roffset_proc[proc]=roffset_proc[proc-1]+rcount_proc[proc-1];
      }

      if(proc>=nprocs_1){ // in case you have requested too many processes
        rcount_proc[proc]=0;
        scount_proc[proc]=0;
      }
      if(scount_proc[proc]==0) soffset_proc[proc]=0;//local_n0*N[1]*n_tuples-1;

      if(proc>=nprocs_0){
        rcount_proc[proc]=0;
        roffset_proc[proc]=0;
      }
      if(rcount_proc[proc]!=0)
        last_recv_count=rcount_proc[proc];
      if(local_n1_proc[proc]!=0)
        last_local_n1=local_n1_proc[proc];
    }
  else if(nprocs_1<=nprocs_0)
    for (int proc=0;proc<nprocs;++proc){

      scount_proc[proc]=local_n1_proc[proc]*local_n0*n_tuples;
      rcount_proc[proc]=local_n1*local_n0_proc[proc]*n_tuples;//scount_proc[proc];



      soffset_proc[proc]=0;
      roffset_proc[proc]=0;
      if(proc==0){
        soffset_proc[proc]=0;
        roffset_proc[proc]=0;
      }
      else{
        soffset_proc[proc]=soffset_proc[proc-1]+scount_proc[proc-1];
        roffset_proc[proc]=roffset_proc[proc-1]+rcount_proc[proc-1];
      }

      if(proc>=nprocs_0){ // in case you have requested too many processes
        rcount_proc[proc]=0;
        scount_proc[proc]=0;
        roffset_proc[proc]=0;
        soffset_proc[proc]=0;
      }

      if(scount_proc[proc]==0) soffset_proc[proc]=0;//local_n0*N[1]*n_tuples-1;

      if(proc>=nprocs_1){
        scount_proc[proc]=0;
        soffset_proc[proc]=0;
      }
      if(rcount_proc[proc]!=0)
        last_recv_count=rcount_proc[proc];

      if(local_n1_proc[proc]!=0)
        last_local_n1=local_n1_proc[proc];
    }


  is_evenly_distributed=0; // use alltoallv
  if((local_n0*nprocs_0-N[0])==0 && (local_n1*nprocs_1-N[1])==0 && nprocs_0==nprocs_1 && nprocs_0==nprocs){
    is_evenly_distributed=1; // use alltoall
  }

  method=5;  //Default Transpose method
  kway=8;  // for transpose_v7
  kway_async=true;
  stype=new MPI_Datatype[nprocs];
  rtype=new MPI_Datatype[nprocs];
  //stype_v8=new MPI_Datatype[nprocs];
  //rtype_v8_=new MPI_Datatype[nprocs];
  //rtype_v8=new MPI_Datatype[nprocs];

  for (int i=0;i<nprocs;i++){
    MPI_Type_vector(howmany,scount_proc[i],local_n0*N[1]*n_tuples, MPI_DOUBLE, &stype[i]);
    MPI_Type_vector(howmany,rcount_proc[i],local_n1*N[0]*n_tuples, MPI_DOUBLE, &rtype[i]);

    MPI_Type_commit(&stype[i]);
    MPI_Type_commit(&rtype[i]);

    soffset_proc_w[i]=soffset_proc[i]*8; //SNAFU in bytes
    roffset_proc_w[i]=roffset_proc[i]*8;
    scount_proc_w[i]=1;
    rcount_proc_w[i]=1;

    soffset_proc_f[i]=soffset_proc[i]*howmany; //SNAFU in bytes
    roffset_proc_f[i]=roffset_proc[i]*howmany;
    scount_proc_f[i]= scount_proc[i]*howmany;
    rcount_proc_f[i]= rcount_proc[i]*howmany;


    /*
    if(local_n0!=0){
      soffset_proc_v8[i]=soffset_proc[i]/local_n0;
      scount_proc_v8[i]=scount_proc[i]/local_n0;
    }
    if(local_n1!=0){
      roffset_proc_v8[i]=roffset_proc[i]/local_n1;
      rcount_proc_v8[i]=rcount_proc[i]/local_n1;
    }

    MPI_Type_vector(local_n0,scount_proc_v8[i],N[1]*n_tuples, MPI_DOUBLE, &stype_v8[i]);
    MPI_Type_vector(local_n1,1*n_tuples,N[0]*n_tuples, MPI_DOUBLE, &rtype_v8_[i]);
    MPI_Type_hvector(rcount_proc_v8[i],1,n_tuples*sizeof(double),rtype_v8_[i], &rtype_v8[i]);

    MPI_Type_commit(&stype_v8[i]);
    MPI_Type_commit(&rtype_v8[i]);
    */

  }

  comm=Comm; // MPI Communicator
  buffer=Mem_mgr->buffer;
  buffer_2=Mem_mgr->buffer_2;
  buffer_d=Mem_mgr->buffer_d;
  //data_cpu=Mem_mgr->data_cpu;

}
T_Plan_gpu::~T_Plan_gpu(){
  free(local_n0_proc);
  free(local_n1_proc);
  free(local_0_start_proc);
  free(local_1_start_proc);
  free(scount_proc);
  free(rcount_proc);
  free(soffset_proc);
  free(roffset_proc);

  free(scount_proc_w);
  free(rcount_proc_w);
  free(soffset_proc_w);
  free(roffset_proc_w);

  free(scount_proc_f);
  free(rcount_proc_f);
  free(soffset_proc_f);
  free(roffset_proc_f);

  //free(scount_proc_v8);
  //free(rcount_proc_v8);
  //free(soffset_proc_v8);
  //free(roffset_proc_v8);
  delete [] stype;
  delete [] rtype;
  //delete [] stype_v8;
  //delete [] rtype_v8;
  //delete [] rtype_v8_;
}
void T_Plan_gpu::which_method_gpu(T_Plan_gpu* T_plan,double* data_d){

  double dummy[4]={0};
  double * time= (double*) malloc(sizeof(double)*(4*(int)log2(nprocs)+4));
  double * g_time= (double*) malloc(sizeof(double)*(4*(int)log2(nprocs)+4));
  for (int i=0;i<4*(int)log2(nprocs)+4;i++)
    time[i]=1000;

  transpose_cuda_v5(T_plan,(double*)data_d,dummy);  // Warmup
  time[0]=-MPI_Wtime();
  transpose_cuda_v5(T_plan,(double*)data_d,dummy);
  time[0]+=MPI_Wtime();

  transpose_cuda_v6(T_plan,(double*)data_d,dummy);  // Warmup
  time[1]=-MPI_Wtime();
  transpose_cuda_v6(T_plan,(double*)data_d,dummy);
  time[1]+=MPI_Wtime();

  if(IsPowerOfTwo(nprocs) && nprocs>511){
    kway_async=true;
#ifndef TORUS_TOPOL
    for (int i=0;i<(int)log2(nprocs)-4;i++){
      kway=nprocs/intpow(2,i);
      MPI_Barrier(T_plan->comm);
      transpose_cuda_v7(T_plan,(double*)data_d,dummy,kway);  // Warmup
      time[2+i]=-MPI_Wtime();
      transpose_cuda_v7(T_plan,(double*)data_d,dummy,kway);
      time[2+i]+=MPI_Wtime();
    }
#endif

    kway_async=false;
#ifdef TORUS_TOPOL
    for (int i=0;i<(int)log2(nprocs)-4;i++){
      kway=nprocs/intpow(2,i);
      MPI_Barrier(T_plan->comm);
      transpose_cuda_v7(T_plan,(double*)data_d,dummy,kway);  // Warmup
      time[2+(int)log2(nprocs)+i]=-MPI_Wtime();
      transpose_cuda_v7(T_plan,(double*)data_d,dummy,kway);
      time[2+(int)log2(nprocs)+i]+=MPI_Wtime();
    }
#endif

#ifndef TORUS_TOPOL
    kway_async=true;
    for (int i=0;i<(int)log2(nprocs)-4;i++){
      kway=nprocs/intpow(2,i);
      MPI_Barrier(T_plan->comm);
      transpose_cuda_v7_2(T_plan,(double*)data_d,dummy,kway);  // Warmup
      time[2+2*(int)log2(nprocs)+i]=-MPI_Wtime();
      transpose_cuda_v7_2(T_plan,(double*)data_d,dummy,kway);
      time[2+2*(int)log2(nprocs)+i]+=MPI_Wtime();
    }
#endif

#ifdef TORUS_TOPOL
    kway_async=false;
    for (int i=0;i<(int)log2(nprocs)-4;i++){
      kway=nprocs/intpow(2,i);
      MPI_Barrier(T_plan->comm);
      transpose_cuda_v7_2(T_plan,(double*)data_d,dummy,kway);  // Warmup
      time[2+3*(int)log2(nprocs)+i]=-MPI_Wtime();
      transpose_cuda_v7_2(T_plan,(double*)data_d,dummy,kway);
      time[2+3*(int)log2(nprocs)+i]+=MPI_Wtime();
    }
#endif
  }

  transpose_cuda_v5_2(T_plan,(double*)data_d,dummy);  // Warmup
  time[4*(int)log2(nprocs)+2]=-MPI_Wtime();
  transpose_cuda_v5_2(T_plan,(double*)data_d,dummy);
  time[4*(int)log2(nprocs)+2]+=MPI_Wtime();

  transpose_cuda_v5_3(T_plan,(double*)data_d,dummy);  // Warmup
  time[4*(int)log2(nprocs)+3]=-MPI_Wtime();
  transpose_cuda_v5_3(T_plan,(double*)data_d,dummy);
  time[4*(int)log2(nprocs)+3]+=MPI_Wtime();

  MPI_Allreduce(time,g_time,(4*(int)log2(nprocs)+4),MPI_DOUBLE,MPI_MAX, T_plan->comm);

  if(VERBOSE>=1)
  if(T_plan->procid==0){
    for(int i=0;i<4*(int)log2(nprocs)+4;++i)
      std::cout<<" time["<<i<<"]= "<<g_time[i]<<" , ";
    std::cout<<'\n';
  }

  double smallest=1000;
  for (int i=0;i<4*(int)log2(nprocs)+4;i++)
    smallest=std::min(smallest,g_time[i]);

  if(g_time[0]==smallest){
    T_plan->method=5;
  }
  else if(g_time[1]==smallest){
    T_plan->method=6;
  }
  else if(g_time[4*(int)log2(nprocs)+2]==smallest){
    T_plan->method=52;
  }
  else if(g_time[4*(int)log2(nprocs)+3]==smallest){
    T_plan->method=53;
  }
  else{
    for (int i=0;i<(int)log2(nprocs);i++)
      if(g_time[2+i]==smallest){
        T_plan->method=7;
        T_plan->kway=nprocs/intpow(2,i);
        T_plan->kway_async=true;
        break;
      }
    for (int i=0;i<(int)log2(nprocs);i++)
      if(g_time[2+(int)log2(nprocs)+i]==smallest){
        T_plan->method=7;
        T_plan->kway=nprocs/intpow(2,i);
        T_plan->kway_async=false;
        break;
      }

    for (int i=0;i<(int)log2(nprocs);i++)
      if(g_time[2+2*(int)log2(nprocs)+i]==smallest){
        T_plan->method=7;
        T_plan->kway=nprocs/intpow(2,i);
        T_plan->kway_async=true;
        break;
      }

    for (int i=0;i<(int)log2(nprocs);i++)
      if(g_time[2+3*(int)log2(nprocs)+i]==smallest){
        T_plan->method=7;
        T_plan->kway=nprocs/intpow(2,i);
        T_plan->kway_async=false;
        break;
      }
  }

  if(VERBOSE>=1){
  PCOUT<<"smallest= "<<smallest<<std::endl;
  PCOUT<<"Using transpose v"<<method<<" kway= "<<T_plan->kway<<" kway_async="<<T_plan->kway_async<<std::endl;
  }
  free(time);
  free(g_time);
  MPI_Barrier(T_plan->comm);

  return;
}
void T_Plan_gpu::execute_gpu(T_Plan_gpu* T_plan,double* data_d,double *timings, unsigned flags, int howmany, int tag){


  if(howmany==1){
    if(method==1)
      fast_transpose_cuda_v1(T_plan,(double*)data_d,timings,flags,howmany, tag);
    if(method==12)
      fast_transpose_cuda_v1_2(T_plan,(double*)data_d,timings,flags,howmany, tag);
    if(method==13)
      fast_transpose_cuda_v1_3(T_plan,(double*)data_d,timings,flags,howmany, tag);
    if(method==2)
      fast_transpose_cuda_v2(T_plan,(double*)data_d,timings,flags,howmany, tag);
    if(method==3)
      fast_transpose_cuda_v3(T_plan,(double*)data_d,timings,kway,flags,howmany, tag);
    if(method==32)
      fast_transpose_cuda_v3_2(T_plan,(double*)data_d,timings,kway,flags,howmany, tag);
  }
  else
  {
    if(method==1)
      fast_transpose_cuda_v1_h(T_plan,(double*)data_d,timings,flags,howmany, tag);
    if(method==12)
      fast_transpose_cuda_v1_2_h(T_plan,(double*)data_d,timings,flags,howmany, tag);
    if(method==13)
      fast_transpose_cuda_v1_3_h(T_plan,(double*)data_d,timings,flags,howmany, tag);
    if(method==2)
      fast_transpose_cuda_v2_h(T_plan,(double*)data_d,timings,flags,howmany, tag);
    if(method==3 || method==32)
      fast_transpose_cuda_v3_h(T_plan,(double*)data_d,timings,kway,flags,howmany, tag);
  }
  if(method==5)
    transpose_cuda_v5(T_plan,(double*)data_d,timings,flags,howmany, tag);
  if(method==52)
    transpose_cuda_v5_2(T_plan,(double*)data_d,timings,flags,howmany, tag);
  if(method==53)
    transpose_cuda_v5_3(T_plan,(double*)data_d,timings,flags,howmany, tag);
  if(method==6)
    transpose_cuda_v6(T_plan,(double*)data_d,timings,flags,howmany, tag);
  if(method==7)
    transpose_cuda_v7(T_plan,(double*)data_d,timings,kway,flags,howmany, tag);
  if(method==72)
    transpose_cuda_v7_2(T_plan,(double*)data_d,timings,kway,flags,howmany, tag);


  return;
}

void T_Plan_gpu::which_fast_method_gpu(T_Plan_gpu* T_plan,double* data_d){

  double dummy[4]={0};
  double * time= (double*) malloc(sizeof(double)*(4*(int)log2(nprocs)+4));
  double * g_time= (double*) malloc(sizeof(double)*(4*(int)log2(nprocs)+4));
  for (int i=0;i<4*(int)log2(nprocs)+4;i++)
    time[i]=1000;

  fast_transpose_cuda_v1(T_plan,(double*)data_d,dummy,2);  // Warmup
  time[0]=-MPI_Wtime();
  fast_transpose_cuda_v1(T_plan,(double*)data_d,dummy,2);
  time[0]+=MPI_Wtime();

  fast_transpose_cuda_v2(T_plan,(double*)data_d,dummy,2);  // Warmup
  time[1]=-MPI_Wtime();
  fast_transpose_cuda_v2(T_plan,(double*)data_d,dummy,2);
  time[1]+=MPI_Wtime();

  if(IsPowerOfTwo(nprocs) && nprocs>511){
    kway_async=true;
#ifndef TORUS_TOPOL
    for (int i=0;i<(int)log2(nprocs)-4;i++){
      kway=nprocs/intpow(2,i);
      MPI_Barrier(T_plan->comm);
      fast_transpose_cuda_v3(T_plan,(double*)data_d,dummy,kway,2);  // Warmup
      time[2+i]=-MPI_Wtime();
      fast_transpose_cuda_v3(T_plan,(double*)data_d,dummy,kway,2);
      time[2+i]+=MPI_Wtime();
    }
#endif

    kway_async=false;
#ifdef TORUS_TOPOL
    for (int i=0;i<(int)log2(nprocs)-4;i++){
      kway=nprocs/intpow(2,i);
      MPI_Barrier(T_plan->comm);
      fast_transpose_cuda_v3(T_plan,(double*)data_d,dummy,kway,2);  // Warmup
      time[2+(int)log2(nprocs)+i]=-MPI_Wtime();
      fast_transpose_cuda_v3(T_plan,(double*)data_d,dummy,kway,2);
      time[2+(int)log2(nprocs)+i]+=MPI_Wtime();
    }
#endif

#ifndef TORUS_TOPOL
    kway_async=true;
    for (int i=0;i<(int)log2(nprocs)-4;i++){
      kway=nprocs/intpow(2,i);
      MPI_Barrier(T_plan->comm);
      fast_transpose_cuda_v3_2(T_plan,(double*)data_d,dummy,kway,2);  // Warmup
      time[2+2*(int)log2(nprocs)+i]=-MPI_Wtime();
      fast_transpose_cuda_v3_2(T_plan,(double*)data_d,dummy,kway,2);
      time[2+2*(int)log2(nprocs)+i]+=MPI_Wtime();
    }
#endif

#ifdef TORUS_TOPOL
    kway_async=false;
    for (int i=0;i<(int)log2(nprocs)-4;i++){
      kway=nprocs/intpow(2,i);
      MPI_Barrier(T_plan->comm);
      fast_transpose_cuda_v3_2(T_plan,(double*)data_d,dummy,kway,2);  // Warmup
      time[2+3*(int)log2(nprocs)+i]=-MPI_Wtime();
      fast_transpose_cuda_v3_2(T_plan,(double*)data_d,dummy,kway,2);
      time[2+3*(int)log2(nprocs)+i]+=MPI_Wtime();
    }
#endif
  }

  fast_transpose_cuda_v1_2(T_plan,(double*)data_d,dummy,2);  // Warmup
  time[4*(int)log2(nprocs)+2]=-MPI_Wtime();
  fast_transpose_cuda_v1_2(T_plan,(double*)data_d,dummy,2);
  time[4*(int)log2(nprocs)+2]+=MPI_Wtime();

  fast_transpose_cuda_v1_3(T_plan,(double*)data_d,dummy,2);  // Warmup
  time[4*(int)log2(nprocs)+3]=-MPI_Wtime();
  fast_transpose_cuda_v1_3(T_plan,(double*)data_d,dummy,2);
  time[4*(int)log2(nprocs)+3]+=MPI_Wtime();

  MPI_Allreduce(time,g_time,(4*(int)log2(nprocs)+4),MPI_DOUBLE,MPI_MAX, T_plan->comm);
  if(VERBOSE>=1)
  if(T_plan->procid==0){
    for(int i=0;i<4*(int)log2(nprocs)+4;++i)
      std::cout<<" time["<<i<<"]= "<<g_time[i]<<" , ";
    std::cout<<'\n';
  }

  double smallest=1000;
  for (int i=0;i<4*(int)log2(nprocs)+4;i++)
    smallest=std::min(smallest,g_time[i]);

  if(g_time[0]==smallest){
    T_plan->method=1;
  }
  else if(g_time[1]==smallest){
    T_plan->method=2;
  }
  else if(g_time[4*(int)log2(nprocs)+2]==smallest){
    T_plan->method=12;
  }
  else if(g_time[4*(int)log2(nprocs)+3]==smallest){
    T_plan->method=13;
  }
  else{
    for (int i=0;i<(int)log2(nprocs);i++)
      if(g_time[2+i]==smallest){
        T_plan->method=3;
        T_plan->kway=nprocs/intpow(2,i);
        T_plan->kway_async=true;
        break;
      }
    for (int i=0;i<(int)log2(nprocs);i++)
      if(g_time[2+(int)log2(nprocs)+i]==smallest){
        T_plan->method=3;
        T_plan->kway=nprocs/intpow(2,i);
        T_plan->kway_async=false;
        break;
      }

    for (int i=0;i<(int)log2(nprocs);i++)
      if(g_time[2+2*(int)log2(nprocs)+i]==smallest){
        T_plan->method=32;
        T_plan->kway=nprocs/intpow(2,i);
        T_plan->kway_async=true;
        break;
      }

    for (int i=0;i<(int)log2(nprocs);i++)
      if(g_time[2+3*(int)log2(nprocs)+i]==smallest){
        T_plan->method=32;
        T_plan->kway=nprocs/intpow(2,i);
        T_plan->kway_async=false;
        break;
      }
  }

  if(VERBOSE>=1){
  PCOUT<<"smallest= "<<smallest<<std::endl;
  PCOUT<<"Using transpose v"<<method<<" kway= "<<T_plan->kway<<" kway_async="<<T_plan->kway_async<<std::endl;
  }
  free(time);
  free(g_time);
  MPI_Barrier(T_plan->comm);

  return;
}
#endif




void fast_transpose_cuda_v1_h(T_Plan_gpu* T_plan, double * data, double *timings, unsigned flags, int howmany, int tag ){

  std::bitset<8> Flags(flags); // 1 Transposed in, 2 Transposed out
  if(Flags[1]==1 && Flags[0]==0 && T_plan->nprocs==1){ // If nprocs==1 and Flags==Transposed_Out return
    MPI_Barrier(T_plan->comm);
    return;
  }
  if(Flags[0]==1){ // If Flags==Transposed_In This function can not handle it, call other versions
    transpose_cuda_v5(T_plan,(double*)data,timings,flags,howmany,tag);
    MPI_Barrier(T_plan->comm);
    return;
  }
  timings[0]-=MPI_Wtime();
  int nprocs, procid;
  int nprocs_0, nprocs_1;
  nprocs=T_plan->nprocs;
  procid=T_plan->procid;
  nprocs_0=T_plan->nprocs_0;
  nprocs_1=T_plan->nprocs_1;
  ptrdiff_t  *N=T_plan->N;
  ptrdiff_t local_n0=T_plan->local_n0;
  ptrdiff_t local_n1=T_plan->local_n1;
  ptrdiff_t n_tuples=T_plan->n_tuples;

  double * data_cpu=T_plan->buffer;  //pinned
  double * send_recv_cpu = T_plan->buffer_2;//pinned
  double * send_recv_d = T_plan->buffer_d;

  int idist=N[1]*local_n0*n_tuples;
  int odist=N[0]*local_n1*n_tuples;

  double comm_time=0*MPI_Wtime(), shuffle_time=0*MPI_Wtime(), reshuffle_time=0*MPI_Wtime(), total_time=0*MPI_Wtime();

  int ptr=0;
  if(VERBOSE>=2) PCOUT<<"INPUT:"<<std::endl;
  if(VERBOSE>=2){
    cudaMemcpy(data_cpu, data, T_plan->alloc_local, cudaMemcpyDeviceToHost);
    for(int h=0;h<howmany;h++)
      for(int id=0;id<nprocs;++id){
        if(procid==id)
          for(int i=0;i<local_n0;i++){
            std::cout<<std::endl;
            for(int j=0;j<N[1];j++){
              ptr=h*idist+(i*N[1]+j)*n_tuples;
              std::cout<<'\t'<<data_cpu[ptr]<<","<<data_cpu[ptr+1];
            }
          }
        std::cout<<'\n';
        MPI_Barrier(T_plan->comm);
      }
  }
  //PCOUT<<" ============================================= "<<std::endl;
  //PCOUT<<" ==============   Local Transpose============= "<<std::endl;
  //PCOUT<<" ============================================= "<<std::endl;
  ptr=0;
  ptrdiff_t *local_n1_proc=&T_plan->local_n1_proc[0];
  ptrdiff_t *local_n0_proc=&T_plan->local_n0_proc[0];
  ptrdiff_t *local_0_start_proc=T_plan->local_0_start_proc;
  ptrdiff_t *local_1_start_proc=T_plan->local_1_start_proc;
  shuffle_time-=MPI_Wtime();
  if(nprocs==1 && Flags[0]==1 && Flags[1]==1){
#pragma omp parallel for
    for(int h=0;h<howmany;h++)
      local_transpose_cuda(local_n1,N[0],n_tuples,&data[h*idist] );
  }
  if(nprocs==1 && Flags[0]==0 && Flags[1]==0){
#pragma omp parallel for
    for(int h=0;h<howmany;h++)
      local_transpose_cuda(N[0],N[1],n_tuples,&data[h*idist] );
  }
  if(nprocs==1){ // Transpose is done!
    shuffle_time+=MPI_Wtime();
    timings[0]+=MPI_Wtime();
    timings[0]+=shuffle_time;
    timings[1]+=shuffle_time;
    timings[2]+=0;
    timings[3]+=0;
    MPI_Barrier(T_plan->comm);
    return;
  }

  // data_d --Th--> send_recv_d  --memcpy-->  data_cpu  --alltoall--> send_recv_cpu --Th--> data_d
  ptr=0;

  //for (int proc=0;proc<nprocs_1;++proc)
  //  for(int h=0;h<howmany;++h){
  //    for(int i=0;i<local_n0;++i){
  //      //for(int j=local_1_start_proc[proc];j<local_1_start_proc[proc]+local_n1_proc[proc];++j){
  //      //  memcpy(&buffer_2[ptr],&data[h*idist+(i*N[1]+j)*n_tuples],sizeof(double)*n_tuples);
  //      //  //std::cout<<"proc= "<<proc<<" h= "<<h<<" (i,j)=("<<i<<","<<j<<")  data_ptr= "<<h*idist+(i*local_n1+j)*n_tuples<< " ptr= "<<ptr <<" cpy= "<<n_tuples*T_plan->local_n1_proc[0] <<std::endl;
  //      //  ptr+=n_tuples;
  //      //}
  //      //memcpy(&buffer_2[ptr],&data[h*idist+(i*N[1]+local_1_start_proc[proc])*n_tuples],sizeof(double)*n_tuples*local_n1_proc[proc]);
  //      cudaMemcpy(&send_recv_d[ptr],&data[h*idist+(i*N[1]+local_1_start_proc[proc])*n_tuples] , sizeof(double)*n_tuples*local_n1_proc[proc] , cudaMemcpyDeviceToDevice);
  //      ptr+=n_tuples*local_n1_proc[proc];
  //    }
  //  }
  memcpy_v1_h1(nprocs_1,howmany,local_n0,n_tuples,local_n1_proc,send_recv_d,data,idist,N[1],local_1_start_proc);
  cudaDeviceSynchronize();


  shuffle_time+=MPI_Wtime();
  ptr=0;
  if(VERBOSE>=2) PCOUT<<"Local Transpose:"<<std::endl;
  if(VERBOSE>=2){
    cudaMemcpy(data_cpu, send_recv_d, T_plan->alloc_local, cudaMemcpyDeviceToHost);
    for(int id=0;id<nprocs;++id){
      for(int h=0;h<howmany;h++)
        if(procid==id)
          for(int i=0;i<N[1];i++){
            std::cout<<std::endl;
            for(int j=0;j<local_n0;j++){
              std::cout<<'\t'<<data_cpu[ptr]<<","<<data_cpu[ptr+1];
              ptr+=n_tuples;
            }
          }
      std::cout<<'\n';
      MPI_Barrier(T_plan->comm);
    }
  }


  //PCOUT<<" ============================================= "<<std::endl;
  //PCOUT<<" ==============   MPIALLTOALL  =============== "<<std::endl;
  //PCOUT<<" ============================================= "<<std::endl;

  int* scount_proc=  T_plan->scount_proc;
  int* rcount_proc=  T_plan->rcount_proc;
  int* soffset_proc= T_plan->soffset_proc;
  int* roffset_proc= T_plan->roffset_proc;

  MPI_Barrier(T_plan->comm);

  //PCOUT<<"nprocs_0= "<<nprocs_0<<" nprocs_1= "<<nprocs_1<<std::endl;
  comm_time-=MPI_Wtime();

  int soffset=0,roffset=0;
  MPI_Status ierr;
  MPI_Request * s_request= new MPI_Request[nprocs];
  MPI_Request * request= new MPI_Request[nprocs];
#pragma omp parallel for
  for (int proc=0;proc<nprocs;++proc){
    request[proc]=MPI_REQUEST_NULL;
    s_request[proc]=MPI_REQUEST_NULL;
  }

  double *s_buf, *r_buf;
  s_buf=data_cpu; r_buf=send_recv_cpu;
  double *r_buf_d=send_recv_d;
  // data_d --Th--> send_recv_d  --memcpy-->  data_cpu  --alltoall--> send_recv_cpu --Th--> data_d


  // Post Receives
  for (int proc=0;proc<nprocs;++proc){
    if(proc!=procid){
      roffset=roffset_proc[proc];
      MPI_Irecv(&r_buf[roffset*howmany],rcount_proc[proc]*howmany,MPI_DOUBLE, proc,
          tag, T_plan->comm, &request[proc]);
    }
  }
  // SEND
  cudaMemcpy(data_cpu, send_recv_d, T_plan->alloc_local, cudaMemcpyDeviceToHost);
  for (int proc=0;proc<nprocs;++proc){
    if(proc!=procid){
      soffset=soffset_proc[proc];
      MPI_Isend(&s_buf[soffset*howmany],scount_proc[proc]*howmany,MPI_DOUBLE,proc, tag,
          T_plan->comm, &s_request[proc]);
    }
  }

  soffset=soffset_proc[procid];//aoffset_proc[proc];//proc*count_proc[proc];
  roffset=roffset_proc[procid];
  memcpy(&r_buf[roffset*howmany],&s_buf[soffset*howmany],howmany*sizeof(double)*scount_proc[procid]);

  for (int proc=0;proc<nprocs;++proc){
    MPI_Wait(&request[proc], &ierr);
    MPI_Wait(&s_request[proc], &ierr);
  }

  // data_d --Th--> send_recv_d  --memcpy-->  data_cpu  --alltoall--> send_recv_cpu --Th--> data_d
  cudaMemcpy(r_buf_d, send_recv_cpu, T_plan->alloc_local, cudaMemcpyHostToDevice);
  cudaDeviceSynchronize();
  comm_time+=MPI_Wtime();


  ptr=0;
  if(VERBOSE>=2) PCOUT<<"MPIAlltoAll:"<<std::endl;
  if(VERBOSE>=2){
    cudaMemcpy(data_cpu, send_recv_d, T_plan->alloc_local, cudaMemcpyDeviceToHost);
    for(int id=0;id<nprocs;++id){
      if(procid==id)
        for(int h=0;h<howmany;h++)
          for(int i=0;i<local_n1;i++){
            std::cout<<std::endl;
            for(int j=0;j<N[0];j++){
              std::cout<<'\t'<<data_cpu[ptr]<<","<<data_cpu[ptr+1];
              ptr+=n_tuples;
            }
          }
      std::cout<<'\n';
      MPI_Barrier(T_plan->comm);
    }
  }
  //PCOUT<<" ============================================= "<<std::endl;
  //PCOUT<<" ============== 2nd Local Trnaspose ========== "<<std::endl;
  //PCOUT<<" ============================================= "<<std::endl;
  // data_d --Th--> send_recv_d  --memcpy-->  data_cpu  --alltoall--> send_recv_cpu --Th--> data_d
  reshuffle_time-=MPI_Wtime();
  ptr=0;
  //for (int proc=0;proc<nprocs_0;++proc)
  //  for(int h=0;h<howmany;++h){
  //    for(int i=local_0_start_proc[proc];i<local_0_start_proc[proc]+local_n0_proc[proc];++i){
  //      //memcpy(&data_cpu[h*odist+(i*local_n1)*n_tuples],&send_recv_cpu[ptr],local_n1*sizeof(double)*n_tuples);
  //      cudaMemcpy(    &data[h*odist+(i*local_n1)*n_tuples],&send_recv_cpu[ptr],local_n1*sizeof(double)*n_tuples,cudaMemcpyHostToDevice);
  //      //std::cout<<"proc= "<<proc<<" h= "<<h<<" i=("<<i<<")  data_ptr= "<<h*odist+(i*local_n1)*n_tuples<< " ptr= "<<ptr <<" cpy= "<<n_tuples*local_n1_proc[proc] <<std::endl;
  //      ptr+=n_tuples*local_n1;
  //      //for(int j=0*local_1_start_proc[proc];j<0*local_1_start_proc[proc]+local_n1;++j){
  //      //  memcpy(&data[h*odist+(i*local_n1+j)*n_tuples],&send_recv[ptr],sizeof(double)*n_tuples);
  //      //std::cout<<"proc= "<<proc<<" h= "<<h<<" (i,j)=("<<i<<","<<j<<")  data_ptr= "<<h*idist+(i*local_n1+j)*n_tuples<< " ptr= "<<ptr <<" cpy= "<<n_tuples*T_plan->local_n1_proc[0] <<std::endl;
  //      //  ptr+=n_tuples;
  //      //}
  //    }
  //  }
 //memcpy_v1_h2(nprocs_0,howmany,local_0_start_proc,local_n0_proc,data,odist,local_n1,n_tuples,send_recv_cpu);
  memcpy_v1_h2(nprocs_0,howmany,local_0_start_proc,local_n0_proc,data,odist,local_n1,n_tuples,send_recv_d);
  cudaDeviceSynchronize();

  if(VERBOSE>=2) PCOUT<<"2nd Transpose"<<std::endl;
  if(VERBOSE>=2){
    cudaMemcpy(data_cpu, data, T_plan->alloc_local, cudaMemcpyDeviceToHost);
    for(int id=0;id<nprocs_1;++id){
      if(procid==id)
        for(int h=0;h<howmany;h++)
          for(int i=0;i<N[0];i++){
            std::cout<<std::endl;
            for(int j=0;j<local_n1;j++){
              ptr=h*odist+(i*local_n1+j)*n_tuples;
              std::cout<<'\t'<<data_cpu[ptr]<<","<<data_cpu[ptr+1];
            }
          }
      std::cout<<'\n';
      MPI_Barrier(T_plan->comm);
    }
  }
  // Right now the data is in transposed out format.
  // If the user did not want this layout, transpose again.
  if(Flags[1]==0){
#pragma omp parallel for
    for(int h=0;h<howmany;h++)
      local_transpose_cuda(N[0],local_n1,n_tuples,&data[h*odist] );

    if(VERBOSE>=2) PCOUT<<"2nd Transpose"<<std::endl;
    if(VERBOSE>=2){
      cudaMemcpy(data_cpu, data, T_plan->alloc_local, cudaMemcpyDeviceToHost);
      for(int id=0;id<nprocs_1;++id){
        if(procid==id)
          for(int h=0;h<howmany;h++)
            for(int i=0;i<N[0];i++){
              std::cout<<std::endl;
              for(int j=0;j<local_n1;j++){
                ptr=h*odist+(i*local_n1+j)*n_tuples;
                std::cout<<'\t'<<data_cpu[ptr]<<","<<data_cpu[ptr+1];
              }
            }
        std::cout<<'\n';
        MPI_Barrier(T_plan->comm);
      }
    }
  }

  reshuffle_time+=MPI_Wtime();
  MPI_Barrier(T_plan->comm);
  delete [] request;
  delete [] s_request;


  if(VERBOSE>=1){
    PCOUT<<"Shuffle Time= "<<shuffle_time<<std::endl;
    PCOUT<<"Alltoall Time= "<<comm_time<<std::endl;
    PCOUT<<"Reshuffle Time= "<<reshuffle_time<<std::endl;
    PCOUT<<"Total Time= "<<(shuffle_time+comm_time+reshuffle_time)<<std::endl;
  }
  timings[0]+=MPI_Wtime();//timings[0]+=shuffle_time+comm_time+reshuffle_time;
  timings[1]+=shuffle_time;
  timings[2]+=comm_time;
  timings[3]+=reshuffle_time;
  return;
}

void fast_transpose_cuda_v1_2_h(T_Plan_gpu* T_plan, double * data, double *timings, unsigned flags, int howmany, int tag ){

  std::bitset<8> Flags(flags); // 1 Transposed in, 2 Transposed out
  if(Flags[1]==1 && Flags[0]==0 && T_plan->nprocs==1){ // If nprocs==1 and Flags==Transposed_Out return
    MPI_Barrier(T_plan->comm);
    return;
  }
  if(Flags[0]==1){ // If Flags==Transposed_In This function can not handle it, call other versions
    transpose_cuda_v5(T_plan,(double*)data,timings,flags,howmany,tag);
    MPI_Barrier(T_plan->comm);
    return;
  }
  timings[0]-=MPI_Wtime();
  int nprocs, procid;
  int nprocs_0, nprocs_1;
  nprocs=T_plan->nprocs;
  procid=T_plan->procid;
  nprocs_0=T_plan->nprocs_0;
  nprocs_1=T_plan->nprocs_1;
  ptrdiff_t  *N=T_plan->N;
  ptrdiff_t local_n0=T_plan->local_n0;
  ptrdiff_t local_n1=T_plan->local_n1;
  ptrdiff_t n_tuples=T_plan->n_tuples;

  double * data_cpu=T_plan->buffer;  //pinned
  double * send_recv_cpu = T_plan->buffer_2;//pinned
  double * send_recv_d = T_plan->buffer_d;

  int idist=N[1]*local_n0*n_tuples;
  int odist=N[0]*local_n1*n_tuples;

  double comm_time=0*MPI_Wtime(), shuffle_time=0*MPI_Wtime(), reshuffle_time=0*MPI_Wtime(), total_time=0*MPI_Wtime();

  int ptr=0;
  if(VERBOSE>=2) PCOUT<<"INPUT:"<<std::endl;
  if(VERBOSE>=2){
    cudaMemcpy(data_cpu, data, T_plan->alloc_local, cudaMemcpyDeviceToHost);
    for(int h=0;h<howmany;h++)
      for(int id=0;id<nprocs;++id){
        if(procid==id)
          for(int i=0;i<local_n0;i++){
            std::cout<<std::endl;
            for(int j=0;j<N[1];j++){
              ptr=h*idist+(i*N[1]+j)*n_tuples;
              std::cout<<'\t'<<data_cpu[ptr]<<","<<data_cpu[ptr+1];
            }
          }
        std::cout<<'\n';
        MPI_Barrier(T_plan->comm);
      }
  }
  //PCOUT<<" ============================================= "<<std::endl;
  //PCOUT<<" ==============   Local Transpose============= "<<std::endl;
  //PCOUT<<" ============================================= "<<std::endl;
  ptr=0;
  ptrdiff_t *local_n1_proc=&T_plan->local_n1_proc[0];
  ptrdiff_t *local_n0_proc=&T_plan->local_n0_proc[0];
  ptrdiff_t *local_0_start_proc=T_plan->local_0_start_proc;
  ptrdiff_t *local_1_start_proc=T_plan->local_1_start_proc;
  shuffle_time-=MPI_Wtime();
  if(nprocs==1 && Flags[0]==1 && Flags[1]==1){
#pragma omp parallel for
    for(int h=0;h<howmany;h++)
      local_transpose_cuda(local_n1,N[0],n_tuples,&data[h*idist] );
  }
  if(nprocs==1 && Flags[0]==0 && Flags[1]==0){
#pragma omp parallel for
    for(int h=0;h<howmany;h++)
      local_transpose_cuda(N[0],N[1],n_tuples,&data[h*idist] );
  }
  if(nprocs==1){ // Transpose is done!
    shuffle_time+=MPI_Wtime();
    timings[0]+=MPI_Wtime();
    timings[0]+=shuffle_time;
    timings[1]+=shuffle_time;
    timings[2]+=0;
    timings[3]+=0;
    MPI_Barrier(T_plan->comm);
    return;
  }

  // data_d --Th--> send_recv_d  --memcpy-->  data_cpu  --alltoall--> send_recv_cpu --Th--> data_d
  ptr=0;

  //for (int proc=0;proc<nprocs_1;++proc)
  //  for(int h=0;h<howmany;++h){
  //    for(int i=0;i<local_n0;++i){
  //      //for(int j=local_1_start_proc[proc];j<local_1_start_proc[proc]+local_n1_proc[proc];++j){
  //      //  memcpy(&buffer_2[ptr],&data[h*idist+(i*N[1]+j)*n_tuples],sizeof(double)*n_tuples);
  //      //  //std::cout<<"proc= "<<proc<<" h= "<<h<<" (i,j)=("<<i<<","<<j<<")  data_ptr= "<<h*idist+(i*local_n1+j)*n_tuples<< " ptr= "<<ptr <<" cpy= "<<n_tuples*T_plan->local_n1_proc[0] <<std::endl;
  //      //  ptr+=n_tuples;
  //      //}
  //      //memcpy(&buffer_2[ptr],&data[h*idist+(i*N[1]+local_1_start_proc[proc])*n_tuples],sizeof(double)*n_tuples*local_n1_proc[proc]);
  //      cudaMemcpy(&send_recv_d[ptr],&data[h*idist+(i*N[1]+local_1_start_proc[proc])*n_tuples] , sizeof(double)*n_tuples*local_n1_proc[proc] , cudaMemcpyDeviceToDevice);
  //      ptr+=n_tuples*local_n1_proc[proc];
  //    }
  //  }
  memcpy_v1_h1(nprocs_1,howmany,local_n0,n_tuples,local_n1_proc,send_recv_d,data,idist,N[1],local_1_start_proc);
  cudaDeviceSynchronize();


  shuffle_time+=MPI_Wtime();
  ptr=0;
  if(VERBOSE>=2) PCOUT<<"Local Transpose:"<<std::endl;
  if(VERBOSE>=2){
    cudaMemcpy(data_cpu, send_recv_d, T_plan->alloc_local, cudaMemcpyDeviceToHost);
    for(int id=0;id<nprocs;++id){
      for(int h=0;h<howmany;h++)
        if(procid==id)
          for(int i=0;i<N[1];i++){
            std::cout<<std::endl;
            for(int j=0;j<local_n0;j++){
              std::cout<<'\t'<<data_cpu[ptr]<<","<<data_cpu[ptr+1];
              ptr+=n_tuples;
            }
          }
      std::cout<<'\n';
      MPI_Barrier(T_plan->comm);
    }
  }


  //PCOUT<<" ============================================= "<<std::endl;
  //PCOUT<<" ==============   MPIALLTOALL  =============== "<<std::endl;
  //PCOUT<<" ============================================= "<<std::endl;

  int* scount_proc=  T_plan->scount_proc;
  int* rcount_proc=  T_plan->rcount_proc;
  int* soffset_proc= T_plan->soffset_proc;
  int* roffset_proc= T_plan->roffset_proc;

  MPI_Barrier(T_plan->comm);

  //PCOUT<<"nprocs_0= "<<nprocs_0<<" nprocs_1= "<<nprocs_1<<std::endl;
  comm_time-=MPI_Wtime();

  int soffset=0,roffset=0;
  MPI_Status ierr;
  MPI_Request * s_request= new MPI_Request[nprocs];
  MPI_Request * request= new MPI_Request[nprocs];
  int flag[nprocs],color[nprocs];
  memset(flag,0,sizeof(int)*nprocs);
  memset(color,0,sizeof(int)*nprocs);
  int counter=1;
#pragma omp parallel for
  for (int proc=0;proc<nprocs;++proc){
    request[proc]=MPI_REQUEST_NULL;
    s_request[proc]=MPI_REQUEST_NULL;
  }

  double *s_buf, *r_buf;
  s_buf=data_cpu; r_buf=send_recv_cpu;
  double *r_buf_d=send_recv_d;
  // data_d --Th--> send_recv_d  --memcpy-->  data_cpu  --alltoall--> send_recv_cpu --memcpy--> send_recv_d --Th--> data_d


  // SEND
  //cudaMemcpy(data_cpu, send_recv_d, T_plan->alloc_local, cudaMemcpyDeviceToHost);
  for (int proc=0;proc<nprocs;++proc){
    if(proc!=procid){
      soffset=soffset_proc[proc];
      cudaMemcpy(&s_buf[soffset*howmany], &send_recv_d[soffset*howmany],howmany*sizeof(double)*scount_proc[proc], cudaMemcpyDeviceToHost);
      MPI_Isend(&s_buf[soffset*howmany],scount_proc[proc]*howmany, MPI_DOUBLE,proc, tag,
          T_plan->comm, &s_request[proc]);

    }
    else{  // copy your own part directly
      //cudaMemcpy(&r_buf_d[roffset_proc[procid]*howmany], &send_recv_d[soffset_proc[procid]*howmany],howmany*sizeof(double)*scount_proc[procid], cudaMemcpyDeviceToDevice);
      // Note that because if Flags[1]=0 the send_d and recv_d are the same, D2D should not be used
      // Otherwise it will corrupt the data in the unbalanced cases
      cudaMemcpy(&r_buf[roffset_proc[procid]*howmany], &send_recv_d[soffset_proc[procid]*howmany],howmany*sizeof(double)*scount_proc[procid], cudaMemcpyDeviceToHost);
      flag[procid]=1;
    }
  }
  for (int proc=0;proc<nprocs;++proc){
    if(proc!=procid){
      roffset=roffset_proc[proc];
      MPI_Irecv(&r_buf[roffset*howmany],rcount_proc[proc]*howmany,MPI_DOUBLE, proc,
          tag, T_plan->comm, &request[proc]);
    }
  }
  while(counter!=nprocs+1){

    for (int proc=0;proc<nprocs;++proc){
      MPI_Test(&request[proc], &flag[proc],&ierr);
      if(flag[proc]==1 && color[proc]==0){
        cudaMemcpyAsync(&r_buf_d[roffset_proc[proc]*howmany],&r_buf[roffset_proc[proc]*howmany],howmany*sizeof(double)*rcount_proc[proc],cudaMemcpyHostToDevice);
        color[proc]=1;
        counter+=1;
      }
    }

  }
  //cudaMemcpy(r_buf_d, send_recv_cpu, T_plan->alloc_local, cudaMemcpyHostToDevice);
  //cudaMemcpy(&r_buf_d[roffset_proc[procid]*howmany], &send_recv_d[soffset_proc[procid]*howmany],howmany*sizeof(double)*scount_proc[procid], cudaMemcpyDeviceToDevice);
  cudaDeviceSynchronize();
  comm_time+=MPI_Wtime();





  ptr=0;
  if(VERBOSE>=2) PCOUT<<"MPIAlltoAll:"<<std::endl;
  if(VERBOSE>=2){
    cudaMemcpy(data_cpu, send_recv_d, T_plan->alloc_local, cudaMemcpyDeviceToHost);
    for(int id=0;id<nprocs;++id){
      if(procid==id)
        for(int h=0;h<howmany;h++)
          for(int i=0;i<local_n1;i++){
            std::cout<<std::endl;
            for(int j=0;j<N[0];j++){
              std::cout<<'\t'<<data_cpu[ptr]<<","<<data_cpu[ptr+1];
              ptr+=n_tuples;
            }
          }
      std::cout<<'\n';
      MPI_Barrier(T_plan->comm);
    }
  }
  //PCOUT<<" ============================================= "<<std::endl;
  //PCOUT<<" ============== 2nd Local Trnaspose ========== "<<std::endl;
  //PCOUT<<" ============================================= "<<std::endl;
  // data_d --Th--> send_recv_d  --memcpy-->  data_cpu  --alltoall--> send_recv_cpu --Th--> data_d
  reshuffle_time-=MPI_Wtime();
  ptr=0;
  //for (int proc=0;proc<nprocs_0;++proc)
  //  for(int h=0;h<howmany;++h){
  //    for(int i=local_0_start_proc[proc];i<local_0_start_proc[proc]+local_n0_proc[proc];++i){
  //      //memcpy(&data_cpu[h*odist+(i*local_n1)*n_tuples],&send_recv_cpu[ptr],local_n1*sizeof(double)*n_tuples);
  //      cudaMemcpy(    &data[h*odist+(i*local_n1)*n_tuples],&send_recv_cpu[ptr],local_n1*sizeof(double)*n_tuples,cudaMemcpyHostToDevice);
  //      //std::cout<<"proc= "<<proc<<" h= "<<h<<" i=("<<i<<")  data_ptr= "<<h*odist+(i*local_n1)*n_tuples<< " ptr= "<<ptr <<" cpy= "<<n_tuples*local_n1_proc[proc] <<std::endl;
  //      ptr+=n_tuples*local_n1;
  //      //for(int j=0*local_1_start_proc[proc];j<0*local_1_start_proc[proc]+local_n1;++j){
  //      //  memcpy(&data[h*odist+(i*local_n1+j)*n_tuples],&send_recv[ptr],sizeof(double)*n_tuples);
  //      //std::cout<<"proc= "<<proc<<" h= "<<h<<" (i,j)=("<<i<<","<<j<<")  data_ptr= "<<h*idist+(i*local_n1+j)*n_tuples<< " ptr= "<<ptr <<" cpy= "<<n_tuples*T_plan->local_n1_proc[0] <<std::endl;
  //      //  ptr+=n_tuples;
  //      //}
  //    }
  //  }
 //memcpy_v1_h2(nprocs_0,howmany,local_0_start_proc,local_n0_proc,data,odist,local_n1,n_tuples,send_recv_cpu);
  memcpy_v1_h2(nprocs_0,howmany,local_0_start_proc,local_n0_proc,data,odist,local_n1,n_tuples,send_recv_d);
  cudaDeviceSynchronize();

  if(VERBOSE>=2) PCOUT<<"2nd Transpose"<<std::endl;
  if(VERBOSE>=2){
    cudaMemcpy(data_cpu, data, T_plan->alloc_local, cudaMemcpyDeviceToHost);
    for(int id=0;id<nprocs_1;++id){
      if(procid==id)
        for(int h=0;h<howmany;h++)
          for(int i=0;i<N[0];i++){
            std::cout<<std::endl;
            for(int j=0;j<local_n1;j++){
              ptr=h*odist+(i*local_n1+j)*n_tuples;
              std::cout<<'\t'<<data_cpu[ptr]<<","<<data_cpu[ptr+1];
            }
          }
      std::cout<<'\n';
      MPI_Barrier(T_plan->comm);
    }
  }
  // Right now the data is in transposed out format.
  // If the user did not want this layout, transpose again.
  if(Flags[1]==0){
#pragma omp parallel for
    for(int h=0;h<howmany;h++)
      local_transpose_cuda(N[0],local_n1,n_tuples,&data[h*odist] );

    if(VERBOSE>=2) PCOUT<<"2nd Transpose"<<std::endl;
    if(VERBOSE>=2){
      cudaMemcpy(data_cpu, data, T_plan->alloc_local, cudaMemcpyDeviceToHost);
      for(int id=0;id<nprocs_1;++id){
        if(procid==id)
          for(int h=0;h<howmany;h++)
            for(int i=0;i<N[0];i++){
              std::cout<<std::endl;
              for(int j=0;j<local_n1;j++){
                ptr=h*odist+(i*local_n1+j)*n_tuples;
                std::cout<<'\t'<<data_cpu[ptr]<<","<<data_cpu[ptr+1];
              }
            }
        std::cout<<'\n';
        MPI_Barrier(T_plan->comm);
      }
    }
  }

  reshuffle_time+=MPI_Wtime();
  MPI_Barrier(T_plan->comm);
  delete [] request;
  delete [] s_request;


  if(VERBOSE>=1){
    PCOUT<<"Shuffle Time= "<<shuffle_time<<std::endl;
    PCOUT<<"Alltoall Time= "<<comm_time<<std::endl;
    PCOUT<<"Reshuffle Time= "<<reshuffle_time<<std::endl;
    PCOUT<<"Total Time= "<<(shuffle_time+comm_time+reshuffle_time)<<std::endl;
  }
  timings[0]+=MPI_Wtime();//timings[0]+=shuffle_time+comm_time+reshuffle_time;
  timings[1]+=shuffle_time;
  timings[2]+=comm_time;
  timings[3]+=reshuffle_time;
  return;
}
void fast_transpose_cuda_v1_3_h(T_Plan_gpu* T_plan, double * data, double *timings, unsigned flags, int howmany, int tag ){

  std::bitset<8> Flags(flags); // 1 Transposed in, 2 Transposed out
  if(Flags[1]==1 && Flags[0]==0 && T_plan->nprocs==1){ // If nprocs==1 and Flags==Transposed_Out return
    MPI_Barrier(T_plan->comm);
    return;
  }
  if(Flags[0]==1){ // If Flags==Transposed_In This function can not handle it, call other versions
    transpose_cuda_v5(T_plan,(double*)data,timings,flags,howmany,tag);
    MPI_Barrier(T_plan->comm);
    return;
  }
  timings[0]-=MPI_Wtime();
  int nprocs, procid;
  int nprocs_0, nprocs_1;
  nprocs=T_plan->nprocs;
  procid=T_plan->procid;
  nprocs_0=T_plan->nprocs_0;
  nprocs_1=T_plan->nprocs_1;
  ptrdiff_t  *N=T_plan->N;
  ptrdiff_t local_n0=T_plan->local_n0;
  ptrdiff_t local_n1=T_plan->local_n1;
  ptrdiff_t n_tuples=T_plan->n_tuples;

  double * data_cpu=T_plan->buffer;  //pinned
  double * send_recv_cpu = T_plan->buffer_2;//pinned
  double * send_recv_d = T_plan->buffer_d;

  int idist=N[1]*local_n0*n_tuples;
  int odist=N[0]*local_n1*n_tuples;

  double comm_time=0*MPI_Wtime(), shuffle_time=0*MPI_Wtime(), reshuffle_time=0*MPI_Wtime(), total_time=0*MPI_Wtime();

  int ptr=0;
  if(VERBOSE>=2) PCOUT<<"INPUT:"<<std::endl;
  if(VERBOSE>=2){
    cudaMemcpy(data_cpu, data, T_plan->alloc_local, cudaMemcpyDeviceToHost);
    for(int h=0;h<howmany;h++)
      for(int id=0;id<nprocs;++id){
        if(procid==id)
          for(int i=0;i<local_n0;i++){
            std::cout<<std::endl;
            for(int j=0;j<N[1];j++){
              ptr=h*idist+(i*N[1]+j)*n_tuples;
              std::cout<<'\t'<<data_cpu[ptr]<<","<<data_cpu[ptr+1];
            }
          }
        std::cout<<'\n';
        MPI_Barrier(T_plan->comm);
      }
  }
  //PCOUT<<" ============================================= "<<std::endl;
  //PCOUT<<" ==============   Local Transpose============= "<<std::endl;
  //PCOUT<<" ============================================= "<<std::endl;
  ptr=0;
  ptrdiff_t *local_n1_proc=&T_plan->local_n1_proc[0];
  ptrdiff_t *local_n0_proc=&T_plan->local_n0_proc[0];
  ptrdiff_t *local_0_start_proc=T_plan->local_0_start_proc;
  ptrdiff_t *local_1_start_proc=T_plan->local_1_start_proc;
  shuffle_time-=MPI_Wtime();
  if(nprocs==1 && Flags[0]==1 && Flags[1]==1){
#pragma omp parallel for
    for(int h=0;h<howmany;h++)
      local_transpose_cuda(local_n1,N[0],n_tuples,&data[h*idist] );
  }
  if(nprocs==1 && Flags[0]==0 && Flags[1]==0){
#pragma omp parallel for
    for(int h=0;h<howmany;h++)
      local_transpose_cuda(N[0],N[1],n_tuples,&data[h*idist] );
  }
  if(nprocs==1){ // Transpose is done!
    shuffle_time+=MPI_Wtime();
    timings[0]+=MPI_Wtime();
    timings[0]+=shuffle_time;
    timings[1]+=shuffle_time;
    timings[2]+=0;
    timings[3]+=0;
    MPI_Barrier(T_plan->comm);
    return;
  }

  // data_d --Th--> send_recv_d  --memcpy-->  data_cpu  --alltoall--> send_recv_cpu --Th--> data_d
  ptr=0;

  //for (int proc=0;proc<nprocs_1;++proc)
  //  for(int h=0;h<howmany;++h){
  //    for(int i=0;i<local_n0;++i){
  //      //for(int j=local_1_start_proc[proc];j<local_1_start_proc[proc]+local_n1_proc[proc];++j){
  //      //  memcpy(&buffer_2[ptr],&data[h*idist+(i*N[1]+j)*n_tuples],sizeof(double)*n_tuples);
  //      //  //std::cout<<"proc= "<<proc<<" h= "<<h<<" (i,j)=("<<i<<","<<j<<")  data_ptr= "<<h*idist+(i*local_n1+j)*n_tuples<< " ptr= "<<ptr <<" cpy= "<<n_tuples*T_plan->local_n1_proc[0] <<std::endl;
  //      //  ptr+=n_tuples;
  //      //}
  //      //memcpy(&buffer_2[ptr],&data[h*idist+(i*N[1]+local_1_start_proc[proc])*n_tuples],sizeof(double)*n_tuples*local_n1_proc[proc]);
  //      cudaMemcpy(&send_recv_d[ptr],&data[h*idist+(i*N[1]+local_1_start_proc[proc])*n_tuples] , sizeof(double)*n_tuples*local_n1_proc[proc] , cudaMemcpyDeviceToDevice);
  //      ptr+=n_tuples*local_n1_proc[proc];
  //    }
  //  }
  memcpy_v1_h1(nprocs_1,howmany,local_n0,n_tuples,local_n1_proc,send_recv_d,data,idist,N[1],local_1_start_proc);
  cudaDeviceSynchronize();


  shuffle_time+=MPI_Wtime();
  ptr=0;
  if(VERBOSE>=2) PCOUT<<"Local Transpose:"<<std::endl;
  if(VERBOSE>=2){
    cudaMemcpy(data_cpu, send_recv_d, T_plan->alloc_local, cudaMemcpyDeviceToHost);
    for(int id=0;id<nprocs;++id){
      for(int h=0;h<howmany;h++)
        if(procid==id)
          for(int i=0;i<N[1];i++){
            std::cout<<std::endl;
            for(int j=0;j<local_n0;j++){
              std::cout<<'\t'<<data_cpu[ptr]<<","<<data_cpu[ptr+1];
              ptr+=n_tuples;
            }
          }
      std::cout<<'\n';
      MPI_Barrier(T_plan->comm);
    }
  }


  //PCOUT<<" ============================================= "<<std::endl;
  //PCOUT<<" ==============   MPIALLTOALL  =============== "<<std::endl;
  //PCOUT<<" ============================================= "<<std::endl;

  int* scount_proc=  T_plan->scount_proc;
  int* rcount_proc=  T_plan->rcount_proc;
  int* soffset_proc= T_plan->soffset_proc;
  int* roffset_proc= T_plan->roffset_proc;

  MPI_Barrier(T_plan->comm);

  //PCOUT<<"nprocs_0= "<<nprocs_0<<" nprocs_1= "<<nprocs_1<<std::endl;
  comm_time-=MPI_Wtime();

  int soffset=0,roffset=0;
  MPI_Status ierr;

  double *s_buf, *r_buf;
  s_buf=data_cpu; r_buf=send_recv_cpu;
  double *r_buf_d=send_recv_d;
  // data_d --Th--> send_recv_d  --memcpy-->  data_cpu  --alltoall--> send_recv_cpu --memcpy--> send_recv_d --Th--> data_d


  // SEND
  //cudaMemcpy(data_cpu, send_recv_d, T_plan->alloc_local, cudaMemcpyDeviceToHost);
  for (int proc=0;proc<nprocs;++proc){
    if(proc!=procid){
      soffset=soffset_proc[proc];
      roffset=roffset_proc[proc];
      cudaMemcpy(&s_buf[soffset*howmany], &send_recv_d[soffset*howmany],howmany*sizeof(double)*scount_proc[proc], cudaMemcpyDeviceToHost);
      MPI_Sendrecv(&s_buf[soffset*howmany],scount_proc[proc]*howmany, MPI_DOUBLE,
          proc, tag,
          &r_buf[roffset*howmany],howmany*rcount_proc[proc], MPI_DOUBLE,
          proc, tag,
          T_plan->comm,&ierr);

    }
    else{  // copy your own part directly
      //cudaMemcpy(&r_buf_d[roffset_proc[procid]*howmany], &send_recv_d[soffset_proc[procid]*howmany],howmany*sizeof(double)*scount_proc[procid], cudaMemcpyDeviceToDevice);
      // Note that because if Flags[1]=0 the send_d and recv_d are the same, D2D should not be used
      // Otherwise it will corrupt the data in the unbalanced cases
      cudaMemcpy(&r_buf[roffset_proc[procid]*howmany], &send_recv_d[soffset_proc[procid]*howmany],howmany*sizeof(double)*scount_proc[procid], cudaMemcpyDeviceToHost);
    }
  }

  cudaMemcpy(r_buf_d,r_buf,T_plan->alloc_local,cudaMemcpyHostToDevice);
  //cudaMemcpy(r_buf_d, send_recv_cpu, T_plan->alloc_local, cudaMemcpyHostToDevice);
  //cudaMemcpy(&r_buf_d[roffset_proc[procid]*howmany], &send_recv_d[soffset_proc[procid]*howmany],howmany*sizeof(double)*scount_proc[procid], cudaMemcpyDeviceToDevice);
  cudaDeviceSynchronize();
  comm_time+=MPI_Wtime();


  ptr=0;
  if(VERBOSE>=2) PCOUT<<"MPIAlltoAll:"<<std::endl;
  if(VERBOSE>=2){
    cudaMemcpy(data_cpu, send_recv_d, T_plan->alloc_local, cudaMemcpyDeviceToHost);
    for(int id=0;id<nprocs;++id){
      if(procid==id)
        for(int h=0;h<howmany;h++)
          for(int i=0;i<local_n1;i++){
            std::cout<<std::endl;
            for(int j=0;j<N[0];j++){
              std::cout<<'\t'<<data_cpu[ptr]<<","<<data_cpu[ptr+1];
              ptr+=n_tuples;
            }
          }
      std::cout<<'\n';
      MPI_Barrier(T_plan->comm);
    }
  }
  //PCOUT<<" ============================================= "<<std::endl;
  //PCOUT<<" ============== 2nd Local Trnaspose ========== "<<std::endl;
  //PCOUT<<" ============================================= "<<std::endl;
  // data_d --Th--> send_recv_d  --memcpy-->  data_cpu  --alltoall--> send_recv_cpu --Th--> data_d
  reshuffle_time-=MPI_Wtime();
  ptr=0;
  //for (int proc=0;proc<nprocs_0;++proc)
  //  for(int h=0;h<howmany;++h){
  //    for(int i=local_0_start_proc[proc];i<local_0_start_proc[proc]+local_n0_proc[proc];++i){
  //      //memcpy(&data_cpu[h*odist+(i*local_n1)*n_tuples],&send_recv_cpu[ptr],local_n1*sizeof(double)*n_tuples);
  //      cudaMemcpy(    &data[h*odist+(i*local_n1)*n_tuples],&send_recv_cpu[ptr],local_n1*sizeof(double)*n_tuples,cudaMemcpyHostToDevice);
  //      //std::cout<<"proc= "<<proc<<" h= "<<h<<" i=("<<i<<")  data_ptr= "<<h*odist+(i*local_n1)*n_tuples<< " ptr= "<<ptr <<" cpy= "<<n_tuples*local_n1_proc[proc] <<std::endl;
  //      ptr+=n_tuples*local_n1;
  //      //for(int j=0*local_1_start_proc[proc];j<0*local_1_start_proc[proc]+local_n1;++j){
  //      //  memcpy(&data[h*odist+(i*local_n1+j)*n_tuples],&send_recv[ptr],sizeof(double)*n_tuples);
  //      //std::cout<<"proc= "<<proc<<" h= "<<h<<" (i,j)=("<<i<<","<<j<<")  data_ptr= "<<h*idist+(i*local_n1+j)*n_tuples<< " ptr= "<<ptr <<" cpy= "<<n_tuples*T_plan->local_n1_proc[0] <<std::endl;
  //      //  ptr+=n_tuples;
  //      //}
  //    }
  //  }
 //memcpy_v1_h2(nprocs_0,howmany,local_0_start_proc,local_n0_proc,data,odist,local_n1,n_tuples,send_recv_cpu);
  memcpy_v1_h2(nprocs_0,howmany,local_0_start_proc,local_n0_proc,data,odist,local_n1,n_tuples,send_recv_d);
  cudaDeviceSynchronize();

  if(VERBOSE>=2) PCOUT<<"2nd Transpose"<<std::endl;
  if(VERBOSE>=2){
    cudaMemcpy(data_cpu, data, T_plan->alloc_local, cudaMemcpyDeviceToHost);
    for(int id=0;id<nprocs_1;++id){
      if(procid==id)
        for(int h=0;h<howmany;h++)
          for(int i=0;i<N[0];i++){
            std::cout<<std::endl;
            for(int j=0;j<local_n1;j++){
              ptr=h*odist+(i*local_n1+j)*n_tuples;
              std::cout<<'\t'<<data_cpu[ptr]<<","<<data_cpu[ptr+1];
            }
          }
      std::cout<<'\n';
      MPI_Barrier(T_plan->comm);
    }
  }
  // Right now the data is in transposed out format.
  // If the user did not want this layout, transpose again.
  if(Flags[1]==0){
#pragma omp parallel for
    for(int h=0;h<howmany;h++)
      local_transpose_cuda(N[0],local_n1,n_tuples,&data[h*odist] );

    if(VERBOSE>=2) PCOUT<<"2nd Transpose"<<std::endl;
    if(VERBOSE>=2){
      cudaMemcpy(data_cpu, data, T_plan->alloc_local, cudaMemcpyDeviceToHost);
      for(int id=0;id<nprocs_1;++id){
        if(procid==id)
          for(int h=0;h<howmany;h++)
            for(int i=0;i<N[0];i++){
              std::cout<<std::endl;
              for(int j=0;j<local_n1;j++){
                ptr=h*odist+(i*local_n1+j)*n_tuples;
                std::cout<<'\t'<<data_cpu[ptr]<<","<<data_cpu[ptr+1];
              }
            }
        std::cout<<'\n';
        MPI_Barrier(T_plan->comm);
      }
    }
  }

  reshuffle_time+=MPI_Wtime();
  MPI_Barrier(T_plan->comm);

  if(VERBOSE>=1){
    PCOUT<<"Shuffle Time= "<<shuffle_time<<std::endl;
    PCOUT<<"Alltoall Time= "<<comm_time<<std::endl;
    PCOUT<<"Reshuffle Time= "<<reshuffle_time<<std::endl;
    PCOUT<<"Total Time= "<<(shuffle_time+comm_time+reshuffle_time)<<std::endl;
  }
  timings[0]+=MPI_Wtime();//timings[0]+=shuffle_time+comm_time+reshuffle_time;
  timings[1]+=shuffle_time;
  timings[2]+=comm_time;
  timings[3]+=reshuffle_time;
  return;
}

void fast_transpose_cuda_v1(T_Plan_gpu* T_plan, double * data, double *timings, unsigned flags, int howmany, int tag ){

  std::bitset<8> Flags(flags); // 1 Transposed in, 2 Transposed out
  if(Flags[1]==1 && Flags[0]==0 && T_plan->nprocs==1){ // If Flags==Transposed_Out return
    MPI_Barrier(T_plan->comm);
    return;
  }
  if(Flags[0]==1){ // If Flags==Transposed_In This function can not handle it, call other versions
    transpose_cuda_v5(T_plan,(double*)data,timings,flags,howmany,tag);
    MPI_Barrier(T_plan->comm);
    return;
  }
  timings[0]-=MPI_Wtime();
  int nprocs, procid;
  nprocs=T_plan->nprocs;
  procid=T_plan->procid;
  int nprocs_1=T_plan->nprocs_1;
  ptrdiff_t  *N=T_plan->N;

  double * data_cpu=T_plan->buffer;  //pinned
  double * send_recv_cpu = T_plan->buffer_2;//pinned
  double * send_recv_d = T_plan->buffer_d;


  ptrdiff_t local_n0=T_plan->local_n0;
  ptrdiff_t local_n1=T_plan->local_n1;
  ptrdiff_t n_tuples=T_plan->n_tuples;

  int idist=N[1]*local_n0*n_tuples;
  int odist=N[0]*local_n1*n_tuples;

  double comm_time=0*MPI_Wtime(), shuffle_time=0*MPI_Wtime(), reshuffle_time=0*MPI_Wtime(), total_time=0*MPI_Wtime();

  if(VERBOSE>=2) PCOUT<<"INPUT:"<<std::endl;
  if(VERBOSE>=2){
    cudaMemcpy(data_cpu, data, T_plan->alloc_local, cudaMemcpyDeviceToHost);
    for(int h=0;h<howmany;h++)
      for(int id=0;id<nprocs;++id){
        if(procid==id)
          for(int i=0;i<local_n0;i++){
            std::cout<<std::endl;
            for(int j=0;j<N[1];j++){
              std::cout<<'\t'<<data_cpu[h*idist+(i*N[1]+j)*n_tuples];
            }
          }
        std::cout<<'\n';
        MPI_Barrier(T_plan->comm);
      }
  }
  //PCOUT<<" ============================================= "<<std::endl;
  //PCOUT<<" ==============   Local Transpose============= "<<std::endl;
  //PCOUT<<" ============================================= "<<std::endl;
  int ptr=0;
  shuffle_time-=MPI_Wtime();


  if(nprocs==1 && Flags[0]==1 && Flags[1]==1){
#pragma omp parallel for
    for(int h=0;h<howmany;h++)
      local_transpose_cuda(local_n1,N[0],n_tuples,&data[h*idist] );
  }
  if(nprocs==1 && Flags[0]==0 && Flags[1]==0){
#pragma omp parallel for
    for(int h=0;h<howmany;h++)
      local_transpose_cuda(N[0],N[1],n_tuples,&data[h*idist] );
  }
  if(nprocs==1){ // Transpose is done!
    shuffle_time+=MPI_Wtime();
    timings[0]+=MPI_Wtime();
    timings[0]+=shuffle_time;
    timings[1]+=shuffle_time;
    timings[2]+=0;
    timings[3]+=0;
    MPI_Barrier(T_plan->comm);
    return;
  }

  // Flags[1]=1 data_d --Th--> send_recv_d  --memcpy-->  data_cpu  --alltoall--> send_recv_cpu --memcpy--> data_d
  // Flags[1]=0 data_d --Th--> send_recv_d  --memcpy-->  data_cpu  --alltoall--> send_recv_cpu --Th--> data_d
  local_transpose_col_cuda(local_n0,nprocs_1,n_tuples*T_plan->local_n1_proc[0], n_tuples*T_plan->last_local_n1,data,send_recv_d );

  shuffle_time+=MPI_Wtime();
  ptr=0;
  if(VERBOSE>=2) PCOUT<<"Local Transpose:"<<std::endl;
  if(VERBOSE>=2){
    cudaMemcpy(data_cpu,send_recv_d, T_plan->alloc_local, cudaMemcpyDeviceToHost);
    for(int h=0;h<howmany;h++)
      for(int id=0;id<nprocs;++id){
        if(procid==id)
          for(int i=0;i<N[1];i++){
            std::cout<<std::endl;
            for(int j=0;j<local_n0;j++){
              std::cout<<'\t'<<data_cpu[ptr];//data[h*idist+(i*local_n0+j)*n_tuples];
              ptr+=n_tuples;
            }
          }
        std::cout<<'\n';
        MPI_Barrier(T_plan->comm);
      }
  }

  //PCOUT<<" ============================================= "<<std::endl;
  //PCOUT<<" ==============   MPIALLTOALL  =============== "<<std::endl;
  //PCOUT<<" ============================================= "<<std::endl;

  int* scount_proc=  T_plan->scount_proc;
  int* rcount_proc=  T_plan->rcount_proc;
  int* soffset_proc= T_plan->soffset_proc;
  int* roffset_proc= T_plan->roffset_proc;

  MPI_Barrier(T_plan->comm);

  //PCOUT<<"nprocs_0= "<<nprocs_0<<" nprocs_1= "<<nprocs_1<<std::endl;
  comm_time-=MPI_Wtime();

  int soffset=0,roffset=0;
  MPI_Status ierr;
  MPI_Request * s_request= new MPI_Request[nprocs];
  MPI_Request * request= new MPI_Request[nprocs];
#pragma omp parallel for
  for (int proc=0;proc<nprocs;++proc){
    request[proc]=MPI_REQUEST_NULL;
    s_request[proc]=MPI_REQUEST_NULL;
  }
  double *s_buf, *r_buf;
  s_buf=data_cpu; r_buf=send_recv_cpu;

  // SEND
  // Flags[1]=1 data_d --Th--> send_recv_d  --memcpy-->  data_cpu  --alltoall--> send_recv_cpu --memcpy--> data_d
  // Flags[1]=0 data_d --Th--> send_recv_d  --memcpy-->  data_cpu  --alltoall--> send_recv_cpu --Th--> data_d
  cudaMemcpy(data_cpu, send_recv_d, T_plan->alloc_local, cudaMemcpyDeviceToHost);
  for (int proc=0;proc<nprocs;++proc){
    if(proc!=procid){
      soffset=soffset_proc[proc];
      roffset=roffset_proc[proc];
      MPI_Isend(&s_buf[soffset],scount_proc[proc],MPI_DOUBLE,proc, tag,
          T_plan->comm, &s_request[proc]);
      MPI_Irecv(&r_buf[roffset],rcount_proc[proc],MPI_DOUBLE, proc,
          tag, T_plan->comm, &request[proc]);
    }
  }
  // Copy Your own part. See the note below for the if condition
  soffset=soffset_proc[procid];//aoffset_proc[proc];//proc*count_proc[proc];
  roffset=roffset_proc[procid];
  for(int h=0;h<howmany;h++)
    memcpy(&r_buf[h*odist+roffset],&s_buf[h*idist+soffset],sizeof(double)*scount_proc[procid]);


  // If the output is to be transposed locally, then everything is done in sendrecv. just copy it
  // Otherwise you have to perform the copy via the multiple stride transpose
  for (int proc=0;proc<nprocs;++proc){
    MPI_Wait(&request[proc], &ierr);
    MPI_Wait(&s_request[proc], &ierr);
  }

  // Flags[1]=1 data_d --Th--> send_recv_d  --memcpy-->  data_cpu  --alltoall--> send_recv_cpu --memcpy--> data_d
  if(Flags[1]==1)
    cudaMemcpy(data, r_buf, T_plan->alloc_local, cudaMemcpyHostToDevice);
  else
    cudaMemcpy(send_recv_d, r_buf, T_plan->alloc_local, cudaMemcpyHostToDevice);
  comm_time+=MPI_Wtime();

  ptr=0;
  if(VERBOSE>=2) PCOUT<<"MPIAlltoAll:"<<std::endl;
  if(VERBOSE>=2){
    cudaMemcpy(data_cpu,send_recv_d, T_plan->alloc_local, cudaMemcpyDeviceToHost);
    for(int h=0;h<howmany;h++)
      for(int id=0;id<nprocs;++id){
        if(procid==id)
          for(int i=0;i<local_n1;i++){
            std::cout<<std::endl;
            for(int j=0;j<N[0];j++){
              std::cout<<'\t'<<data_cpu[ptr];//send_recv[h*odist+(i*N[0]+j)*n_tuples];//<<","<<send_recv[(i*N[0]+j)*n_tuples+1];
              ptr+=n_tuples;
            }
          }
        std::cout<<'\n';
        MPI_Barrier(T_plan->comm);
      }
  }
  //PCOUT<<" ============================================= "<<std::endl;
  //PCOUT<<" ============== 2nd Local Trnaspose ========== "<<std::endl;
  //PCOUT<<" ============================================= "<<std::endl;
  reshuffle_time-=MPI_Wtime();
  ptr=0;
  if(Flags[1]==0)
    local_transpose_cuda(N[0],local_n1,n_tuples,send_recv_d,data );




  if(VERBOSE>=2) PCOUT<<"2nd Transpose"<<std::endl;
  if(VERBOSE>=2){
    cudaMemcpy(data_cpu, data, T_plan->alloc_local, cudaMemcpyDeviceToHost);
    for(int h=0;h<howmany;h++)
      for(int id=0;id<nprocs_1;++id){
        if(procid==id)
          for(int i=0;i<local_n1;i++){
            std::cout<<std::endl;
            for(int j=0;j<N[0];j++){
              std::cout<<'\t'<<data_cpu[h*odist+(i*N[0]+j)*n_tuples];
            }
          }
        std::cout<<'\n';
        MPI_Barrier(T_plan->comm);
      }
  }

  reshuffle_time+=MPI_Wtime();
  delete [] request;
  delete [] s_request;


  if(VERBOSE>=1){
    PCOUT<<"Shuffle Time= "<<shuffle_time<<std::endl;
    PCOUT<<"Alltoall Time= "<<comm_time<<std::endl;
    PCOUT<<"Reshuffle Time= "<<reshuffle_time<<std::endl;
    PCOUT<<"Total Time= "<<(shuffle_time+comm_time+reshuffle_time)<<std::endl;
  }
  timings[0]+=MPI_Wtime();//timings[0]+=shuffle_time+comm_time+reshuffle_time;
  timings[1]+=shuffle_time;
  timings[2]+=comm_time;
  timings[3]+=reshuffle_time;
  return;


}

void fast_transpose_cuda_v1_2(T_Plan_gpu* T_plan, double * data, double *timings, unsigned flags, int howmany, int tag ){

  std::bitset<8> Flags(flags); // 1 Transposed in, 2 Transposed out
  if(Flags[1]==1 && Flags[0]==0 && T_plan->nprocs==1){ // If Flags==Transposed_Out return
    MPI_Barrier(T_plan->comm);
    return;
  }
  if(Flags[0]==1){ // If Flags==Transposed_In This function can not handle it, call other versions
    transpose_cuda_v5(T_plan,(double*)data,timings,flags,howmany,tag);
    MPI_Barrier(T_plan->comm);
    return;
  }
  timings[0]-=MPI_Wtime();
  int nprocs, procid;
  nprocs=T_plan->nprocs;
  procid=T_plan->procid;
  int nprocs_1=T_plan->nprocs_1;
  ptrdiff_t  *N=T_plan->N;

  double * data_cpu=T_plan->buffer;  //pinned
  double * send_recv_cpu = T_plan->buffer_2;//pinned
  double * send_recv_d = T_plan->buffer_d;


  ptrdiff_t local_n0=T_plan->local_n0;
  ptrdiff_t local_n1=T_plan->local_n1;
  ptrdiff_t n_tuples=T_plan->n_tuples;

  int idist=N[1]*local_n0*n_tuples;
  int odist=N[0]*local_n1*n_tuples;

  double comm_time=0*MPI_Wtime(), shuffle_time=0*MPI_Wtime(), reshuffle_time=0*MPI_Wtime(), total_time=0*MPI_Wtime();

  if(VERBOSE>=2) PCOUT<<"INPUT:"<<std::endl;
  if(VERBOSE>=2){
    cudaMemcpy(data_cpu, data, T_plan->alloc_local, cudaMemcpyDeviceToHost);
    for(int h=0;h<howmany;h++)
      for(int id=0;id<nprocs;++id){
        if(procid==id)
          for(int i=0;i<local_n0;i++){
            std::cout<<std::endl;
            for(int j=0;j<N[1];j++){
              std::cout<<'\t'<<data_cpu[h*idist+(i*N[1]+j)*n_tuples];
            }
          }
        std::cout<<'\n';
        MPI_Barrier(T_plan->comm);
      }
  }
  //PCOUT<<" ============================================= "<<std::endl;
  //PCOUT<<" ==============   Local Transpose============= "<<std::endl;
  //PCOUT<<" ============================================= "<<std::endl;
  int ptr=0;
  shuffle_time-=MPI_Wtime();


  if(nprocs==1 && Flags[0]==1 && Flags[1]==1){
#pragma omp parallel for
    for(int h=0;h<howmany;h++)
      local_transpose_cuda(local_n1,N[0],n_tuples,&data[h*idist] );
  }
  if(nprocs==1 && Flags[0]==0 && Flags[1]==0){
#pragma omp parallel for
    for(int h=0;h<howmany;h++)
      local_transpose_cuda(N[0],N[1],n_tuples,&data[h*idist] );
  }
  if(nprocs==1){ // Transpose is done!
    shuffle_time+=MPI_Wtime();
    timings[0]+=MPI_Wtime();
    timings[0]+=shuffle_time;
    timings[1]+=shuffle_time;
    timings[2]+=0;
    timings[3]+=0;
    MPI_Barrier(T_plan->comm);
    return;
  }

  // Flags[1]=1 data_d --Th--> send_recv_d  --memcpy-->  data_cpu  --alltoall--> send_recv_cpu --memcpy--> data_d
  // Flags[1]=0 data_d --Th--> send_recv_d  --memcpy-->  data_cpu  --alltoall--> send_recv_cpu --Th--> data_d
  local_transpose_col_cuda(local_n0,nprocs_1,n_tuples*T_plan->local_n1_proc[0], n_tuples*T_plan->last_local_n1,data,send_recv_d );

  shuffle_time+=MPI_Wtime();
  ptr=0;
  if(VERBOSE>=2) PCOUT<<"Local Transpose:"<<std::endl;
  if(VERBOSE>=2){
    cudaMemcpy(data_cpu,send_recv_d, T_plan->alloc_local, cudaMemcpyDeviceToHost);
    for(int h=0;h<howmany;h++)
      for(int id=0;id<nprocs;++id){
        if(procid==id)
          for(int i=0;i<N[1];i++){
            std::cout<<std::endl;
            for(int j=0;j<local_n0;j++){
              std::cout<<'\t'<<data_cpu[ptr];//data[h*idist+(i*local_n0+j)*n_tuples];
              ptr+=n_tuples;
            }
          }
        std::cout<<'\n';
        MPI_Barrier(T_plan->comm);
      }
  }

  //PCOUT<<" ============================================= "<<std::endl;
  //PCOUT<<" ==============   MPIALLTOALL  =============== "<<std::endl;
  //PCOUT<<" ============================================= "<<std::endl;

  int* scount_proc=  T_plan->scount_proc;
  int* rcount_proc=  T_plan->rcount_proc;
  int* soffset_proc= T_plan->soffset_proc;
  int* roffset_proc= T_plan->roffset_proc;

  MPI_Barrier(T_plan->comm);

  //PCOUT<<"nprocs_0= "<<nprocs_0<<" nprocs_1= "<<nprocs_1<<std::endl;
  comm_time-=MPI_Wtime();

  int soffset=0,roffset=0;
  MPI_Status ierr;
  MPI_Request * s_request= new MPI_Request[nprocs];
  MPI_Request * request= new MPI_Request[nprocs];
  int flag[nprocs],color[nprocs];
  memset(flag,0,sizeof(int)*nprocs);
  memset(color,0,sizeof(int)*nprocs);
  int counter=1;
#pragma omp parallel for
  for (int proc=0;proc<nprocs;++proc){
    request[proc]=MPI_REQUEST_NULL;
    s_request[proc]=MPI_REQUEST_NULL;
  }
  double *s_buf, *r_buf;
  s_buf=data_cpu; r_buf=send_recv_cpu;
  double * r_buf_d;
  if(Flags[1]==1)
    r_buf_d=data;
  else
    r_buf_d=send_recv_d;

  // SEND
  // Flags[1]=1 data_d --Th--> send_recv_d  --memcpy-->  data_cpu  --alltoall--> send_recv_cpu --memcpy--> data_d
  // Flags[1]=0 data_d --Th--> send_recv_d  --memcpy-->  data_cpu  --alltoall--> send_recv_cpu --memcpy-->send_recv_d --Th--> data_d


  for (int proc=0;proc<nprocs;++proc){
    if(proc!=procid){
      roffset=roffset_proc[proc];
      MPI_Irecv(&r_buf[roffset],rcount_proc[proc], MPI_DOUBLE, proc,
          tag, T_plan->comm, &request[proc]);
    }
  }
  // SEND
  //cudaMemcpy(data_cpu, send_recv_d, T_plan->alloc_local, cudaMemcpyDeviceToHost);
  for (int proc=0;proc<nprocs;++proc){
    if(proc!=procid){
      soffset=soffset_proc[proc];
      cudaMemcpy(&s_buf[soffset], &send_recv_d[soffset],sizeof(double)*scount_proc[proc], cudaMemcpyDeviceToHost);
      MPI_Isend(&s_buf[soffset],scount_proc[proc], MPI_DOUBLE,proc, tag,
          T_plan->comm, &s_request[proc]);

    }
    else{  // copy your own part directly
      //cudaMemcpy(&r_buf_d[roffset_proc[procid]], &send_recv_d[soffset_proc[procid]],sizeof(double)*scount_proc[procid], cudaMemcpyDeviceToDevice);
      // Note that because if Flags[1]=0 the send_d and recv_d are the same, D2D should not be used
      // Otherwise it will corrupt the data in the unbalanced cases
      cudaMemcpy(&r_buf[roffset_proc[procid]], &send_recv_d[soffset_proc[procid]],sizeof(double)*scount_proc[procid], cudaMemcpyDeviceToHost);
      flag[procid]=1;
    }
  }
  while(counter!=nprocs+1){

    for (int proc=0;proc<nprocs;++proc){
      MPI_Test(&request[proc], &flag[proc],&ierr);
      if(flag[proc]==1 && color[proc]==0){
        cudaMemcpyAsync(&r_buf_d[roffset_proc[proc]],&r_buf[roffset_proc[proc]],sizeof(double)*rcount_proc[proc],cudaMemcpyHostToDevice);
        color[proc]=1;
        counter+=1;
      }
    }

  }
  cudaDeviceSynchronize();
  comm_time+=MPI_Wtime();








  ptr=0;
  if(VERBOSE>=2) PCOUT<<"MPIAlltoAll:"<<std::endl;
  if(VERBOSE>=2){
    cudaMemcpy(data_cpu,send_recv_d, T_plan->alloc_local, cudaMemcpyDeviceToHost);
    for(int h=0;h<howmany;h++)
      for(int id=0;id<nprocs;++id){
        if(procid==id)
          for(int i=0;i<local_n1;i++){
            std::cout<<std::endl;
            for(int j=0;j<N[0];j++){
              std::cout<<'\t'<<data_cpu[ptr];//send_recv[h*odist+(i*N[0]+j)*n_tuples];//<<","<<send_recv[(i*N[0]+j)*n_tuples+1];
              ptr+=n_tuples;
            }
          }
        std::cout<<'\n';
        MPI_Barrier(T_plan->comm);
      }
  }
  //PCOUT<<" ============================================= "<<std::endl;
  //PCOUT<<" ============== 2nd Local Trnaspose ========== "<<std::endl;
  //PCOUT<<" ============================================= "<<std::endl;
  reshuffle_time-=MPI_Wtime();
  ptr=0;
  if(Flags[1]==0)
    local_transpose_cuda(N[0],local_n1,n_tuples,send_recv_d,data );




  if(VERBOSE>=2) PCOUT<<"2nd Transpose"<<std::endl;
  if(VERBOSE>=2){
    cudaMemcpy(data_cpu, data, T_plan->alloc_local, cudaMemcpyDeviceToHost);
    for(int h=0;h<howmany;h++)
      for(int id=0;id<nprocs_1;++id){
        if(procid==id)
          for(int i=0;i<local_n1;i++){
            std::cout<<std::endl;
            for(int j=0;j<N[0];j++){
              std::cout<<'\t'<<data_cpu[h*odist+(i*N[0]+j)*n_tuples];
            }
          }
        std::cout<<'\n';
        MPI_Barrier(T_plan->comm);
      }
  }

  reshuffle_time+=MPI_Wtime();
  delete [] request;
  delete [] s_request;


  if(VERBOSE>=1){
    PCOUT<<"Shuffle Time= "<<shuffle_time<<std::endl;
    PCOUT<<"Alltoall Time= "<<comm_time<<std::endl;
    PCOUT<<"Reshuffle Time= "<<reshuffle_time<<std::endl;
    PCOUT<<"Total Time= "<<(shuffle_time+comm_time+reshuffle_time)<<std::endl;
  }
  timings[0]+=MPI_Wtime();//timings[0]+=shuffle_time+comm_time+reshuffle_time;
  timings[1]+=shuffle_time;
  timings[2]+=comm_time;
  timings[3]+=reshuffle_time;
  return;


}
void fast_transpose_cuda_v1_3(T_Plan_gpu* T_plan, double * data, double *timings, unsigned flags, int howmany, int tag ){

  std::bitset<8> Flags(flags); // 1 Transposed in, 2 Transposed out
  if(Flags[1]==1 && Flags[0]==0 && T_plan->nprocs==1){ // If Flags==Transposed_Out return
    MPI_Barrier(T_plan->comm);
    return;
  }
  if(Flags[0]==1){ // If Flags==Transposed_In This function can not handle it, call other versions
    transpose_cuda_v5(T_plan,(double*)data,timings,flags,howmany,tag);
    MPI_Barrier(T_plan->comm);
    return;
  }
  timings[0]-=MPI_Wtime();
  int nprocs, procid;
  nprocs=T_plan->nprocs;
  procid=T_plan->procid;
  int nprocs_1=T_plan->nprocs_1;
  ptrdiff_t  *N=T_plan->N;

  double * data_cpu=T_plan->buffer;  //pinned
  double * send_recv_cpu = T_plan->buffer_2;//pinned
  double * send_recv_d = T_plan->buffer_d;


  ptrdiff_t local_n0=T_plan->local_n0;
  ptrdiff_t local_n1=T_plan->local_n1;
  ptrdiff_t n_tuples=T_plan->n_tuples;

  int idist=N[1]*local_n0*n_tuples;
  int odist=N[0]*local_n1*n_tuples;

  double comm_time=0*MPI_Wtime(), shuffle_time=0*MPI_Wtime(), reshuffle_time=0*MPI_Wtime(), total_time=0*MPI_Wtime();

  if(VERBOSE>=2) PCOUT<<"INPUT:"<<std::endl;
  if(VERBOSE>=2){
    cudaMemcpy(data_cpu, data, T_plan->alloc_local, cudaMemcpyDeviceToHost);
    for(int h=0;h<howmany;h++)
      for(int id=0;id<nprocs;++id){
        if(procid==id)
          for(int i=0;i<local_n0;i++){
            std::cout<<std::endl;
            for(int j=0;j<N[1];j++){
              std::cout<<'\t'<<data_cpu[h*idist+(i*N[1]+j)*n_tuples];
            }
          }
        std::cout<<'\n';
        MPI_Barrier(T_plan->comm);
      }
  }
  //PCOUT<<" ============================================= "<<std::endl;
  //PCOUT<<" ==============   Local Transpose============= "<<std::endl;
  //PCOUT<<" ============================================= "<<std::endl;
  int ptr=0;
  shuffle_time-=MPI_Wtime();


  if(nprocs==1 && Flags[0]==1 && Flags[1]==1){
#pragma omp parallel for
    for(int h=0;h<howmany;h++)
      local_transpose_cuda(local_n1,N[0],n_tuples,&data[h*idist] );
  }
  if(nprocs==1 && Flags[0]==0 && Flags[1]==0){
#pragma omp parallel for
    for(int h=0;h<howmany;h++)
      local_transpose_cuda(N[0],N[1],n_tuples,&data[h*idist] );
  }
  if(nprocs==1){ // Transpose is done!
    shuffle_time+=MPI_Wtime();
    timings[0]+=MPI_Wtime();
    timings[0]+=shuffle_time;
    timings[1]+=shuffle_time;
    timings[2]+=0;
    timings[3]+=0;
    MPI_Barrier(T_plan->comm);
    return;
  }

  // Flags[1]=1 data_d --Th--> send_recv_d  --memcpy-->  data_cpu  --alltoall--> send_recv_cpu --memcpy--> data_d
  // Flags[1]=0 data_d --Th--> send_recv_d  --memcpy-->  data_cpu  --alltoall--> send_recv_cpu --Th--> data_d
  local_transpose_col_cuda(local_n0,nprocs_1,n_tuples*T_plan->local_n1_proc[0], n_tuples*T_plan->last_local_n1,data,send_recv_d );

  shuffle_time+=MPI_Wtime();
  ptr=0;
  if(VERBOSE>=2) PCOUT<<"Local Transpose:"<<std::endl;
  if(VERBOSE>=2){
    cudaMemcpy(data_cpu,send_recv_d, T_plan->alloc_local, cudaMemcpyDeviceToHost);
    for(int h=0;h<howmany;h++)
      for(int id=0;id<nprocs;++id){
        if(procid==id)
          for(int i=0;i<N[1];i++){
            std::cout<<std::endl;
            for(int j=0;j<local_n0;j++){
              std::cout<<'\t'<<data_cpu[ptr];//data[h*idist+(i*local_n0+j)*n_tuples];
              ptr+=n_tuples;
            }
          }
        std::cout<<'\n';
        MPI_Barrier(T_plan->comm);
      }
  }

  //PCOUT<<" ============================================= "<<std::endl;
  //PCOUT<<" ==============   MPIALLTOALL  =============== "<<std::endl;
  //PCOUT<<" ============================================= "<<std::endl;

  int* scount_proc=  T_plan->scount_proc;
  int* rcount_proc=  T_plan->rcount_proc;
  int* soffset_proc= T_plan->soffset_proc;
  int* roffset_proc= T_plan->roffset_proc;

  MPI_Barrier(T_plan->comm);

  //PCOUT<<"nprocs_0= "<<nprocs_0<<" nprocs_1= "<<nprocs_1<<std::endl;
  comm_time-=MPI_Wtime();

  int soffset=0,roffset=0;
  MPI_Status ierr;
  MPI_Request * s_request= new MPI_Request[nprocs];
  MPI_Request * request= new MPI_Request[nprocs];
  int flag[nprocs],color[nprocs];
  memset(flag,0,sizeof(int)*nprocs);
  memset(color,0,sizeof(int)*nprocs);
#pragma omp parallel for
  for (int proc=0;proc<nprocs;++proc){
    request[proc]=MPI_REQUEST_NULL;
    s_request[proc]=MPI_REQUEST_NULL;
  }
  double *s_buf, *r_buf;
  s_buf=data_cpu; r_buf=send_recv_cpu;
  double * r_buf_d;
  if(Flags[1]==1)
    r_buf_d=data;
  else
    r_buf_d=send_recv_d;

  // SEND
  // Flags[1]=1 data_d --Th--> send_recv_d  --memcpy-->  data_cpu  --alltoall--> send_recv_cpu --memcpy--> data_d
  // Flags[1]=0 data_d --Th--> send_recv_d  --memcpy-->  data_cpu  --alltoall--> send_recv_cpu --Th--> data_d


  for (int proc=0;proc<nprocs;++proc){
    if(proc!=procid){
      soffset=soffset_proc[proc];
      roffset=roffset_proc[proc];
      cudaMemcpy(&s_buf[soffset], &send_recv_d[soffset],sizeof(double)*scount_proc[proc], cudaMemcpyDeviceToHost);
      MPI_Sendrecv(&s_buf[soffset],scount_proc[proc], MPI_DOUBLE,
          proc, tag,
          &r_buf[roffset],rcount_proc[proc], MPI_DOUBLE,
          proc, tag,
          T_plan->comm,&ierr);

    }
    else{  // copy your own part directly
      //cudaMemcpyAsync(&r_buf_d[roffset_proc[procid]], &send_recv_d[soffset_proc[procid]],sizeof(double)*scount_proc[procid], cudaMemcpyDeviceToDevice);
      // Note that because if Flags[1]=0 the send_d and recv_d are the same, D2D should not be used
      // Otherwise it will corrupt the data in the unbalanced cases
      cudaMemcpy(&r_buf[roffset_proc[procid]], &send_recv_d[soffset_proc[procid]],sizeof(double)*scount_proc[procid], cudaMemcpyDeviceToHost);
      flag[procid]=1;
    }
    if(Flags[1]==1)
      cudaMemcpyAsync(&r_buf_d[roffset_proc[proc]],&r_buf[roffset_proc[proc]],sizeof(double)*rcount_proc[proc],cudaMemcpyHostToDevice);
  }
  if(Flags[1]==0)
    cudaMemcpy(r_buf_d,r_buf,T_plan->alloc_local,cudaMemcpyHostToDevice);
  cudaDeviceSynchronize();
  comm_time+=MPI_Wtime();


  ptr=0;
  if(VERBOSE>=2) PCOUT<<"MPIAlltoAll:"<<std::endl;
  if(VERBOSE>=2){
    cudaMemcpy(data_cpu,send_recv_d, T_plan->alloc_local, cudaMemcpyDeviceToHost);
    for(int h=0;h<howmany;h++)
      for(int id=0;id<nprocs;++id){
        if(procid==id)
          for(int i=0;i<local_n1;i++){
            std::cout<<std::endl;
            for(int j=0;j<N[0];j++){
              std::cout<<'\t'<<data_cpu[ptr];//send_recv[h*odist+(i*N[0]+j)*n_tuples];//<<","<<send_recv[(i*N[0]+j)*n_tuples+1];
              ptr+=n_tuples;
            }
          }
        std::cout<<'\n';
        MPI_Barrier(T_plan->comm);
      }
  }
  //PCOUT<<" ============================================= "<<std::endl;
  //PCOUT<<" ============== 2nd Local Trnaspose ========== "<<std::endl;
  //PCOUT<<" ============================================= "<<std::endl;
  reshuffle_time-=MPI_Wtime();
  ptr=0;
  if(Flags[1]==0)
    local_transpose_cuda(N[0],local_n1,n_tuples,send_recv_d,data );




  if(VERBOSE>=2) PCOUT<<"2nd Transpose"<<std::endl;
  if(VERBOSE>=2){
    cudaMemcpy(data_cpu, data, T_plan->alloc_local, cudaMemcpyDeviceToHost);
    for(int h=0;h<howmany;h++)
      for(int id=0;id<nprocs_1;++id){
        if(procid==id)
          for(int i=0;i<local_n1;i++){
            std::cout<<std::endl;
            for(int j=0;j<N[0];j++){
              std::cout<<'\t'<<data_cpu[h*odist+(i*N[0]+j)*n_tuples];
            }
          }
        std::cout<<'\n';
        MPI_Barrier(T_plan->comm);
      }
  }

  reshuffle_time+=MPI_Wtime();
  delete [] request;
  delete [] s_request;


  if(VERBOSE>=1){
    PCOUT<<"Shuffle Time= "<<shuffle_time<<std::endl;
    PCOUT<<"Alltoall Time= "<<comm_time<<std::endl;
    PCOUT<<"Reshuffle Time= "<<reshuffle_time<<std::endl;
    PCOUT<<"Total Time= "<<(shuffle_time+comm_time+reshuffle_time)<<std::endl;
  }
  timings[0]+=MPI_Wtime();//timings[0]+=shuffle_time+comm_time+reshuffle_time;
  timings[1]+=shuffle_time;
  timings[2]+=comm_time;
  timings[3]+=reshuffle_time;
  return;


}




void fast_transpose_cuda_v2(T_Plan_gpu* T_plan, double * data, double *timings, unsigned flags, int howmany, int tag ){
  // This function handles cases where howmany=1 (it is more optimal)


  std::bitset<8> Flags(flags); // 1 Transposed in, 2 Transposed out
  if(Flags[1]==1 && Flags[0]==0 && T_plan->nprocs==1){ // If Flags==Transposed_Out return
    MPI_Barrier(T_plan->comm);
    return;
  }
  if(Flags[0]==1){ // If Flags==Transposed_In This function can not handle it, call other versions
    transpose_cuda_v6(T_plan,(double*)data,timings,flags,howmany,tag);
    MPI_Barrier(T_plan->comm);
    return;
  }
  timings[0]-=MPI_Wtime();
  int nprocs, procid;
  nprocs=T_plan->nprocs;
  procid=T_plan->procid;
  int nprocs_1=T_plan->nprocs_1;
  ptrdiff_t  *N=T_plan->N;

  double * data_cpu=T_plan->buffer;  //pinned
  double * send_recv_cpu = T_plan->buffer_2;//pinned
  double * send_recv_d = T_plan->buffer_d;


  ptrdiff_t local_n0=T_plan->local_n0;
  ptrdiff_t local_n1=T_plan->local_n1;
  ptrdiff_t n_tuples=T_plan->n_tuples;

  int idist=N[1]*local_n0*n_tuples;
  int odist=N[0]*local_n1*n_tuples;

  double comm_time=0*MPI_Wtime(), shuffle_time=0*MPI_Wtime(), reshuffle_time=0*MPI_Wtime(), total_time=0*MPI_Wtime();

  if(VERBOSE>=2) PCOUT<<"INPUT:"<<std::endl;
  if(VERBOSE>=2){
    cudaMemcpy(data_cpu, data, T_plan->alloc_local, cudaMemcpyDeviceToHost);
    for(int h=0;h<howmany;h++)
      for(int id=0;id<nprocs;++id){
        if(procid==id)
          for(int i=0;i<local_n0;i++){
            std::cout<<std::endl;
            for(int j=0;j<N[1];j++){
              std::cout<<'\t'<<data_cpu[h*idist+(i*N[1]+j)*n_tuples];
            }
          }
        std::cout<<'\n';
        MPI_Barrier(T_plan->comm);
      }
  }
  //PCOUT<<" ============================================= "<<std::endl;
  //PCOUT<<" ==============   Local Transpose============= "<<std::endl;
  //PCOUT<<" ============================================= "<<std::endl;
  int ptr=0;
  shuffle_time-=MPI_Wtime();


  if(nprocs==1 && Flags[0]==1 && Flags[1]==1){
#pragma omp parallel for
    for(int h=0;h<howmany;h++)
      local_transpose_cuda(local_n1,N[0],n_tuples,&data[h*idist] );
  }
  if(nprocs==1 && Flags[0]==0 && Flags[1]==0){
#pragma omp parallel for
    for(int h=0;h<howmany;h++)
      local_transpose_cuda(N[0],N[1],n_tuples,&data[h*idist] );
  }
  if(nprocs==1){ // Transpose is done!
    shuffle_time+=MPI_Wtime();
    timings[0]+=MPI_Wtime();
    timings[0]+=shuffle_time;
    timings[1]+=shuffle_time;
    timings[2]+=0;
    timings[3]+=0;
    MPI_Barrier(T_plan->comm);
    return;
  }

  // Flags[1]=1 data_d --Th--> send_recv_d  --memcpy-->  data_cpu  --alltoall--> send_recv_cpu --memcpy--> data_d
  // Flags[1]=0 data_d --Th--> send_recv_d  --memcpy-->  data_cpu  --alltoall--> send_recv_cpu --Th--> data_d
  local_transpose_col_cuda(local_n0,nprocs_1,n_tuples*T_plan->local_n1_proc[0], n_tuples*T_plan->last_local_n1,data,send_recv_d );

  shuffle_time+=MPI_Wtime();
  ptr=0;
  if(VERBOSE>=2) PCOUT<<"Local Transpose:"<<std::endl;
  if(VERBOSE>=2){
    cudaMemcpy(data_cpu,send_recv_d, T_plan->alloc_local, cudaMemcpyDeviceToHost);
    for(int h=0;h<howmany;h++)
      for(int id=0;id<nprocs;++id){
        if(procid==id)
          for(int i=0;i<N[1];i++){
            std::cout<<std::endl;
            for(int j=0;j<local_n0;j++){
              std::cout<<'\t'<<data_cpu[ptr];//data[h*idist+(i*local_n0+j)*n_tuples];
              ptr+=n_tuples;
            }
          }
        std::cout<<'\n';
        MPI_Barrier(T_plan->comm);
      }
  }

  //PCOUT<<" ============================================= "<<std::endl;
  //PCOUT<<" ==============   MPIALLTOALL  =============== "<<std::endl;
  //PCOUT<<" ============================================= "<<std::endl;

  int* scount_proc_f=  T_plan->scount_proc_f;
  int* rcount_proc_f=  T_plan->rcount_proc_f;
  int* soffset_proc_f= T_plan->soffset_proc_f;
  int* roffset_proc_f= T_plan->roffset_proc_f;

  MPI_Barrier(T_plan->comm);

  //PCOUT<<"nprocs_0= "<<nprocs_0<<" nprocs_1= "<<nprocs_1<<std::endl;
  comm_time-=MPI_Wtime();

  MPI_Request * s_request= new MPI_Request[nprocs];
  MPI_Request * request= new MPI_Request[nprocs];
#pragma omp parallel for
  for (int proc=0;proc<nprocs;++proc){
    request[proc]=MPI_REQUEST_NULL;
    s_request[proc]=MPI_REQUEST_NULL;
  }
  double *s_buf, *r_buf;
  s_buf=data_cpu; r_buf=send_recv_cpu;

  // SEND
  // Flags[1]=1 data_d --Th--> send_recv_d  --memcpy-->  data_cpu  --alltoall--> send_recv_cpu --memcpy--> data_d
  // Flags[1]=0 data_d --Th--> send_recv_d  --memcpy-->  data_cpu  --alltoall--> send_recv_cpu --Th--> data_d
  cudaMemcpy(data_cpu, send_recv_d, T_plan->alloc_local, cudaMemcpyDeviceToHost);
  if(T_plan->is_evenly_distributed==0)
    MPI_Alltoallv(s_buf,scount_proc_f,
        soffset_proc_f, MPI_DOUBLE,r_buf,
        rcount_proc_f,roffset_proc_f, MPI_DOUBLE,
        T_plan->comm);
  else
    MPI_Alltoall(s_buf, scount_proc_f[0], MPI_DOUBLE,
        r_buf, rcount_proc_f[0], MPI_DOUBLE,
        T_plan->comm);


  // Flags[1]=1 data_d --Th--> send_recv_d  --memcpy-->  data_cpu  --alltoall--> send_recv_cpu --memcpy--> data_d
  if(Flags[1]==1)
    cudaMemcpy(data, r_buf, T_plan->alloc_local, cudaMemcpyHostToDevice);
  else
    cudaMemcpy(send_recv_d, r_buf, T_plan->alloc_local, cudaMemcpyHostToDevice);
  comm_time+=MPI_Wtime();

  ptr=0;
  if(VERBOSE>=2) PCOUT<<"MPIAlltoAll:"<<std::endl;
  if(VERBOSE>=2){
    cudaMemcpy(data_cpu,send_recv_d, T_plan->alloc_local, cudaMemcpyDeviceToHost);
    for(int h=0;h<howmany;h++)
      for(int id=0;id<nprocs;++id){
        if(procid==id)
          for(int i=0;i<local_n1;i++){
            std::cout<<std::endl;
            for(int j=0;j<N[0];j++){
              std::cout<<'\t'<<data_cpu[ptr];//send_recv[h*odist+(i*N[0]+j)*n_tuples];//<<","<<send_recv[(i*N[0]+j)*n_tuples+1];
              ptr+=n_tuples;
            }
          }
        std::cout<<'\n';
        MPI_Barrier(T_plan->comm);
      }
  }
  //PCOUT<<" ============================================= "<<std::endl;
  //PCOUT<<" ============== 2nd Local Trnaspose ========== "<<std::endl;
  //PCOUT<<" ============================================= "<<std::endl;
  reshuffle_time-=MPI_Wtime();
  ptr=0;
  if(Flags[1]==0)
    local_transpose_cuda(N[0],local_n1,n_tuples,send_recv_d,data );




  if(VERBOSE>=2) PCOUT<<"2nd Transpose"<<std::endl;
  if(VERBOSE>=2){
    cudaMemcpy(data_cpu, data, T_plan->alloc_local, cudaMemcpyDeviceToHost);
    for(int h=0;h<howmany;h++)
      for(int id=0;id<nprocs_1;++id){
        if(procid==id)
          for(int i=0;i<local_n1;i++){
            std::cout<<std::endl;
            for(int j=0;j<N[0];j++){
              std::cout<<'\t'<<data_cpu[h*odist+(i*N[0]+j)*n_tuples];
            }
          }
        std::cout<<'\n';
        MPI_Barrier(T_plan->comm);
      }
  }

  reshuffle_time+=MPI_Wtime();
  delete [] request;
  delete [] s_request;


  if(VERBOSE>=1){
    PCOUT<<"Shuffle Time= "<<shuffle_time<<std::endl;
    PCOUT<<"Alltoall Time= "<<comm_time<<std::endl;
    PCOUT<<"Reshuffle Time= "<<reshuffle_time<<std::endl;
    PCOUT<<"Total Time= "<<(shuffle_time+comm_time+reshuffle_time)<<std::endl;
  }
  timings[0]+=MPI_Wtime();//timings[0]+=shuffle_time+comm_time+reshuffle_time;
  timings[1]+=shuffle_time;
  timings[2]+=comm_time;
  timings[3]+=reshuffle_time;
  return;

}
void fast_transpose_cuda_v3(T_Plan_gpu* T_plan, double * data, double *timings,int kway, unsigned flags, int howmany, int tag ){
  // This function handles cases where howmany=1 (it is more optimal)


  std::bitset<8> Flags(flags); // 1 Transposed in, 2 Transposed out
  if(Flags[1]==1 && Flags[0]==0 && T_plan->nprocs==1){ // If Flags==Transposed_Out return
    MPI_Barrier(T_plan->comm);
    return;
  }
  if(Flags[0]==1){ // If Flags==Transposed_In This function can not handle it, call other versions
    transpose_cuda_v5(T_plan,(double*)data,timings,flags,howmany,tag);
    MPI_Barrier(T_plan->comm);
    return;
  }
  timings[0]-=MPI_Wtime();
  int nprocs, procid;
  nprocs=T_plan->nprocs;
  procid=T_plan->procid;
  int nprocs_1=T_plan->nprocs_1;
  ptrdiff_t  *N=T_plan->N;

  double * data_cpu=T_plan->buffer;  //pinned
  double * send_recv_cpu = T_plan->buffer_2;//pinned
  double * send_recv_d = T_plan->buffer_d;


  ptrdiff_t local_n0=T_plan->local_n0;
  ptrdiff_t local_n1=T_plan->local_n1;
  ptrdiff_t n_tuples=T_plan->n_tuples;

  int idist=N[1]*local_n0*n_tuples;
  int odist=N[0]*local_n1*n_tuples;

  double comm_time=0*MPI_Wtime(), shuffle_time=0*MPI_Wtime(), reshuffle_time=0*MPI_Wtime(), total_time=0*MPI_Wtime();

  if(VERBOSE>=2) PCOUT<<"INPUT:"<<std::endl;
  if(VERBOSE>=2){
    cudaMemcpy(data_cpu, data, T_plan->alloc_local, cudaMemcpyDeviceToHost);
    for(int h=0;h<howmany;h++)
      for(int id=0;id<nprocs;++id){
        if(procid==id)
          for(int i=0;i<local_n0;i++){
            std::cout<<std::endl;
            for(int j=0;j<N[1];j++){
              std::cout<<'\t'<<data_cpu[h*idist+(i*N[1]+j)*n_tuples];
            }
          }
        std::cout<<'\n';
        MPI_Barrier(T_plan->comm);
      }
  }
  //PCOUT<<" ============================================= "<<std::endl;
  //PCOUT<<" ==============   Local Transpose============= "<<std::endl;
  //PCOUT<<" ============================================= "<<std::endl;
  int ptr=0;
  shuffle_time-=MPI_Wtime();


  if(nprocs==1 && Flags[0]==1 && Flags[1]==1){
#pragma omp parallel for
    for(int h=0;h<howmany;h++)
      local_transpose_cuda(local_n1,N[0],n_tuples,&data[h*idist] );
  }
  if(nprocs==1 && Flags[0]==0 && Flags[1]==0){
#pragma omp parallel for
    for(int h=0;h<howmany;h++)
      local_transpose_cuda(N[0],N[1],n_tuples,&data[h*idist] );
  }
  if(nprocs==1){ // Transpose is done!
    shuffle_time+=MPI_Wtime();
    timings[0]+=MPI_Wtime();
    timings[0]+=shuffle_time;
    timings[1]+=shuffle_time;
    timings[2]+=0;
    timings[3]+=0;
    MPI_Barrier(T_plan->comm);
    return;
  }

  // Flags[1]=1 data_d --Th--> send_recv_d  --memcpy-->  data_cpu  --alltoall--> send_recv_cpu --memcpy--> data_d
  // Flags[1]=0 data_d --Th--> send_recv_d  --memcpy-->  data_cpu  --alltoall--> send_recv_cpu --Th--> data_d
  local_transpose_col_cuda(local_n0,nprocs_1,n_tuples*T_plan->local_n1_proc[0], n_tuples*T_plan->last_local_n1,data,send_recv_d );

  shuffle_time+=MPI_Wtime();
  ptr=0;
  if(VERBOSE>=2) PCOUT<<"Local Transpose:"<<std::endl;
  if(VERBOSE>=2){
    cudaMemcpy(data_cpu,send_recv_d, T_plan->alloc_local, cudaMemcpyDeviceToHost);
    for(int h=0;h<howmany;h++)
      for(int id=0;id<nprocs;++id){
        if(procid==id)
          for(int i=0;i<N[1];i++){
            std::cout<<std::endl;
            for(int j=0;j<local_n0;j++){
              std::cout<<'\t'<<data_cpu[ptr];//data[h*idist+(i*local_n0+j)*n_tuples];
              ptr+=n_tuples;
            }
          }
        std::cout<<'\n';
        MPI_Barrier(T_plan->comm);
      }
  }

  //PCOUT<<" ============================================= "<<std::endl;
  //PCOUT<<" ==============   MPIALLTOALL  =============== "<<std::endl;
  //PCOUT<<" ============================================= "<<std::endl;

  int* scount_proc_f=  T_plan->scount_proc_f;
  int* rcount_proc_f=  T_plan->rcount_proc_f;
  int* soffset_proc_f= T_plan->soffset_proc_f;
  int* roffset_proc_f= T_plan->roffset_proc_f;

  MPI_Barrier(T_plan->comm);

  //PCOUT<<"nprocs_0= "<<nprocs_0<<" nprocs_1= "<<nprocs_1<<std::endl;
  comm_time-=MPI_Wtime();

  MPI_Request * s_request= new MPI_Request[nprocs];
  MPI_Request * request= new MPI_Request[nprocs];
#pragma omp parallel for
  for (int proc=0;proc<nprocs;++proc){
    request[proc]=MPI_REQUEST_NULL;
    s_request[proc]=MPI_REQUEST_NULL;
  }
  double *s_buf, *r_buf;
  s_buf=data_cpu; r_buf=send_recv_cpu;

  // SEND
  // Flags[1]=1 data_d --Th--> send_recv_d  --memcpy-->  data_cpu  --alltoall--> send_recv_cpu --memcpy--> data_d
  // Flags[1]=0 data_d --Th--> send_recv_d  --memcpy-->  data_cpu  --alltoall--> send_recv_cpu --Th--> data_d
  cudaMemcpy(data_cpu, send_recv_d, T_plan->alloc_local, cudaMemcpyDeviceToHost);
  if(T_plan->kway_async)
    par::Mpi_Alltoallv_dense<double,true>(s_buf     , scount_proc_f, soffset_proc_f,
        r_buf, rcount_proc_f, roffset_proc_f, T_plan->comm,kway);
  else
    par::Mpi_Alltoallv_dense<double,false>(s_buf     , scount_proc_f, soffset_proc_f,
        r_buf, rcount_proc_f, roffset_proc_f, T_plan->comm,kway);

  // Flags[1]=1 data_d --Th--> send_recv_d  --memcpy-->  data_cpu  --alltoall--> send_recv_cpu --memcpy--> data_d
  if(Flags[1]==1)
    cudaMemcpy(data, r_buf, T_plan->alloc_local, cudaMemcpyHostToDevice);
  else
    cudaMemcpy(send_recv_d, r_buf, T_plan->alloc_local, cudaMemcpyHostToDevice);
  comm_time+=MPI_Wtime();

  ptr=0;
  if(VERBOSE>=2) PCOUT<<"MPIAlltoAll:"<<std::endl;
  if(VERBOSE>=2){
    cudaMemcpy(data_cpu,send_recv_d, T_plan->alloc_local, cudaMemcpyDeviceToHost);
    for(int h=0;h<howmany;h++)
      for(int id=0;id<nprocs;++id){
        if(procid==id)
          for(int i=0;i<local_n1;i++){
            std::cout<<std::endl;
            for(int j=0;j<N[0];j++){
              std::cout<<'\t'<<data_cpu[ptr];//send_recv[h*odist+(i*N[0]+j)*n_tuples];//<<","<<send_recv[(i*N[0]+j)*n_tuples+1];
              ptr+=n_tuples;
            }
          }
        std::cout<<'\n';
        MPI_Barrier(T_plan->comm);
      }
  }
  //PCOUT<<" ============================================= "<<std::endl;
  //PCOUT<<" ============== 2nd Local Trnaspose ========== "<<std::endl;
  //PCOUT<<" ============================================= "<<std::endl;
  reshuffle_time-=MPI_Wtime();
  ptr=0;
  if(Flags[1]==0)
    local_transpose_cuda(N[0],local_n1,n_tuples,send_recv_d,data );




  if(VERBOSE>=2) PCOUT<<"2nd Transpose"<<std::endl;
  if(VERBOSE>=2){
    cudaMemcpy(data_cpu, data, T_plan->alloc_local, cudaMemcpyDeviceToHost);
    for(int h=0;h<howmany;h++)
      for(int id=0;id<nprocs_1;++id){
        if(procid==id)
          for(int i=0;i<local_n1;i++){
            std::cout<<std::endl;
            for(int j=0;j<N[0];j++){
              std::cout<<'\t'<<data_cpu[h*odist+(i*N[0]+j)*n_tuples];
            }
          }
        std::cout<<'\n';
        MPI_Barrier(T_plan->comm);
      }
  }

  reshuffle_time+=MPI_Wtime();
  delete [] request;
  delete [] s_request;


  if(VERBOSE>=1){
    PCOUT<<"Shuffle Time= "<<shuffle_time<<std::endl;
    PCOUT<<"Alltoall Time= "<<comm_time<<std::endl;
    PCOUT<<"Reshuffle Time= "<<reshuffle_time<<std::endl;
    PCOUT<<"Total Time= "<<(shuffle_time+comm_time+reshuffle_time)<<std::endl;
  }
  timings[0]+=MPI_Wtime();//timings[0]+=shuffle_time+comm_time+reshuffle_time;
  timings[1]+=shuffle_time;
  timings[2]+=comm_time;
  timings[3]+=reshuffle_time;
  return;

}

void fast_transpose_cuda_v3_2(T_Plan_gpu* T_plan, double * data, double *timings,int kway, unsigned flags, int howmany, int tag ){
  // This function handles cases where howmany=1 (it is more optimal)


  std::bitset<8> Flags(flags); // 1 Transposed in, 2 Transposed out
  if(Flags[1]==1 && Flags[0]==0 && T_plan->nprocs==1){ // If Flags==Transposed_Out return
    MPI_Barrier(T_plan->comm);
    return;
  }
  if(Flags[0]==1){ // If Flags==Transposed_In This function can not handle it, call other versions
    transpose_cuda_v7_2(T_plan,(double*)data,timings,kway,flags,howmany, tag);
    MPI_Barrier(T_plan->comm);
    return;
  }
  if(Flags[1]==0){// because in the communicator part it will need another gpu array.
    fast_transpose_cuda_v3(T_plan,(double*)data,timings,kway,flags,howmany, tag);
    MPI_Barrier(T_plan->comm);
    return;
  }
  timings[0]-=MPI_Wtime();
  int nprocs, procid;
  nprocs=T_plan->nprocs;
  procid=T_plan->procid;
  int nprocs_1=T_plan->nprocs_1;
  ptrdiff_t  *N=T_plan->N;

  double * data_cpu=T_plan->buffer;  //pinned
  //double * send_recv_cpu = T_plan->buffer;//pinned
  double * send_recv_d = T_plan->buffer_d;


  ptrdiff_t local_n0=T_plan->local_n0;
  ptrdiff_t local_n1=T_plan->local_n1;
  ptrdiff_t n_tuples=T_plan->n_tuples;

  int idist=N[1]*local_n0*n_tuples;
  int odist=N[0]*local_n1*n_tuples;

  double comm_time=0*MPI_Wtime(), shuffle_time=0*MPI_Wtime(), reshuffle_time=0*MPI_Wtime(), total_time=0*MPI_Wtime();

  if(VERBOSE>=2) PCOUT<<"INPUT:"<<std::endl;
  if(VERBOSE>=2){
    cudaMemcpy(data_cpu, data, T_plan->alloc_local, cudaMemcpyDeviceToHost);
    for(int h=0;h<howmany;h++)
      for(int id=0;id<nprocs;++id){
        if(procid==id)
          for(int i=0;i<local_n0;i++){
            std::cout<<std::endl;
            for(int j=0;j<N[1];j++){
              std::cout<<'\t'<<data_cpu[h*idist+(i*N[1]+j)*n_tuples];
            }
          }
        std::cout<<'\n';
        MPI_Barrier(T_plan->comm);
      }
  }
  //PCOUT<<" ============================================= "<<std::endl;
  //PCOUT<<" ==============   Local Transpose============= "<<std::endl;
  //PCOUT<<" ============================================= "<<std::endl;
  int ptr=0;
  shuffle_time-=MPI_Wtime();


  if(nprocs==1 && Flags[0]==1 && Flags[1]==1){
#pragma omp parallel for
    for(int h=0;h<howmany;h++)
      local_transpose_cuda(local_n1,N[0],n_tuples,&data[h*idist] );
  }
  if(nprocs==1 && Flags[0]==0 && Flags[1]==0){
#pragma omp parallel for
    for(int h=0;h<howmany;h++)
      local_transpose_cuda(N[0],N[1],n_tuples,&data[h*idist] );
  }
  if(nprocs==1){ // Transpose is done!
    shuffle_time+=MPI_Wtime();
    timings[0]+=MPI_Wtime();
    timings[0]+=shuffle_time;
    timings[1]+=shuffle_time;
    timings[2]+=0;
    timings[3]+=0;
    MPI_Barrier(T_plan->comm);
    return;
  }

  // Flags[1]=1 data_d --Th--> send_recv_d  --memcpy-->  data_cpu  --alltoall--> send_recv_cpu --memcpy--> data_d
  // Flags[1]=0 data_d --Th--> send_recv_d  --memcpy-->  data_cpu  --alltoall--> send_recv_cpu --Th--> data_d
  local_transpose_col_cuda(local_n0,nprocs_1,n_tuples*T_plan->local_n1_proc[0], n_tuples*T_plan->last_local_n1,data,send_recv_d );

  shuffle_time+=MPI_Wtime();
  ptr=0;
  if(VERBOSE>=2) PCOUT<<"Local Transpose:"<<std::endl;
  if(VERBOSE>=2){
    cudaMemcpy(data_cpu,send_recv_d, T_plan->alloc_local, cudaMemcpyDeviceToHost);
    for(int h=0;h<howmany;h++)
      for(int id=0;id<nprocs;++id){
        if(procid==id)
          for(int i=0;i<N[1];i++){
            std::cout<<std::endl;
            for(int j=0;j<local_n0;j++){
              std::cout<<'\t'<<data_cpu[ptr];//data[h*idist+(i*local_n0+j)*n_tuples];
              ptr+=n_tuples;
            }
          }
        std::cout<<'\n';
        MPI_Barrier(T_plan->comm);
      }
  }

  //PCOUT<<" ============================================= "<<std::endl;
  //PCOUT<<" ==============   MPIALLTOALL  =============== "<<std::endl;
  //PCOUT<<" ============================================= "<<std::endl;

  int* scount_proc_f=  T_plan->scount_proc_f;
  int* rcount_proc_f=  T_plan->rcount_proc_f;
  int* soffset_proc_f= T_plan->soffset_proc_f;
  int* roffset_proc_f= T_plan->roffset_proc_f;

  MPI_Barrier(T_plan->comm);

  //PCOUT<<"nprocs_0= "<<nprocs_0<<" nprocs_1= "<<nprocs_1<<std::endl;
  comm_time-=MPI_Wtime();

  MPI_Request * s_request= new MPI_Request[nprocs];
  MPI_Request * request= new MPI_Request[nprocs];
#pragma omp parallel for
  for (int proc=0;proc<nprocs;++proc){
    request[proc]=MPI_REQUEST_NULL;
    s_request[proc]=MPI_REQUEST_NULL;
  }
  double * r_buf_d;
  // if Flags[1]==0 the other function should have been called and returned already.
  if(Flags[1]==1)
    r_buf_d=data;
  else
    r_buf_d=send_recv_d;

  // SEND
  // Flags[1]=1 data_d --Th--> send_recv_d  --memcpy-->  data_cpu  --alltoall--> send_recv_cpu --memcpy--> data_d
  // Flags[1]=0 data_d --Th--> send_recv_d  --memcpy-->  data_cpu  --alltoall--> send_recv_cpu --Th--> data_d

  if(T_plan->kway_async)
    par::Mpi_Alltoallv_dense_gpu<double,true>(send_recv_d , scount_proc_f, soffset_proc_f,
        r_buf_d, rcount_proc_f, roffset_proc_f, T_plan->comm,kway);
  else
    par::Mpi_Alltoallv_dense_gpu<double,false>(send_recv_d , scount_proc_f, soffset_proc_f,
        r_buf_d, rcount_proc_f, roffset_proc_f, T_plan->comm,kway);



  comm_time+=MPI_Wtime();

  ptr=0;
  if(VERBOSE>=2) PCOUT<<"MPIAlltoAll:"<<std::endl;
  if(VERBOSE>=2){
    cudaMemcpy(data_cpu,send_recv_d, T_plan->alloc_local, cudaMemcpyDeviceToHost);
    for(int h=0;h<howmany;h++)
      for(int id=0;id<nprocs;++id){
        if(procid==id)
          for(int i=0;i<local_n1;i++){
            std::cout<<std::endl;
            for(int j=0;j<N[0];j++){
              std::cout<<'\t'<<data_cpu[ptr];//send_recv[h*odist+(i*N[0]+j)*n_tuples];//<<","<<send_recv[(i*N[0]+j)*n_tuples+1];
              ptr+=n_tuples;
            }
          }
        std::cout<<'\n';
        MPI_Barrier(T_plan->comm);
      }
  }
  //PCOUT<<" ============================================= "<<std::endl;
  //PCOUT<<" ============== 2nd Local Trnaspose ========== "<<std::endl;
  //PCOUT<<" ============================================= "<<std::endl;
  reshuffle_time-=MPI_Wtime();
  ptr=0;
  if(Flags[1]==0)
    local_transpose_cuda(N[0],local_n1,n_tuples,send_recv_d,data );




  if(VERBOSE>=2) PCOUT<<"2nd Transpose"<<std::endl;
  if(VERBOSE>=2){
    cudaMemcpy(data_cpu, data, T_plan->alloc_local, cudaMemcpyDeviceToHost);
    for(int h=0;h<howmany;h++)
      for(int id=0;id<nprocs_1;++id){
        if(procid==id)
          for(int i=0;i<local_n1;i++){
            std::cout<<std::endl;
            for(int j=0;j<N[0];j++){
              std::cout<<'\t'<<data_cpu[h*odist+(i*N[0]+j)*n_tuples];
            }
          }
        std::cout<<'\n';
        MPI_Barrier(T_plan->comm);
      }
  }

  reshuffle_time+=MPI_Wtime();
  delete [] request;
  delete [] s_request;


  if(VERBOSE>=1){
    PCOUT<<"Shuffle Time= "<<shuffle_time<<std::endl;
    PCOUT<<"Alltoall Time= "<<comm_time<<std::endl;
    PCOUT<<"Reshuffle Time= "<<reshuffle_time<<std::endl;
    PCOUT<<"Total Time= "<<(shuffle_time+comm_time+reshuffle_time)<<std::endl;
  }
  timings[0]+=MPI_Wtime();//timings[0]+=shuffle_time+comm_time+reshuffle_time;
  timings[1]+=shuffle_time;
  timings[2]+=comm_time;
  timings[3]+=reshuffle_time;
  return;

}




void fast_transpose_cuda_v2_h(T_Plan_gpu* T_plan, double * data, double *timings, unsigned flags, int howmany, int tag ){

  std::bitset<8> Flags(flags); // 1 Transposed in, 2 Transposed out
  if(Flags[1]==1 && Flags[0]==0 && T_plan->nprocs==1){ // If nprocs==1 and Flags==Transposed_Out return
    MPI_Barrier(T_plan->comm);
    return;
  }
  if(Flags[0]==1){ // If Flags==Transposed_In This function can not handle it, call other versions
    transpose_cuda_v6(T_plan,(double*)data,timings,flags,howmany,tag);
    MPI_Barrier(T_plan->comm);
    return;
  }
  timings[0]-=MPI_Wtime();
  int nprocs, procid;
  int nprocs_0, nprocs_1;
  nprocs=T_plan->nprocs;
  procid=T_plan->procid;
  nprocs_0=T_plan->nprocs_0;
  nprocs_1=T_plan->nprocs_1;
  ptrdiff_t  *N=T_plan->N;
  ptrdiff_t local_n0=T_plan->local_n0;
  ptrdiff_t local_n1=T_plan->local_n1;
  ptrdiff_t n_tuples=T_plan->n_tuples;

  double * data_cpu=T_plan->buffer;  //pinned
  double * send_recv_cpu = T_plan->buffer_2;//pinned
  double * send_recv_d = T_plan->buffer_d;

  int idist=N[1]*local_n0*n_tuples;
  int odist=N[0]*local_n1*n_tuples;

  double comm_time=0*MPI_Wtime(), shuffle_time=0*MPI_Wtime(), reshuffle_time=0*MPI_Wtime(), total_time=0*MPI_Wtime();

  int ptr=0;
  if(VERBOSE>=2) PCOUT<<"INPUT:"<<std::endl;
  if(VERBOSE>=2){
    cudaMemcpy(data_cpu, data, T_plan->alloc_local, cudaMemcpyDeviceToHost);
    for(int h=0;h<howmany;h++)
      for(int id=0;id<nprocs;++id){
        if(procid==id)
          for(int i=0;i<local_n0;i++){
            std::cout<<std::endl;
            for(int j=0;j<N[1];j++){
              ptr=h*idist+(i*N[1]+j)*n_tuples;
              std::cout<<'\t'<<data_cpu[ptr]<<","<<data_cpu[ptr+1];
            }
          }
        std::cout<<'\n';
        MPI_Barrier(T_plan->comm);
      }
  }
  //PCOUT<<" ============================================= "<<std::endl;
  //PCOUT<<" ==============   Local Transpose============= "<<std::endl;
  //PCOUT<<" ============================================= "<<std::endl;
  ptr=0;
  ptrdiff_t *local_n1_proc=&T_plan->local_n1_proc[0];
  ptrdiff_t *local_n0_proc=&T_plan->local_n0_proc[0];
  ptrdiff_t *local_0_start_proc=T_plan->local_0_start_proc;
  ptrdiff_t *local_1_start_proc=T_plan->local_1_start_proc;
  shuffle_time-=MPI_Wtime();
  if(nprocs==1 && Flags[0]==1 && Flags[1]==1){
#pragma omp parallel for
    for(int h=0;h<howmany;h++)
      local_transpose_cuda(local_n1,N[0],n_tuples,&data[h*idist] );
  }
  if(nprocs==1 && Flags[0]==0 && Flags[1]==0){
#pragma omp parallel for
    for(int h=0;h<howmany;h++)
      local_transpose_cuda(N[0],N[1],n_tuples,&data[h*idist] );
  }
  if(nprocs==1){ // Transpose is done!
    shuffle_time+=MPI_Wtime();
    timings[0]+=MPI_Wtime();
    timings[0]+=shuffle_time;
    timings[1]+=shuffle_time;
    timings[2]+=0;
    timings[3]+=0;
    MPI_Barrier(T_plan->comm);
    return;
  }

  // data_d --Th--> send_recv_d  --memcpy-->  data_cpu  --alltoall--> send_recv_cpu --Th--> data_d
  ptr=0;

  if(0)
    for (int proc=0;proc<nprocs_1;++proc)
      for(int h=0;h<howmany;++h){
        for(int i=0;i<local_n0;++i){
          //for(int j=local_1_start_proc[proc];j<local_1_start_proc[proc]+local_n1_proc[proc];++j){
          //  memcpy(&buffer_2[ptr],&data[h*idist+(i*N[1]+j)*n_tuples],sizeof(double)*n_tuples);
          //  //std::cout<<"proc= "<<proc<<" h= "<<h<<" (i,j)=("<<i<<","<<j<<")  data_ptr= "<<h*idist+(i*local_n1+j)*n_tuples<< " ptr= "<<ptr <<" cpy= "<<n_tuples*T_plan->local_n1_proc[0] <<std::endl;
          //  ptr+=n_tuples;
          //}
          //memcpy(&buffer_2[ptr],&data[h*idist+(i*N[1]+local_1_start_proc[proc])*n_tuples],sizeof(double)*n_tuples*local_n1_proc[proc]);
          cudaMemcpy(&send_recv_d[ptr],&data[h*idist+(i*N[1]+local_1_start_proc[proc])*n_tuples] , sizeof(double)*n_tuples*local_n1_proc[proc] , cudaMemcpyDeviceToDevice);
          ptr+=n_tuples*local_n1_proc[proc];
        }
      }
  memcpy_v1_h1(nprocs_1,howmany,local_n0,n_tuples,local_n1_proc,send_recv_d,data,idist,N[1],local_1_start_proc);
  cudaDeviceSynchronize();

  shuffle_time+=MPI_Wtime();
  ptr=0;
  if(VERBOSE>=2) PCOUT<<"Local Transpose:"<<std::endl;
  if(VERBOSE>=2){
    cudaMemcpy(data_cpu, send_recv_d, T_plan->alloc_local, cudaMemcpyDeviceToHost);
    for(int id=0;id<nprocs;++id){
      for(int h=0;h<howmany;h++)
        if(procid==id)
          for(int i=0;i<N[1];i++){
            std::cout<<std::endl;
            for(int j=0;j<local_n0;j++){
              std::cout<<'\t'<<data_cpu[ptr]<<","<<data_cpu[ptr+1];
              ptr+=n_tuples;
            }
          }
      std::cout<<'\n';
      MPI_Barrier(T_plan->comm);
    }
  }


  //PCOUT<<" ============================================= "<<std::endl;
  //PCOUT<<" ==============   MPIALLTOALL  =============== "<<std::endl;
  //PCOUT<<" ============================================= "<<std::endl;

  int* scount_proc_f=  T_plan->scount_proc_f;
  int* rcount_proc_f=  T_plan->rcount_proc_f;
  int* soffset_proc_f= T_plan->soffset_proc_f;
  int* roffset_proc_f= T_plan->roffset_proc_f;

  MPI_Barrier(T_plan->comm);

  //PCOUT<<"nprocs_0= "<<nprocs_0<<" nprocs_1= "<<nprocs_1<<std::endl;
  comm_time-=MPI_Wtime();


  double *s_buf, *r_buf;
  s_buf=data_cpu; r_buf=send_recv_cpu;
  // data_d --Th--> send_recv_d  --memcpy-->  data_cpu  --alltoall--> send_recv_cpu --Th--> data_d

  // SEND
  cudaMemcpy(data_cpu, send_recv_d, T_plan->alloc_local, cudaMemcpyDeviceToHost);
  if(T_plan->is_evenly_distributed==0)
    MPI_Alltoallv(s_buf,scount_proc_f,
        soffset_proc_f, MPI_DOUBLE,r_buf,
        rcount_proc_f,roffset_proc_f, MPI_DOUBLE,
        T_plan->comm);
  else
    MPI_Alltoall(s_buf, scount_proc_f[0], MPI_DOUBLE,
        r_buf, rcount_proc_f[0], MPI_DOUBLE,
        T_plan->comm);


  // data_d --Th--> send_recv_d  --memcpy-->  data_cpu  --alltoall--> send_recv_cpu --Th--> data_d
  //cudaMemcpy(send_recv_d, send_recv_cpu, T_plan->alloc_local, cudaMemcpyHostToDevice);
  comm_time+=MPI_Wtime();


  ptr=0;
  if(VERBOSE>=2) PCOUT<<"MPIAlltoAll:"<<std::endl;
  if(VERBOSE>=2){
    cudaMemcpy(data_cpu, send_recv_d, T_plan->alloc_local, cudaMemcpyDeviceToHost);
    for(int id=0;id<nprocs;++id){
      if(procid==id)
        for(int h=0;h<howmany;h++)
          for(int i=0;i<local_n1;i++){
            std::cout<<std::endl;
            for(int j=0;j<N[0];j++){
              std::cout<<'\t'<<data_cpu[ptr]<<","<<data_cpu[ptr+1];
              ptr+=n_tuples;
            }
          }
      std::cout<<'\n';
      MPI_Barrier(T_plan->comm);
    }
  }
  //PCOUT<<" ============================================= "<<std::endl;
  //PCOUT<<" ============== 2nd Local Trnaspose ========== "<<std::endl;
  //PCOUT<<" ============================================= "<<std::endl;
  // data_d --Th--> send_recv_d  --memcpy-->  data_cpu  --alltoall--> send_recv_cpu --Th--> data_d
  reshuffle_time-=MPI_Wtime();
  ptr=0;
  if(0)
    for (int proc=0;proc<nprocs_0;++proc)
      for(int h=0;h<howmany;++h){
        for(int i=local_0_start_proc[proc];i<local_0_start_proc[proc]+local_n0_proc[proc];++i){
          //memcpy(&data_cpu[h*odist+(i*local_n1)*n_tuples],&send_recv_cpu[ptr],local_n1*sizeof(double)*n_tuples);
          cudaMemcpy(    &data[h*odist+(i*local_n1)*n_tuples],&send_recv_cpu[ptr],local_n1*sizeof(double)*n_tuples,cudaMemcpyHostToDevice);
          //std::cout<<"proc= "<<proc<<" h= "<<h<<" i=("<<i<<")  data_ptr= "<<h*odist+(i*local_n1)*n_tuples<< " ptr= "<<ptr <<" cpy= "<<n_tuples*local_n1_proc[proc] <<std::endl;
          ptr+=n_tuples*local_n1;
          //for(int j=0*local_1_start_proc[proc];j<0*local_1_start_proc[proc]+local_n1;++j){
          //  memcpy(&data[h*odist+(i*local_n1+j)*n_tuples],&send_recv[ptr],sizeof(double)*n_tuples);
          //std::cout<<"proc= "<<proc<<" h= "<<h<<" (i,j)=("<<i<<","<<j<<")  data_ptr= "<<h*idist+(i*local_n1+j)*n_tuples<< " ptr= "<<ptr <<" cpy= "<<n_tuples*T_plan->local_n1_proc[0] <<std::endl;
          //  ptr+=n_tuples;
          //}
        }
      }
  memcpy_v1_h2(nprocs_0,howmany,local_0_start_proc,local_n0_proc,data,odist,local_n1,n_tuples,send_recv_cpu);
  cudaDeviceSynchronize();

  if(VERBOSE>=2) PCOUT<<"2nd Transpose"<<std::endl;
  if(VERBOSE>=2){
    cudaMemcpy(data_cpu, data, T_plan->alloc_local, cudaMemcpyDeviceToHost);
    for(int id=0;id<nprocs_1;++id){
      if(procid==id)
        for(int h=0;h<howmany;h++)
          for(int i=0;i<N[0];i++){
            std::cout<<std::endl;
            for(int j=0;j<local_n1;j++){
              ptr=h*odist+(i*local_n1+j)*n_tuples;
              std::cout<<'\t'<<data_cpu[ptr]<<","<<data_cpu[ptr+1];
            }
          }
      std::cout<<'\n';
      MPI_Barrier(T_plan->comm);
    }
  }
  // Right now the data is in transposed out format.
  // If the user did not want this layout, transpose again.
  if(Flags[1]==0){
#pragma omp parallel for
    for(int h=0;h<howmany;h++)
      local_transpose_cuda(N[0],local_n1,n_tuples,&data[h*odist] );

    if(VERBOSE>=2) PCOUT<<"2nd Transpose"<<std::endl;
    if(VERBOSE>=2){
      cudaMemcpy(data_cpu, data, T_plan->alloc_local, cudaMemcpyDeviceToHost);
      for(int id=0;id<nprocs_1;++id){
        if(procid==id)
          for(int h=0;h<howmany;h++)
            for(int i=0;i<N[0];i++){
              std::cout<<std::endl;
              for(int j=0;j<local_n1;j++){
                ptr=h*odist+(i*local_n1+j)*n_tuples;
                std::cout<<'\t'<<data_cpu[ptr]<<","<<data_cpu[ptr+1];
              }
            }
        std::cout<<'\n';
        MPI_Barrier(T_plan->comm);
      }
    }
  }

  reshuffle_time+=MPI_Wtime();
  MPI_Barrier(T_plan->comm);


  if(VERBOSE>=1){
    PCOUT<<"Shuffle Time= "<<shuffle_time<<std::endl;
    PCOUT<<"Alltoall Time= "<<comm_time<<std::endl;
    PCOUT<<"Reshuffle Time= "<<reshuffle_time<<std::endl;
    PCOUT<<"Total Time= "<<(shuffle_time+comm_time+reshuffle_time)<<std::endl;
  }
  timings[0]+=MPI_Wtime();//timings[0]+=shuffle_time+comm_time+reshuffle_time;
  timings[1]+=shuffle_time;
  timings[2]+=comm_time;
  timings[3]+=reshuffle_time;
  return;
}

void fast_transpose_cuda_v3_h(T_Plan_gpu* T_plan, double * data, double *timings,int kway, unsigned flags, int howmany , int tag){

  std::bitset<8> Flags(flags); // 1 Transposed in, 2 Transposed out
  if(Flags[1]==1 && Flags[0]==0 && T_plan->nprocs==1){ // If nprocs==1 and Flags==Transposed_Out return
    MPI_Barrier(T_plan->comm);
    return;
  }
  if(Flags[0]==1){ // If Flags==Transposed_In This function can not handle it, call other versions
    transpose_cuda_v6(T_plan,(double*)data,timings,flags,howmany,tag);
    MPI_Barrier(T_plan->comm);
    return;
  }
  timings[0]-=MPI_Wtime();
  int nprocs, procid;
  int nprocs_0, nprocs_1;
  nprocs=T_plan->nprocs;
  procid=T_plan->procid;
  nprocs_0=T_plan->nprocs_0;
  nprocs_1=T_plan->nprocs_1;
  ptrdiff_t  *N=T_plan->N;
  ptrdiff_t local_n0=T_plan->local_n0;
  ptrdiff_t local_n1=T_plan->local_n1;
  ptrdiff_t n_tuples=T_plan->n_tuples;

  double * data_cpu=T_plan->buffer;  //pinned
  double * send_recv_cpu = T_plan->buffer_2;//pinned
  double * send_recv_d = T_plan->buffer_d;

  int idist=N[1]*local_n0*n_tuples;
  int odist=N[0]*local_n1*n_tuples;

  double comm_time=0*MPI_Wtime(), shuffle_time=0*MPI_Wtime(), reshuffle_time=0*MPI_Wtime(), total_time=0*MPI_Wtime();

  int ptr=0;
  if(VERBOSE>=2) PCOUT<<"INPUT:"<<std::endl;
  if(VERBOSE>=2){
    cudaMemcpy(data_cpu, data, T_plan->alloc_local, cudaMemcpyDeviceToHost);
    for(int h=0;h<howmany;h++)
      for(int id=0;id<nprocs;++id){
        if(procid==id)
          for(int i=0;i<local_n0;i++){
            std::cout<<std::endl;
            for(int j=0;j<N[1];j++){
              ptr=h*idist+(i*N[1]+j)*n_tuples;
              std::cout<<'\t'<<data_cpu[ptr]<<","<<data_cpu[ptr+1];
            }
          }
        std::cout<<'\n';
        MPI_Barrier(T_plan->comm);
      }
  }
  //PCOUT<<" ============================================= "<<std::endl;
  //PCOUT<<" ==============   Local Transpose============= "<<std::endl;
  //PCOUT<<" ============================================= "<<std::endl;
  ptr=0;
  ptrdiff_t *local_n1_proc=&T_plan->local_n1_proc[0];
  ptrdiff_t *local_n0_proc=&T_plan->local_n0_proc[0];
  ptrdiff_t *local_0_start_proc=T_plan->local_0_start_proc;
  ptrdiff_t *local_1_start_proc=T_plan->local_1_start_proc;
  shuffle_time-=MPI_Wtime();
  if(nprocs==1 && Flags[0]==1 && Flags[1]==1){
#pragma omp parallel for
    for(int h=0;h<howmany;h++)
      local_transpose_cuda(local_n1,N[0],n_tuples,&data[h*idist] );
  }
  if(nprocs==1 && Flags[0]==0 && Flags[1]==0){
#pragma omp parallel for
    for(int h=0;h<howmany;h++)
      local_transpose_cuda(N[0],N[1],n_tuples,&data[h*idist] );
  }
  if(nprocs==1){ // Transpose is done!
    shuffle_time+=MPI_Wtime();
    timings[0]+=MPI_Wtime();
    timings[0]+=shuffle_time;
    timings[1]+=shuffle_time;
    timings[2]+=0;
    timings[3]+=0;
    MPI_Barrier(T_plan->comm);
    return;
  }

  // data_d --Th--> send_recv_d  --memcpy-->  data_cpu  --alltoall--> send_recv_cpu --Th--> data_d
  ptr=0;

  if(0)
    for (int proc=0;proc<nprocs_1;++proc)
      for(int h=0;h<howmany;++h){
        for(int i=0;i<local_n0;++i){
          //for(int j=local_1_start_proc[proc];j<local_1_start_proc[proc]+local_n1_proc[proc];++j){
          //  memcpy(&buffer_2[ptr],&data[h*idist+(i*N[1]+j)*n_tuples],sizeof(double)*n_tuples);
          //  //std::cout<<"proc= "<<proc<<" h= "<<h<<" (i,j)=("<<i<<","<<j<<")  data_ptr= "<<h*idist+(i*local_n1+j)*n_tuples<< " ptr= "<<ptr <<" cpy= "<<n_tuples*T_plan->local_n1_proc[0] <<std::endl;
          //  ptr+=n_tuples;
          //}
          //memcpy(&buffer_2[ptr],&data[h*idist+(i*N[1]+local_1_start_proc[proc])*n_tuples],sizeof(double)*n_tuples*local_n1_proc[proc]);
          cudaMemcpy(&send_recv_d[ptr],&data[h*idist+(i*N[1]+local_1_start_proc[proc])*n_tuples] , sizeof(double)*n_tuples*local_n1_proc[proc] , cudaMemcpyDeviceToDevice);
          ptr+=n_tuples*local_n1_proc[proc];
        }
      }
  memcpy_v1_h1(nprocs_1,howmany,local_n0,n_tuples,local_n1_proc,send_recv_d,data,idist,N[1],local_1_start_proc);
  cudaDeviceSynchronize();

  shuffle_time+=MPI_Wtime();
  ptr=0;
  if(VERBOSE>=2) PCOUT<<"Local Transpose:"<<std::endl;
  if(VERBOSE>=2){
    cudaMemcpy(data_cpu, send_recv_d, T_plan->alloc_local, cudaMemcpyDeviceToHost);
    for(int id=0;id<nprocs;++id){
      for(int h=0;h<howmany;h++)
        if(procid==id)
          for(int i=0;i<N[1];i++){
            std::cout<<std::endl;
            for(int j=0;j<local_n0;j++){
              std::cout<<'\t'<<data_cpu[ptr]<<","<<data_cpu[ptr+1];
              ptr+=n_tuples;
            }
          }
      std::cout<<'\n';
      MPI_Barrier(T_plan->comm);
    }
  }


  //PCOUT<<" ============================================= "<<std::endl;
  //PCOUT<<" ==============   MPIALLTOALL  =============== "<<std::endl;
  //PCOUT<<" ============================================= "<<std::endl;

  int* scount_proc_f=  T_plan->scount_proc_f;
  int* rcount_proc_f=  T_plan->rcount_proc_f;
  int* soffset_proc_f= T_plan->soffset_proc_f;
  int* roffset_proc_f= T_plan->roffset_proc_f;

  MPI_Barrier(T_plan->comm);

  //PCOUT<<"nprocs_0= "<<nprocs_0<<" nprocs_1= "<<nprocs_1<<std::endl;
  comm_time-=MPI_Wtime();

  MPI_Request * s_request= new MPI_Request[nprocs];
  MPI_Request * request= new MPI_Request[nprocs];
#pragma omp parallel for
  for (int proc=0;proc<nprocs;++proc){
    request[proc]=MPI_REQUEST_NULL;
    s_request[proc]=MPI_REQUEST_NULL;
  }

  double *s_buf, *r_buf;
  s_buf=data_cpu; r_buf=send_recv_cpu;
  // data_d --Th--> send_recv_d  --memcpy-->  data_cpu  --alltoall--> send_recv_cpu --Th--> data_d

  // SEND
  cudaMemcpy(data_cpu, send_recv_d, T_plan->alloc_local, cudaMemcpyDeviceToHost);
  if(T_plan->kway_async)
    par::Mpi_Alltoallv_dense<double,true>(s_buf     , scount_proc_f, soffset_proc_f,
        r_buf, rcount_proc_f, roffset_proc_f, T_plan->comm,kway);
  else
    par::Mpi_Alltoallv_dense<double,false>(s_buf     , scount_proc_f, soffset_proc_f,
        r_buf, rcount_proc_f, roffset_proc_f, T_plan->comm,kway);

  // data_d --Th--> send_recv_d  --memcpy-->  data_cpu  --alltoall--> send_recv_cpu --Th--> data_d
  //cudaMemcpy(send_recv_d, send_recv_cpu, T_plan->alloc_local, cudaMemcpyHostToDevice);
  comm_time+=MPI_Wtime();


  ptr=0;
  if(VERBOSE>=2) PCOUT<<"MPIAlltoAll:"<<std::endl;
  if(VERBOSE>=2){
    cudaMemcpy(data_cpu, send_recv_d, T_plan->alloc_local, cudaMemcpyDeviceToHost);
    for(int id=0;id<nprocs;++id){
      if(procid==id)
        for(int h=0;h<howmany;h++)
          for(int i=0;i<local_n1;i++){
            std::cout<<std::endl;
            for(int j=0;j<N[0];j++){
              std::cout<<'\t'<<data_cpu[ptr]<<","<<data_cpu[ptr+1];
              ptr+=n_tuples;
            }
          }
      std::cout<<'\n';
      MPI_Barrier(T_plan->comm);
    }
  }
  //PCOUT<<" ============================================= "<<std::endl;
  //PCOUT<<" ============== 2nd Local Trnaspose ========== "<<std::endl;
  //PCOUT<<" ============================================= "<<std::endl;
  // data_d --Th--> send_recv_d  --memcpy-->  data_cpu  --alltoall--> send_recv_cpu --Th--> data_d
  reshuffle_time-=MPI_Wtime();
  ptr=0;
  if(0)
    for (int proc=0;proc<nprocs_0;++proc)
      for(int h=0;h<howmany;++h){
        for(int i=local_0_start_proc[proc];i<local_0_start_proc[proc]+local_n0_proc[proc];++i){
          //memcpy(&data_cpu[h*odist+(i*local_n1)*n_tuples],&send_recv_cpu[ptr],local_n1*sizeof(double)*n_tuples);
          cudaMemcpy(    &data[h*odist+(i*local_n1)*n_tuples],&send_recv_cpu[ptr],local_n1*sizeof(double)*n_tuples,cudaMemcpyHostToDevice);
          //std::cout<<"proc= "<<proc<<" h= "<<h<<" i=("<<i<<")  data_ptr= "<<h*odist+(i*local_n1)*n_tuples<< " ptr= "<<ptr <<" cpy= "<<n_tuples*local_n1_proc[proc] <<std::endl;
          ptr+=n_tuples*local_n1;
          //for(int j=0*local_1_start_proc[proc];j<0*local_1_start_proc[proc]+local_n1;++j){
          //  memcpy(&data[h*odist+(i*local_n1+j)*n_tuples],&send_recv[ptr],sizeof(double)*n_tuples);
          //std::cout<<"proc= "<<proc<<" h= "<<h<<" (i,j)=("<<i<<","<<j<<")  data_ptr= "<<h*idist+(i*local_n1+j)*n_tuples<< " ptr= "<<ptr <<" cpy= "<<n_tuples*T_plan->local_n1_proc[0] <<std::endl;
          //  ptr+=n_tuples;
          //}
        }
      }
  memcpy_v1_h2(nprocs_0,howmany,local_0_start_proc,local_n0_proc,data,odist,local_n1,n_tuples,send_recv_cpu);
  cudaDeviceSynchronize();

  if(VERBOSE>=2) PCOUT<<"2nd Transpose"<<std::endl;
  if(VERBOSE>=2){
    cudaMemcpy(data_cpu, data, T_plan->alloc_local, cudaMemcpyDeviceToHost);
    for(int id=0;id<nprocs_1;++id){
      if(procid==id)
        for(int h=0;h<howmany;h++)
          for(int i=0;i<N[0];i++){
            std::cout<<std::endl;
            for(int j=0;j<local_n1;j++){
              ptr=h*odist+(i*local_n1+j)*n_tuples;
              std::cout<<'\t'<<data_cpu[ptr]<<","<<data_cpu[ptr+1];
            }
          }
      std::cout<<'\n';
      MPI_Barrier(T_plan->comm);
    }
  }
  // Right now the data is in transposed out format.
  // If the user did not want this layout, transpose again.
  if(Flags[1]==0){
#pragma omp parallel for
    for(int h=0;h<howmany;h++)
      local_transpose_cuda(N[0],local_n1,n_tuples,&data[h*odist] );

    if(VERBOSE>=2) PCOUT<<"2nd Transpose"<<std::endl;
    if(VERBOSE>=2){
      cudaMemcpy(data_cpu, data, T_plan->alloc_local, cudaMemcpyDeviceToHost);
      for(int id=0;id<nprocs_1;++id){
        if(procid==id)
          for(int h=0;h<howmany;h++)
            for(int i=0;i<N[0];i++){
              std::cout<<std::endl;
              for(int j=0;j<local_n1;j++){
                ptr=h*odist+(i*local_n1+j)*n_tuples;
                std::cout<<'\t'<<data_cpu[ptr]<<","<<data_cpu[ptr+1];
              }
            }
        std::cout<<'\n';
        MPI_Barrier(T_plan->comm);
      }
    }
  }

  reshuffle_time+=MPI_Wtime();
  MPI_Barrier(T_plan->comm);
  delete [] request;
  delete [] s_request;


  if(VERBOSE>=1){
    PCOUT<<"Shuffle Time= "<<shuffle_time<<std::endl;
    PCOUT<<"Alltoall Time= "<<comm_time<<std::endl;
    PCOUT<<"Reshuffle Time= "<<reshuffle_time<<std::endl;
    PCOUT<<"Total Time= "<<(shuffle_time+comm_time+reshuffle_time)<<std::endl;
  }
  timings[0]+=MPI_Wtime();//timings[0]+=shuffle_time+comm_time+reshuffle_time;
  timings[1]+=shuffle_time;
  timings[2]+=comm_time;
  timings[3]+=reshuffle_time;
  return;
}

void transpose_cuda_v5(T_Plan_gpu* T_plan, double * data, double *timings, unsigned flags, int howmany, int tag ){

  std::bitset<8> Flags(flags); // 1 Transposed in, 2 Transposed out
  if(Flags[1]==1 && Flags[0]==0 && T_plan->nprocs==1){ // If Flags==Transposed_Out return
    MPI_Barrier(T_plan->comm);
    return;
  }
  timings[0]-=MPI_Wtime();
  int nprocs, procid;
  int nprocs_0, nprocs_1;
  nprocs=T_plan->nprocs;
  procid=T_plan->procid;
  nprocs_0=T_plan->nprocs_0;
  nprocs_1=T_plan->nprocs_1;
  ptrdiff_t  *N=T_plan->N;
  double* data_cpu=T_plan->buffer;
  double * send_recv_cpu = T_plan->buffer_2;
  double * send_recv = T_plan->buffer_d;
  ptrdiff_t local_n0=T_plan->local_n0;
  ptrdiff_t local_n1=T_plan->local_n1;
  ptrdiff_t n_tuples=T_plan->n_tuples;
  int idist=N[1]*local_n0*n_tuples;
  int odist=N[0]*local_n1*n_tuples;

  double comm_time=0*MPI_Wtime(), shuffle_time=0*MPI_Wtime(), reshuffle_time=0*MPI_Wtime(), total_time=0*MPI_Wtime();

  if(VERBOSE>=2) PCOUT<<"INPUT:"<<std::endl;
  if(VERBOSE>=2){
    cudaMemcpy(data_cpu, data, T_plan->alloc_local, cudaMemcpyDeviceToHost);
    for(int h=0;h<howmany;h++)
      for(int id=0;id<nprocs;++id){
        if(procid==id)
          for(int i=0;i<local_n0;i++){
            std::cout<<std::endl;
            for(int j=0;j<N[1];j++){
              std::cout<<'\t'<<data_cpu[h*idist+(i*N[1]+j)*n_tuples];
            }
          }
        std::cout<<'\n';
        MPI_Barrier(T_plan->comm);
      }
  }
  //PCOUT<<" ============================================= "<<std::endl;
  //PCOUT<<" ==============   Local Transpose============= "<<std::endl;
  //PCOUT<<" ============================================= "<<std::endl;
  shuffle_time-=MPI_Wtime();

  if(Flags[0]==0){
    for(int h=0;h<howmany;h++)
      local_transpose_cuda(local_n0,N[1],n_tuples,&data[h*idist] );
  }
  cudaDeviceSynchronize();

  shuffle_time+=MPI_Wtime();
  if(VERBOSE>=2) PCOUT<<"Local Transpose:"<<std::endl;
  if(VERBOSE>=2){
    cudaMemcpy(data_cpu, data, T_plan->alloc_local, cudaMemcpyDeviceToHost);
    for(int h=0;h<howmany;h++)
      for(int id=0;id<nprocs;++id){
        if(procid==id)
          for(int i=0;i<N[1];i++){
            std::cout<<std::endl;
            for(int j=0;j<local_n0;j++){
              std::cout<<'\t'<<data_cpu[h*idist+(i*local_n0+j)*n_tuples];
            }
          }
        std::cout<<'\n';
        MPI_Barrier(T_plan->comm);
      }
  }

  if(nprocs==1 && Flags[0]==1 && Flags[1]==1){
    for(int h=0;h<howmany;h++)
      local_transpose_cuda(local_n1,N[0],n_tuples,&data[h*idist] );
  }
  if(nprocs==1){ // Transpose is done!
    timings[0]+=MPI_Wtime();
    timings[1]+=shuffle_time;
    timings[2]+=0;
    timings[3]+=0;
    return;
  }



  //PCOUT<<" ============================================= "<<std::endl;
  //PCOUT<<" ==============   MPIALLTOALL  =============== "<<std::endl;
  //PCOUT<<" ============================================= "<<std::endl;
  int* scount_proc=  T_plan->scount_proc;
  //int* rcount_proc=  T_plan->rcount_proc;
  int* soffset_proc= T_plan->soffset_proc;
  int* roffset_proc= T_plan->roffset_proc;

  comm_time-=MPI_Wtime();

  int soffset=0,roffset=0;
  MPI_Status ierr;
  MPI_Request request[nprocs], s_request[nprocs];
#pragma omp parallel for
  for (int proc=0;proc<nprocs;++proc){
    request[proc]=MPI_REQUEST_NULL;
    s_request[proc]=MPI_REQUEST_NULL;
  }

  MPI_Datatype *stype=T_plan->stype;
  MPI_Datatype *rtype=T_plan->rtype;
  MPI_Barrier(T_plan->comm);

  // Post Receives
  for (int proc=0;proc<nprocs;++proc){
    if(proc!=procid){
      roffset=roffset_proc[proc];
      MPI_Irecv(&send_recv_cpu[roffset],1, rtype[proc], proc,
          tag, T_plan->comm, &request[proc]);
    }
  }
  // SEND
  cudaMemcpy(data_cpu, data, T_plan->alloc_local, cudaMemcpyDeviceToHost);
  for (int proc=0;proc<nprocs;++proc){
    if(proc!=procid){
      soffset=soffset_proc[proc];
      MPI_Isend(&data_cpu[soffset],1, stype[proc],proc, tag,
          T_plan->comm, &s_request[proc]);
    }
  }

  soffset=soffset_proc[procid];//aoffset_proc[proc];SNAFU //proc*count_proc[proc];
  roffset=roffset_proc[procid];
  for(int h=0;h<howmany;h++)
    memcpy(&send_recv_cpu[h*odist+roffset],&data_cpu[h*idist+soffset],sizeof(double)*scount_proc[procid]);
  for (int proc=0;proc<nprocs;++proc){
    MPI_Wait(&request[proc], &ierr);
    MPI_Wait(&s_request[proc], &ierr);
  }

  cudaMemcpy(send_recv, send_recv_cpu, T_plan->alloc_local, cudaMemcpyHostToDevice);
  cudaDeviceSynchronize();
  comm_time+=MPI_Wtime();

  if(VERBOSE>=2) PCOUT<<"MPIAlltoAll:"<<std::endl;
  if(VERBOSE>=2){
    for(int h=0;h<howmany;h++)
      for(int id=0;id<nprocs;++id){
        if(procid==id)
          for(int i=0;i<local_n1;i++){
            std::cout<<std::endl;
            for(int j=0;j<N[0];j++){
              std::cout<<'\t'<<send_recv_cpu[h*odist+(i*N[0]+j)*n_tuples];//<<","<<send_recv[(i*N[0]+j)*n_tuples+1];
            }
          }
        std::cout<<'\n';
        MPI_Barrier(T_plan->comm);
      }
  }



  //PCOUT<<" ============================================= "<<std::endl;
  //PCOUT<<" ============== 2nd Local Trnaspose ========== "<<std::endl;
  //PCOUT<<" ============================================= "<<std::endl;

  reshuffle_time-=MPI_Wtime();

  // int first_local_n0, last_local_n0;
  // first_local_n0=local_n0_proc[0]; last_local_n0=local_n0_proc[nprocs_1-1];

  //local_transpose(nprocs_1,local_n1,n_tuples*local_n0,send_recv,data );

  int last_ntuples=0,first_ntuples=T_plan->local_n0_proc[0]*n_tuples;
  if(local_n1!=0)
    last_ntuples=T_plan->last_recv_count/((int)local_n1);

  for(int h=0;h<howmany;h++){
    if(local_n1==1)
      cudaMemcpy(&data[h*odist],&send_recv[h*odist],T_plan->alloc_local/howmany ,cudaMemcpyDeviceToDevice); // you are done, no additional transpose is needed.
    else if(last_ntuples!=first_ntuples){
      local_transpose_cuda((nprocs_0-1),local_n1,first_ntuples,&send_recv[h*odist] );
      local_transpose_cuda(2,local_n1,(nprocs_0-1)*first_ntuples,last_ntuples,&send_recv[h*odist],&data[h*odist] );
    }
    else if(last_ntuples==first_ntuples){
      //local_transpose_cuda(nprocs_0,local_n1,n_tuples*local_n0,send_recv );
      local_transpose_cuda(nprocs_0,local_n1,first_ntuples,&send_recv[h*odist],&data[h*odist] );
    }
  }
  cudaDeviceSynchronize();



  if(VERBOSE>=2) PCOUT<<"2nd Transpose"<<std::endl;
  if(VERBOSE>=2){
    cudaMemcpy(data_cpu, data, T_plan->alloc_local, cudaMemcpyDeviceToHost);
    for(int h=0;h<howmany;h++)
      for(int id=0;id<nprocs_1;++id){
        if(procid==id)
          for(int i=0;i<local_n1;i++){
            std::cout<<std::endl;
            for(int j=0;j<N[0];j++){
              std::cout<<'\t'<<data_cpu[h*odist+(i*N[0]+j)*n_tuples];
            }
          }
        std::cout<<'\n';
        MPI_Barrier(T_plan->comm);
      }
  }
  if(Flags[1]==1){ // Transpose output
    for(int h=0;h<howmany;h++)
      local_transpose_cuda(local_n1,N[0],n_tuples,&data[h*odist] );
  }
  reshuffle_time+=MPI_Wtime();
  MPI_Barrier(T_plan->comm);
  if(VERBOSE>=1){
    PCOUT<<"Shuffle Time= "<<shuffle_time<<std::endl;
    PCOUT<<"Alltoall Time= "<<comm_time<<std::endl;
    PCOUT<<"Reshuffle Time= "<<reshuffle_time<<std::endl;
    PCOUT<<"Total Time= "<<(shuffle_time+comm_time+reshuffle_time)<<std::endl;
  }
  timings[0]+=MPI_Wtime();//timings[0]+=shuffle_time+comm_time+reshuffle_time;
  timings[1]+=shuffle_time;
  timings[2]+=comm_time;
  timings[3]+=reshuffle_time;
  return;
}

void transpose_cuda_v5_2(T_Plan_gpu* T_plan, double * data, double *timings, unsigned flags, int howmany, int tag ){

  std::bitset<8> Flags(flags); // 1 Transposed in, 2 Transposed out
  if(Flags[1]==1 && Flags[0]==0 && T_plan->nprocs==1){ // If Flags==Transposed_Out return
    MPI_Barrier(T_plan->comm);
    return;
  }
  timings[0]-=MPI_Wtime();
  int nprocs, procid;
  int nprocs_0, nprocs_1;
  nprocs=T_plan->nprocs;
  procid=T_plan->procid;
  nprocs_0=T_plan->nprocs_0;
  nprocs_1=T_plan->nprocs_1;
  ptrdiff_t  *N=T_plan->N;
  double* data_cpu=T_plan->buffer;
  double * send_recv_cpu = T_plan->buffer_2;
  double * send_recv = T_plan->buffer_d;
  ptrdiff_t local_n0=T_plan->local_n0;
  ptrdiff_t local_n1=T_plan->local_n1;
  ptrdiff_t n_tuples=T_plan->n_tuples;
  int idist=N[1]*local_n0*n_tuples;
  int odist=N[0]*local_n1*n_tuples;

  double comm_time=0*MPI_Wtime(), shuffle_time=0*MPI_Wtime(), reshuffle_time=0*MPI_Wtime(), total_time=0*MPI_Wtime();

  if(VERBOSE>=2) PCOUT<<"INPUT:"<<std::endl;
  if(VERBOSE>=2){
    cudaMemcpy(data_cpu, data, T_plan->alloc_local, cudaMemcpyDeviceToHost);
    for(int h=0;h<howmany;h++)
      for(int id=0;id<nprocs;++id){
        if(procid==id)
          for(int i=0;i<local_n0;i++){
            std::cout<<std::endl;
            for(int j=0;j<N[1];j++){
              std::cout<<'\t'<<data_cpu[h*idist+(i*N[1]+j)*n_tuples];
            }
          }
        std::cout<<'\n';
        MPI_Barrier(T_plan->comm);
      }
  }
  //PCOUT<<" ============================================= "<<std::endl;
  //PCOUT<<" ==============   Local Transpose============= "<<std::endl;
  //PCOUT<<" ============================================= "<<std::endl;
  shuffle_time-=MPI_Wtime();

  if(Flags[0]==0){
    for(int h=0;h<howmany;h++)
      local_transpose_cuda(local_n0,N[1],n_tuples,&data[h*idist] );
  }
  cudaDeviceSynchronize();

  shuffle_time+=MPI_Wtime();
  if(VERBOSE>=2) PCOUT<<"Local Transpose:"<<std::endl;
  if(VERBOSE>=2){
    cudaMemcpy(data_cpu, data, T_plan->alloc_local, cudaMemcpyDeviceToHost);
    for(int h=0;h<howmany;h++)
      for(int id=0;id<nprocs;++id){
        if(procid==id)
          for(int i=0;i<N[1];i++){
            std::cout<<std::endl;
            for(int j=0;j<local_n0;j++){
              std::cout<<'\t'<<data_cpu[h*idist+(i*local_n0+j)*n_tuples];
            }
          }
        std::cout<<'\n';
        MPI_Barrier(T_plan->comm);
      }
  }
  if(nprocs==1 && Flags[0]==1 && Flags[1]==1){
    for(int h=0;h<howmany;h++)
      local_transpose_cuda(local_n1,N[0],n_tuples,&data[h*idist] );
  }
  if(nprocs==1){ // Transpose is done!
    timings[0]+=MPI_Wtime();
    timings[1]+=shuffle_time;
    timings[2]+=0;
    timings[3]+=0;
    return;
  }


  //PCOUT<<" ============================================= "<<std::endl;
  //PCOUT<<" ==============   MPIALLTOALL  =============== "<<std::endl;
  //PCOUT<<" ============================================= "<<std::endl;
  int* scount_proc=  T_plan->scount_proc;
  int* rcount_proc=  T_plan->rcount_proc;
  int* soffset_proc= T_plan->soffset_proc;
  int* roffset_proc= T_plan->roffset_proc;

  comm_time-=MPI_Wtime();

  int soffset=0,roffset=0;
  MPI_Status ierr;
  MPI_Request request[nprocs], s_request[nprocs];
  int flag[nprocs],color[nprocs];
  memset(flag,0,sizeof(int)*nprocs);
  memset(color,0,sizeof(int)*nprocs);
#pragma omp parallel for
  for (int proc=0;proc<nprocs;++proc){
    request[proc]=MPI_REQUEST_NULL;
    s_request[proc]=MPI_REQUEST_NULL;
  }

  int counter=1;
  MPI_Datatype *stype=T_plan->stype;
  MPI_Datatype *rtype=T_plan->rtype;
  // Post Receives
  for (int proc=0;proc<nprocs;++proc){
    if(proc!=procid){
      roffset=roffset_proc[proc];
      MPI_Irecv(&send_recv_cpu[roffset],1, rtype[proc], proc,
          tag, T_plan->comm, &request[proc]);
    }
  }
  // SEND
  //cudaMemcpy(data_cpu, data, T_plan->alloc_local, cudaMemcpyDeviceToHost);
  for (int proc=0;proc<nprocs;++proc){
    if(proc!=procid){
      soffset=soffset_proc[proc];
      for(int h=0;h<howmany;h++)
        cudaMemcpy(&data_cpu[h*idist+soffset], &data[h*idist+soffset],sizeof(double)*scount_proc[proc], cudaMemcpyDeviceToHost);
      MPI_Isend(&data_cpu[soffset],1, stype[proc],proc, tag,
          T_plan->comm, &s_request[proc]);

    }
    else{  // copy your own part directly
      for(int h=0;h<howmany;h++)
        cudaMemcpy(&send_recv[h*odist+roffset_proc[procid]], &data[h*idist+soffset_proc[procid]],sizeof(double)*scount_proc[procid], cudaMemcpyDeviceToDevice);
    }
  }
  while(counter!=nprocs){

    for (int proc=0;proc<nprocs;++proc){
      MPI_Test(&request[proc], &flag[proc],&ierr);
      if(flag[proc]==1 && color[proc]==0 && proc!=procid){
        for(int h=0;h<howmany;h++)
          cudaMemcpyAsync(&send_recv[h*odist+roffset_proc[proc]],&send_recv_cpu[h*odist+roffset_proc[proc]],sizeof(double)*rcount_proc[proc],cudaMemcpyHostToDevice);
        color[proc]=1;
        counter+=1;
      }
    }

  }
  cudaDeviceSynchronize();
  //cudaMemcpy(send_recv, send_recv_cpu, T_plan->alloc_local, cudaMemcpyHostToDevice);
  comm_time+=MPI_Wtime();

  if(VERBOSE>=2) PCOUT<<"MPIAlltoAll:"<<std::endl;
  if(VERBOSE>=2){
    cudaMemcpy(send_recv_cpu, send_recv, T_plan->alloc_local, cudaMemcpyDeviceToHost);
    for(int h=0;h<howmany;h++)
      for(int id=0;id<nprocs;++id){
        if(procid==id)
          for(int i=0;i<local_n1;i++){
            std::cout<<std::endl;
            for(int j=0;j<N[0];j++){
              std::cout<<'\t'<<send_recv_cpu[odist*h+(i*N[0]+j)*n_tuples];//<<","<<send_recv[(i*N[0]+j)*n_tuples+1];
            }
          }
        std::cout<<'\n';
        MPI_Barrier(T_plan->comm);
      }
  }



  //PCOUT<<" ============================================= "<<std::endl;
  //PCOUT<<" ============== 2nd Local Trnaspose ========== "<<std::endl;
  //PCOUT<<" ============================================= "<<std::endl;

  reshuffle_time-=MPI_Wtime();

  // int first_local_n0, last_local_n0;
  // first_local_n0=local_n0_proc[0]; last_local_n0=local_n0_proc[nprocs_1-1];

  //local_transpose(nprocs_1,local_n1,n_tuples*local_n0,send_recv,data );

  int last_ntuples=0,first_ntuples=T_plan->local_n0_proc[0]*n_tuples;
  if(local_n1!=0)
    last_ntuples=T_plan->last_recv_count/((int)local_n1);

  for(int h=0;h<howmany;h++){
    if(local_n1==1)
      cudaMemcpy(&data[h*odist],&send_recv[h*odist],T_plan->alloc_local/howmany ,cudaMemcpyDeviceToDevice); // you are done, no additional transpose is needed.
    else if(last_ntuples!=first_ntuples){
      local_transpose_cuda((nprocs_0-1),local_n1,first_ntuples,&send_recv[h*odist] );
      local_transpose_cuda(2,local_n1,(nprocs_0-1)*first_ntuples,last_ntuples,&send_recv[h*odist],&data[h*odist] );
    }
    else if(last_ntuples==first_ntuples){
      //local_transpose_cuda(nprocs_0,local_n1,n_tuples*local_n0,send_recv );
      local_transpose_cuda(nprocs_0,local_n1,first_ntuples,&send_recv[h*odist],&data[h*odist] );
    }
  }
  cudaDeviceSynchronize();


  if(VERBOSE>=2) PCOUT<<"2nd Transpose"<<std::endl;
  if(VERBOSE>=2){
    cudaMemcpy(data_cpu, data, T_plan->alloc_local, cudaMemcpyDeviceToHost);
    for(int h=0;h<howmany;h++)
      for(int id=0;id<nprocs_1;++id){
        if(procid==id)
          for(int i=0;i<local_n1;i++){
            std::cout<<std::endl;
            for(int j=0;j<N[0];j++){
              std::cout<<'\t'<<data_cpu[h*odist+(i*N[0]+j)*n_tuples];
            }
          }
        std::cout<<'\n';
        MPI_Barrier(T_plan->comm);
      }
  }
  if(Flags[1]==1){ // Transpose output
    for(int h=0;h<howmany;h++)
      local_transpose_cuda(local_n1,N[0],n_tuples,&data[h*odist] );
  }
  reshuffle_time+=MPI_Wtime();
  MPI_Barrier(T_plan->comm);
  if(VERBOSE>=1){
    PCOUT<<"Shuffle Time= "<<shuffle_time<<std::endl;
    PCOUT<<"Alltoall Time= "<<comm_time<<std::endl;
    PCOUT<<"Reshuffle Time= "<<reshuffle_time<<std::endl;
    PCOUT<<"Total Time= "<<(shuffle_time+comm_time+reshuffle_time)<<std::endl;
  }
  timings[0]+=MPI_Wtime();//timings[0]+=shuffle_time+comm_time+reshuffle_time;
  timings[1]+=shuffle_time;
  timings[2]+=comm_time;
  timings[3]+=reshuffle_time;
  return;
}


void transpose_cuda_v5_3(T_Plan_gpu* T_plan, double * data, double *timings, unsigned flags, int howmany , int tag){

  std::bitset<8> Flags(flags); // 1 Transposed in, 2 Transposed out
  if(Flags[1]==1 && Flags[0]==0 && T_plan->nprocs==1){ // If Flags==Transposed_Out return
    MPI_Barrier(T_plan->comm);
    return;
  }
  timings[0]-=MPI_Wtime();
  int nprocs, procid;
  int nprocs_0, nprocs_1;
  nprocs=T_plan->nprocs;
  procid=T_plan->procid;
  nprocs_0=T_plan->nprocs_0;
  nprocs_1=T_plan->nprocs_1;
  ptrdiff_t  *N=T_plan->N;
  double* data_cpu=T_plan->buffer;
  double * send_recv_cpu = T_plan->buffer_2;
  double * send_recv = T_plan->buffer_d;
  ptrdiff_t local_n0=T_plan->local_n0;
  ptrdiff_t local_n1=T_plan->local_n1;
  ptrdiff_t n_tuples=T_plan->n_tuples;
  int idist=N[1]*local_n0*n_tuples;
  int odist=N[0]*local_n1*n_tuples;

  double comm_time=0*MPI_Wtime(), shuffle_time=0*MPI_Wtime(), reshuffle_time=0*MPI_Wtime(), total_time=0*MPI_Wtime();

  if(VERBOSE>=2) PCOUT<<"INPUT:"<<std::endl;
  if(VERBOSE>=2){
    cudaMemcpy(data_cpu, data, T_plan->alloc_local, cudaMemcpyDeviceToHost);
    for(int h=0;h<howmany;h++)
      for(int id=0;id<nprocs;++id){
        if(procid==id)
          for(int i=0;i<local_n0;i++){
            std::cout<<std::endl;
            for(int j=0;j<N[1];j++){
              std::cout<<'\t'<<data_cpu[h*idist+(i*N[1]+j)*n_tuples];
            }
          }
        std::cout<<'\n';
        MPI_Barrier(T_plan->comm);
      }
  }
  //PCOUT<<" ============================================= "<<std::endl;
  //PCOUT<<" ==============   Local Transpose============= "<<std::endl;
  //PCOUT<<" ============================================= "<<std::endl;
  shuffle_time-=MPI_Wtime();

  if(Flags[0]==0){
    for(int h=0;h<howmany;h++)
      local_transpose_cuda(local_n0,N[1],n_tuples,&data[h*idist] );
  }
  cudaDeviceSynchronize();

  shuffle_time+=MPI_Wtime();
  if(VERBOSE>=2) PCOUT<<"Local Transpose:"<<std::endl;
  if(VERBOSE>=2){
    cudaMemcpy(data_cpu, data, T_plan->alloc_local, cudaMemcpyDeviceToHost);
    for(int h=0;h<howmany;h++)
      for(int id=0;id<nprocs;++id){
        if(procid==id)
          for(int i=0;i<N[1];i++){
            std::cout<<std::endl;
            for(int j=0;j<local_n0;j++){
              std::cout<<'\t'<<data_cpu[h*idist+(i*local_n0+j)*n_tuples];
            }
          }
        std::cout<<'\n';
        MPI_Barrier(T_plan->comm);
      }
  }
  if(nprocs==1 && Flags[0]==1 && Flags[1]==1){
    for(int h=0;h<howmany;h++)
      local_transpose_cuda(local_n1,N[0],n_tuples,&data[h*idist] );
  }
  if(nprocs==1){ // Transpose is done!
    timings[0]+=MPI_Wtime();
    timings[1]+=shuffle_time;
    timings[2]+=0;
    timings[3]+=0;
    return;
  }


  //PCOUT<<" ============================================= "<<std::endl;
  //PCOUT<<" ==============   MPIALLTOALL  =============== "<<std::endl;
  //PCOUT<<" ============================================= "<<std::endl;
  int* scount_proc=  T_plan->scount_proc;
  int* rcount_proc=  T_plan->rcount_proc;
  int* soffset_proc= T_plan->soffset_proc;
  int* roffset_proc= T_plan->roffset_proc;

  comm_time-=MPI_Wtime();

  int soffset=0,roffset=0;
  MPI_Status ierr;

  MPI_Datatype *stype=T_plan->stype;
  MPI_Datatype *rtype=T_plan->rtype;

  //cudaMemcpy(data_cpu, data, T_plan->alloc_local, cudaMemcpyDeviceToHost);
  for (int proc=0;proc<nprocs;++proc){
    if(proc!=procid){
      soffset=soffset_proc[proc];
      roffset=roffset_proc[proc];
      for(int h=0;h<howmany;h++)
        cudaMemcpy(&data_cpu[h*idist+soffset], &data[h*idist+soffset],sizeof(double)*scount_proc[proc], cudaMemcpyDeviceToHost);
      MPI_Sendrecv(&data_cpu[soffset],1, stype[proc],
          proc, tag,
          &send_recv_cpu[roffset],1, rtype[proc],
          proc, tag,
          T_plan->comm,&ierr);
      for(int h=0;h<howmany;h++)
        cudaMemcpyAsync(&send_recv[h*odist+roffset_proc[proc]],&send_recv_cpu[h*odist+roffset_proc[proc]],sizeof(double)*rcount_proc[proc],cudaMemcpyHostToDevice);

    }
    else{  // copy your own part directly
      for(int h=0;h<howmany;h++)
        cudaMemcpyAsync(&send_recv[h*odist+roffset_proc[procid]], &data[h*idist+soffset_proc[procid]],sizeof(double)*scount_proc[procid], cudaMemcpyDeviceToDevice);
    }
  }


  cudaDeviceSynchronize();
  //cudaMemcpy(send_recv, send_recv_cpu, T_plan->alloc_local, cudaMemcpyHostToDevice);
  comm_time+=MPI_Wtime();

  if(VERBOSE>=2) PCOUT<<"MPIAlltoAll:"<<std::endl;
  if(VERBOSE>=2){
    cudaMemcpy(send_recv_cpu, send_recv, T_plan->alloc_local, cudaMemcpyDeviceToHost);
    for(int h=0;h<howmany;h++)
      for(int id=0;id<nprocs;++id){
        if(procid==id)
          for(int i=0;i<local_n1;i++){
            std::cout<<std::endl;
            for(int j=0;j<N[0];j++){
              std::cout<<'\t'<<send_recv_cpu[odist*h+(i*N[0]+j)*n_tuples];//<<","<<send_recv[(i*N[0]+j)*n_tuples+1];
            }
          }
        std::cout<<'\n';
        MPI_Barrier(T_plan->comm);
      }
  }



  //PCOUT<<" ============================================= "<<std::endl;
  //PCOUT<<" ============== 2nd Local Trnaspose ========== "<<std::endl;
  //PCOUT<<" ============================================= "<<std::endl;

  reshuffle_time-=MPI_Wtime();

  // int first_local_n0, last_local_n0;
  // first_local_n0=local_n0_proc[0]; last_local_n0=local_n0_proc[nprocs_1-1];

  //local_transpose(nprocs_1,local_n1,n_tuples*local_n0,send_recv,data );

  int last_ntuples=0,first_ntuples=T_plan->local_n0_proc[0]*n_tuples;
  if(local_n1!=0)
    last_ntuples=T_plan->last_recv_count/((int)local_n1);

  for(int h=0;h<howmany;h++){
    if(local_n1==1)
      cudaMemcpy(&data[h*odist],&send_recv[h*odist],T_plan->alloc_local/howmany ,cudaMemcpyDeviceToDevice); // you are done, no additional transpose is needed.
    else if(last_ntuples!=first_ntuples){
      local_transpose_cuda((nprocs_0-1),local_n1,first_ntuples,&send_recv[h*odist] );
      local_transpose_cuda(2,local_n1,(nprocs_0-1)*first_ntuples,last_ntuples,&send_recv[h*odist],&data[h*odist] );
    }
    else if(last_ntuples==first_ntuples){
      //local_transpose_cuda(nprocs_0,local_n1,n_tuples*local_n0,send_recv );
      local_transpose_cuda(nprocs_0,local_n1,first_ntuples,&send_recv[h*odist],&data[h*odist] );
    }
  }
  cudaDeviceSynchronize();


  if(VERBOSE>=2) PCOUT<<"2nd Transpose"<<std::endl;
  if(VERBOSE>=2){
    cudaMemcpy(data_cpu, data, T_plan->alloc_local, cudaMemcpyDeviceToHost);
    for(int h=0;h<howmany;h++)
      for(int id=0;id<nprocs_1;++id){
        if(procid==id)
          for(int i=0;i<local_n1;i++){
            std::cout<<std::endl;
            for(int j=0;j<N[0];j++){
              std::cout<<'\t'<<data_cpu[h*odist+(i*N[0]+j)*n_tuples];
            }
          }
        std::cout<<'\n';
        MPI_Barrier(T_plan->comm);
      }
  }
  if(Flags[1]==1){ // Transpose output
    for(int h=0;h<howmany;h++)
      local_transpose_cuda(local_n1,N[0],n_tuples,&data[h*odist] );
  }
  reshuffle_time+=MPI_Wtime();
  MPI_Barrier(T_plan->comm);
  if(VERBOSE>=1){
    PCOUT<<"Shuffle Time= "<<shuffle_time<<std::endl;
    PCOUT<<"Alltoall Time= "<<comm_time<<std::endl;
    PCOUT<<"Reshuffle Time= "<<reshuffle_time<<std::endl;
    PCOUT<<"Total Time= "<<(shuffle_time+comm_time+reshuffle_time)<<std::endl;
  }
  timings[0]+=MPI_Wtime();//timings[0]+=shuffle_time+comm_time+reshuffle_time;
  timings[1]+=shuffle_time;
  timings[2]+=comm_time;
  timings[3]+=reshuffle_time;
  return;
}



void transpose_cuda_v6(T_Plan_gpu* T_plan, double * data, double *timings, unsigned flags, int howmany, int tag ){

  std::bitset<8> Flags(flags); // 1 Transposed in, 2 Transposed out
  if(Flags[1]==1 && Flags[0]==0 && T_plan->nprocs==1){ // If Flags==Transposed_Out return
    MPI_Barrier(T_plan->comm);
    return;
  }
  timings[0]-=MPI_Wtime();
  int nprocs, procid;
  int nprocs_0, nprocs_1;
  nprocs=T_plan->nprocs;
  procid=T_plan->procid;
  nprocs_0=T_plan->nprocs_0;
  nprocs_1=T_plan->nprocs_1;
  ptrdiff_t  *N=T_plan->N;
  double* data_cpu=T_plan->buffer;
  double * send_recv_cpu = T_plan->buffer_2;
  double * send_recv = T_plan->buffer_d;
  ptrdiff_t local_n0=T_plan->local_n0;
  ptrdiff_t local_n1=T_plan->local_n1;
  ptrdiff_t n_tuples=T_plan->n_tuples;
  int idist=N[1]*local_n0*n_tuples;
  int odist=N[0]*local_n1*n_tuples;

  double comm_time=0*MPI_Wtime(), shuffle_time=0*MPI_Wtime(), reshuffle_time=0*MPI_Wtime(), total_time=0*MPI_Wtime();

  if(VERBOSE>=2) PCOUT<<"INPUT:"<<std::endl;
  if(VERBOSE>=2){
    cudaMemcpy(data_cpu, data, T_plan->alloc_local, cudaMemcpyDeviceToHost);
    for(int h=0;h<howmany;h++)
      for(int id=0;id<1+0*nprocs;++id){
        if(procid==id)
          for(int i=0;i<local_n0;i++){
            std::cout<<std::endl;
            for(int j=0;j<N[1];j++){
              std::cout<<'\t'<<data_cpu[h*idist+(i*N[1]+j)*n_tuples];
            }
          }
        std::cout<<'\n';
        MPI_Barrier(T_plan->comm);
      }
  }
  //PCOUT<<" ============================================= "<<std::endl;
  //PCOUT<<" ==============   Local Transpose============= "<<std::endl;
  //PCOUT<<" ============================================= "<<std::endl;
  shuffle_time-=MPI_Wtime();

  if(Flags[0]==0){
    for(int h=0;h<howmany;h++)
      local_transpose_cuda(local_n0,N[1],n_tuples,&data[h*idist] );
  }
  cudaDeviceSynchronize();

  shuffle_time+=MPI_Wtime();
  if(VERBOSE>=2) PCOUT<<"Local Transpose:"<<std::endl;
  if(VERBOSE>=2){
    cudaMemcpy(data_cpu, data, T_plan->alloc_local, cudaMemcpyDeviceToHost);
    for(int h=0;h<howmany;h++)
      for(int id=0;id<nprocs;++id){
        if(procid==id)
          for(int i=0;i<N[1];i++){
            std::cout<<std::endl;
            for(int j=0;j<local_n0;j++){
              std::cout<<'\t'<<data_cpu[h*idist+(i*local_n0+j)*n_tuples];
            }
          }
        std::cout<<'\n';
        MPI_Barrier(T_plan->comm);
      }
  }
  if(nprocs==1 && Flags[0]==1 && Flags[1]==1){
    for(int h=0;h<howmany;h++)
      local_transpose_cuda(local_n1,N[0],n_tuples,&data[h*idist] );
  }
  if(nprocs==1){ // Transpose is done!
    timings[0]+=MPI_Wtime();
    timings[1]+=shuffle_time;
    timings[2]+=0;
    timings[3]+=0;
    return;
  }


  //PCOUT<<" ============================================= "<<std::endl;
  //PCOUT<<" ==============   MPIALLTOALL  =============== "<<std::endl;
  //PCOUT<<" ============================================= "<<std::endl;
  int* scount_proc=  T_plan->scount_proc;
  int* rcount_proc=  T_plan->rcount_proc;
  int* soffset_proc= T_plan->soffset_proc;
  int* roffset_proc= T_plan->roffset_proc;

  MPI_Datatype *stype=T_plan->stype;
  MPI_Datatype *rtype=T_plan->rtype;
  MPI_Barrier(T_plan->comm);
  comm_time-=MPI_Wtime();
  // SEND
  cudaMemcpy(data_cpu, data, T_plan->alloc_local, cudaMemcpyDeviceToHost);
  if(howmany>1){
    MPI_Alltoallw(data_cpu,T_plan->scount_proc_w,
        T_plan->soffset_proc_w, stype,
        send_recv_cpu,T_plan->rcount_proc_w, T_plan->roffset_proc_w,
        rtype, T_plan->comm);
  }
  else if(T_plan->is_evenly_distributed==0)
    MPI_Alltoallv(data_cpu,scount_proc,
        soffset_proc, MPI_DOUBLE,send_recv_cpu,
        rcount_proc,roffset_proc, MPI_DOUBLE,
        T_plan->comm);
  else
    MPI_Alltoall(data_cpu, scount_proc[0], MPI_DOUBLE,
        send_recv_cpu, rcount_proc[0], MPI_DOUBLE,
        T_plan->comm);

  cudaMemcpy(send_recv, send_recv_cpu, T_plan->alloc_local, cudaMemcpyHostToDevice);
  comm_time+=MPI_Wtime();

  if(VERBOSE>=2) PCOUT<<"MPIAlltoAll:"<<std::endl;
  if(VERBOSE>=2)
    for(int h=0;h<howmany;h++)
      for(int id=0;id<nprocs;++id){
        if(procid==id)
          for(int i=0;i<local_n1;i++){
            std::cout<<std::endl;
            for(int j=0;j<N[0];j++){
              std::cout<<'\t'<<send_recv_cpu[h*odist+(i*N[0]+j)*n_tuples];
            }
          }
        std::cout<<'\n';
        MPI_Barrier(T_plan->comm);
      }




  //PCOUT<<" ============================================= "<<std::endl;
  //PCOUT<<" ============== 2nd Local Trnaspose ========== "<<std::endl;
  //PCOUT<<" ============================================= "<<std::endl;

  reshuffle_time-=MPI_Wtime();

  // int first_local_n0, last_local_n0;
  // first_local_n0=local_n0_proc[0]; last_local_n0=local_n0_proc[nprocs_1-1];

  //local_transpose(nprocs_1,local_n1,n_tuples*local_n0,send_recv,data );

  int last_ntuples=0,first_ntuples=T_plan->local_n0_proc[0]*n_tuples;
  if(local_n1!=0)
    last_ntuples=T_plan->last_recv_count/((int)local_n1);

  for(int h=0;h<howmany;h++){
    if(local_n1==1)
      cudaMemcpy(&data[h*odist],&send_recv[h*odist],T_plan->alloc_local/howmany ,cudaMemcpyDeviceToDevice); // you are done, no additional transpose is needed.
    else if(last_ntuples!=first_ntuples){
      local_transpose_cuda((nprocs_0-1),local_n1,first_ntuples,&send_recv[h*odist] );
      local_transpose_cuda(2,local_n1,(nprocs_0-1)*first_ntuples,last_ntuples,&send_recv[h*odist],&data[h*odist] );
    }
    else if(last_ntuples==first_ntuples){
      //local_transpose_cuda(nprocs_0,local_n1,n_tuples*local_n0,send_recv );
      local_transpose_cuda(nprocs_0,local_n1,first_ntuples,&send_recv[h*odist],&data[h*odist] );
    }
  }

  cudaDeviceSynchronize();


  if(VERBOSE>=2) PCOUT<<"2nd Transpose"<<std::endl;
  if(VERBOSE>=2){
    cudaMemcpy(data_cpu, data, T_plan->alloc_local, cudaMemcpyDeviceToHost);
    for(int h=0;h<howmany;h++)
      for(int id=0;id<nprocs_1;++id){
        if(procid==id)
          for(int i=0;i<local_n1;i++){
            std::cout<<std::endl;
            for(int j=0;j<N[0];j++){
              std::cout<<'\t'<<data_cpu[h*odist+(i*N[0]+j)*n_tuples];
            }
          }
        std::cout<<'\n';
        MPI_Barrier(T_plan->comm);
      }
  }
  if(Flags[1]==1){ // Transpose output
    for(int h=0;h<howmany;h++)
      local_transpose_cuda(local_n1,N[0],n_tuples,&data[h*odist] );
  }
  reshuffle_time+=MPI_Wtime();

  MPI_Barrier(T_plan->comm);
  if(VERBOSE>=1){
    PCOUT<<"Shuffle Time= "<<shuffle_time<<std::endl;
    PCOUT<<"Alltoall Time= "<<comm_time<<std::endl;
    PCOUT<<"Reshuffle Time= "<<reshuffle_time<<std::endl;
    PCOUT<<"Total Time= "<<(shuffle_time+comm_time+reshuffle_time)<<std::endl;
  }
  timings[0]+=MPI_Wtime();//timings[0]+=shuffle_time+comm_time+reshuffle_time;
  timings[1]+=shuffle_time;
  timings[2]+=comm_time;
  timings[3]+=reshuffle_time;
  return;
}
void transpose_cuda_v7(T_Plan_gpu* T_plan, double * data, double *timings,int kway, unsigned flags, int howmany, int tag ){

  std::bitset<8> Flags(flags); // 1 Transposed in, 2 Transposed out
  if(Flags[1]==1 && Flags[0]==0 && T_plan->nprocs==1){ // If Flags==Transposed_Out return
    MPI_Barrier(T_plan->comm);
    return;
  }
  timings[0]-=MPI_Wtime();
  int nprocs, procid;
  int nprocs_0, nprocs_1;
  nprocs=T_plan->nprocs;
  procid=T_plan->procid;
  nprocs_0=T_plan->nprocs_0;
  nprocs_1=T_plan->nprocs_1;
  ptrdiff_t  *N=T_plan->N;
  double* data_cpu=T_plan->buffer;
  double * send_recv_cpu = T_plan->buffer_2;
  double * send_recv = T_plan->buffer_d;
  ptrdiff_t local_n0=T_plan->local_n0;
  ptrdiff_t local_n1=T_plan->local_n1;
  ptrdiff_t n_tuples=T_plan->n_tuples;
  int idist=N[1]*local_n0*n_tuples;
  int odist=N[0]*local_n1*n_tuples;

  double comm_time=0*MPI_Wtime(), shuffle_time=0*MPI_Wtime(), reshuffle_time=0*MPI_Wtime(), total_time=0*MPI_Wtime();

  if(VERBOSE>=2) PCOUT<<"INPUT:"<<std::endl;
  if(VERBOSE>=2){
    cudaMemcpy(data_cpu, data, T_plan->alloc_local, cudaMemcpyDeviceToHost);
    for(int h=0;h<howmany;h++)
      for(int id=0;id<nprocs;++id){
        if(procid==id)
          for(int i=0;i<local_n0;i++){
            std::cout<<std::endl;
            for(int j=0;j<N[1];j++){
              std::cout<<'\t'<<data_cpu[h*idist+(i*N[1]+j)*n_tuples];
            }
          }
        std::cout<<'\n';
        MPI_Barrier(T_plan->comm);
      }
  }
  //PCOUT<<" ============================================= "<<std::endl;
  //PCOUT<<" ==============   Local Transpose============= "<<std::endl;
  //PCOUT<<" ============================================= "<<std::endl;
  shuffle_time-=MPI_Wtime();

  if(Flags[0]==0){
    for(int h=0;h<howmany;h++)
      local_transpose_cuda(local_n0,N[1],n_tuples,&data[h*idist] );
  }
  cudaDeviceSynchronize();

  shuffle_time+=MPI_Wtime();
  if(VERBOSE>=2) PCOUT<<"Local Transpose:"<<std::endl;
  if(VERBOSE>=2){
    cudaMemcpy(data_cpu, data, T_plan->alloc_local, cudaMemcpyDeviceToHost);
    for(int h=0;h<howmany;h++)
      for(int id=0;id<nprocs;++id){
        if(procid==id)
          for(int i=0;i<N[1];i++){
            std::cout<<std::endl;
            for(int j=0;j<local_n0;j++){
              std::cout<<'\t'<<data_cpu[h*idist+(i*local_n0+j)*n_tuples];
            }
          }
        std::cout<<'\n';
        MPI_Barrier(T_plan->comm);
      }
  }
  if(nprocs==1 && Flags[0]==1 && Flags[1]==1){
    for(int h=0;h<howmany;h++)
      local_transpose_cuda(local_n1,N[0],n_tuples,&data[h*idist] );
  }
  if(nprocs==1){ // Transpose is done!
    timings[0]+=MPI_Wtime();
    timings[1]+=shuffle_time;
    timings[2]+=0;
    timings[3]+=0;
    return;
  }


  //PCOUT<<" ============================================= "<<std::endl;
  //PCOUT<<" ==============   MPIALLTOALL  =============== "<<std::endl;
  //PCOUT<<" ============================================= "<<std::endl;
  int* scount_proc=  T_plan->scount_proc;
  int* rcount_proc=  T_plan->rcount_proc;
  int* soffset_proc= T_plan->soffset_proc;
  int* roffset_proc= T_plan->roffset_proc;

  comm_time-=MPI_Wtime();
  // SEND
  cudaMemcpy(data_cpu, data, T_plan->alloc_local, cudaMemcpyDeviceToHost);
  //if(T_plan->is_evenly_distributed==0)
  //MPI_Alltoallv(data_cpu,scount_proc,
  //                soffset_proc, MPI_DOUBLE,send_recv_cpu,
  //                rcount_proc,roffset_proc, MPI_DOUBLE,
  //                T_plan->comm);
  //else
  //MPI_Alltoall(data_cpu, scount_proc[0], MPI_DOUBLE,
  //               send_recv_cpu, rcount_proc[0], MPI_DOUBLE,
  //               T_plan->comm);
  if(T_plan->kway_async)
    par::Mpi_Alltoallv_dense<double,true>(data_cpu , scount_proc, soffset_proc,
        send_recv_cpu, rcount_proc, roffset_proc, T_plan->comm,kway);
  else
    par::Mpi_Alltoallv_dense<double,false>(data_cpu , scount_proc, soffset_proc,
        send_recv_cpu, rcount_proc, roffset_proc, T_plan->comm,kway);


  cudaMemcpy(send_recv, send_recv_cpu, T_plan->alloc_local, cudaMemcpyHostToDevice);
  comm_time+=MPI_Wtime();

  if(VERBOSE>=2) PCOUT<<"MPIAlltoAll:"<<std::endl;
  if(VERBOSE>=2)
    for(int h=0;h<howmany;h++)
      for(int id=0;id<nprocs;++id){
        if(procid==id)
          for(int i=0;i<local_n1;i++){
            std::cout<<std::endl;
            for(int j=0;j<N[0];j++){
              std::cout<<'\t'<<send_recv_cpu[h*odist+(i*N[0]+j)*n_tuples];//<<","<<send_recv[(i*N[0]+j)*n_tuples+1];
            }
          }
        std::cout<<'\n';
        MPI_Barrier(T_plan->comm);
      }




  //PCOUT<<" ============================================= "<<std::endl;
  //PCOUT<<" ============== 2nd Local Trnaspose ========== "<<std::endl;
  //PCOUT<<" ============================================= "<<std::endl;

  reshuffle_time-=MPI_Wtime();

  // int first_local_n0, last_local_n0;
  // first_local_n0=local_n0_proc[0]; last_local_n0=local_n0_proc[nprocs_1-1];

  //local_transpose(nprocs_1,local_n1,n_tuples*local_n0,send_recv,data );

  int last_ntuples=0,first_ntuples=T_plan->local_n0_proc[0]*n_tuples;
  if(local_n1!=0)
    last_ntuples=T_plan->last_recv_count/((int)local_n1);

  for(int h=0;h<howmany;h++){
    if(local_n1==1)
      cudaMemcpy(&data[h*odist],&send_recv[h*odist],T_plan->alloc_local/howmany ,cudaMemcpyDeviceToDevice); // you are done, no additional transpose is needed.
    else if(last_ntuples!=first_ntuples){
      local_transpose_cuda((nprocs_0-1),local_n1,first_ntuples,&send_recv[h*odist] );
      local_transpose_cuda(2,local_n1,(nprocs_0-1)*first_ntuples,last_ntuples,&send_recv[h*odist],&data[h*odist] );
    }
    else if(last_ntuples==first_ntuples){
      //local_transpose_cuda(nprocs_0,local_n1,n_tuples*local_n0,send_recv );
      local_transpose_cuda(nprocs_0,local_n1,first_ntuples,&send_recv[h*odist],&data[h*odist] );
    }
  }
  cudaDeviceSynchronize();

  if(Flags[1]==1){ // Transpose output
    for(int h=0;h<howmany;h++)
      local_transpose_cuda(local_n1,N[0],n_tuples,&data[h*odist] );
  }
  reshuffle_time+=MPI_Wtime();

  if(VERBOSE>=2) PCOUT<<"2nd Transpose"<<std::endl;
  if(VERBOSE>=2){
    cudaMemcpy(data_cpu, data, T_plan->alloc_local, cudaMemcpyDeviceToHost);
    for(int h=0;h<howmany;h++)
      for(int id=0;id<nprocs_1;++id){
        if(procid==id)
          for(int i=0;i<local_n1;i++){
            std::cout<<std::endl;
            for(int j=0;j<N[0];j++){
              std::cout<<'\t'<<data_cpu[h*odist+(i*N[0]+j)*n_tuples];
            }
          }
        std::cout<<'\n';
        MPI_Barrier(T_plan->comm);
      }
  }
  MPI_Barrier(T_plan->comm);
  if(VERBOSE>=1){
    PCOUT<<"Shuffle Time= "<<shuffle_time<<std::endl;
    PCOUT<<"Alltoall Time= "<<comm_time<<std::endl;
    PCOUT<<"Reshuffle Time= "<<reshuffle_time<<std::endl;
    PCOUT<<"Total Time= "<<(shuffle_time+comm_time+reshuffle_time)<<std::endl;
  }
  timings[0]+=MPI_Wtime();//timings[0]+=shuffle_time+comm_time+reshuffle_time;
  timings[1]+=shuffle_time;
  timings[2]+=comm_time;
  timings[3]+=reshuffle_time;
  return;
}
void transpose_cuda_v7_2(T_Plan_gpu* T_plan, double * data, double *timings,int kway, unsigned flags, int howmany, int tag ){

  std::bitset<8> Flags(flags); // 1 Transposed in, 2 Transposed out
  if(Flags[1]==1 && Flags[0]==0 && T_plan->nprocs==1){ // If Flags==Transposed_Out return
    MPI_Barrier(T_plan->comm);
    return;
  }
  timings[0]-=MPI_Wtime();
  int nprocs, procid;
  int nprocs_0, nprocs_1;
  nprocs=T_plan->nprocs;
  procid=T_plan->procid;
  nprocs_0=T_plan->nprocs_0;
  nprocs_1=T_plan->nprocs_1;
  ptrdiff_t  *N=T_plan->N;
  double* data_cpu=T_plan->buffer;
  double * send_recv_cpu = T_plan->buffer_2;
  double * send_recv = T_plan->buffer_d;
  ptrdiff_t local_n0=T_plan->local_n0;
  ptrdiff_t local_n1=T_plan->local_n1;
  ptrdiff_t n_tuples=T_plan->n_tuples;
  int idist=N[1]*local_n0*n_tuples;
  int odist=N[0]*local_n1*n_tuples;

  double comm_time=0*MPI_Wtime(), shuffle_time=0*MPI_Wtime(), reshuffle_time=0*MPI_Wtime(), total_time=0*MPI_Wtime();

  if(VERBOSE>=2) PCOUT<<"INPUT:"<<std::endl;
  if(VERBOSE>=2){
    cudaMemcpy(data_cpu, data, T_plan->alloc_local, cudaMemcpyDeviceToHost);
    for(int h=0;h<howmany;h++)
      for(int id=0;id<nprocs;++id){
        if(procid==id)
          for(int i=0;i<local_n0;i++){
            std::cout<<std::endl;
            for(int j=0;j<N[1];j++){
              std::cout<<'\t'<<data_cpu[h*idist+(i*N[1]+j)*n_tuples];
            }
          }
        std::cout<<'\n';
        MPI_Barrier(T_plan->comm);
      }
  }
  //PCOUT<<" ============================================= "<<std::endl;
  //PCOUT<<" ==============   Local Transpose============= "<<std::endl;
  //PCOUT<<" ============================================= "<<std::endl;
  shuffle_time-=MPI_Wtime();

  if(Flags[0]==0){
    for(int h=0;h<howmany;h++)
      local_transpose_cuda(local_n0,N[1],n_tuples,&data[h*idist] );
  }
  cudaDeviceSynchronize();

  shuffle_time+=MPI_Wtime();
  if(VERBOSE>=2) PCOUT<<"Local Transpose:"<<std::endl;
  if(VERBOSE>=2){
    cudaMemcpy(data_cpu, data, T_plan->alloc_local, cudaMemcpyDeviceToHost);
    for(int h=0;h<howmany;h++)
      for(int id=0;id<nprocs;++id){
        if(procid==id)
          for(int i=0;i<N[1];i++){
            std::cout<<std::endl;
            for(int j=0;j<local_n0;j++){
              std::cout<<'\t'<<data_cpu[h*idist+(i*local_n0+j)*n_tuples];
            }
          }
        std::cout<<'\n';
        MPI_Barrier(T_plan->comm);
      }
  }
  if(nprocs==1 && Flags[0]==1 && Flags[1]==1){
    for(int h=0;h<howmany;h++)
      local_transpose_cuda(local_n1,N[0],n_tuples,&data[h*idist] );
  }
  if(nprocs==1){ // Transpose is done!
    timings[0]+=MPI_Wtime();
    timings[1]+=shuffle_time;
    timings[2]+=0;
    timings[3]+=0;
    return;
  }


  //PCOUT<<" ============================================= "<<std::endl;
  //PCOUT<<" ==============   MPIALLTOALL  =============== "<<std::endl;
  //PCOUT<<" ============================================= "<<std::endl;
  int* scount_proc=  T_plan->scount_proc;
  int* rcount_proc=  T_plan->rcount_proc;
  int* soffset_proc= T_plan->soffset_proc;
  int* roffset_proc= T_plan->roffset_proc;

  comm_time-=MPI_Wtime();
  // SEND
  //  cudaMemcpy(data_cpu, data, T_plan->alloc_local, cudaMemcpyDeviceToHost);
  //if(T_plan->is_evenly_distributed==0)
  //MPI_Alltoallv(data_cpu,scount_proc,
  //                soffset_proc, MPI_DOUBLE,send_recv_cpu,
  //                rcount_proc,roffset_proc, MPI_DOUBLE,
  //                T_plan->comm);
  //else
  //MPI_Alltoall(data_cpu, scount_proc[0], MPI_DOUBLE,
  //               send_recv_cpu, rcount_proc[0], MPI_DOUBLE,
  //               T_plan->comm);
  if(T_plan->kway_async)
    par::Mpi_Alltoallv_dense_gpu<double,true>(data , scount_proc, soffset_proc,
        send_recv, rcount_proc, roffset_proc, T_plan->comm,kway);
  else
    par::Mpi_Alltoallv_dense_gpu<double,false>(data , scount_proc, soffset_proc,
        send_recv, rcount_proc, roffset_proc, T_plan->comm,kway);


  //  cudaMemcpy(send_recv, send_recv_cpu, T_plan->alloc_local, cudaMemcpyHostToDevice);
  comm_time+=MPI_Wtime();

  if(VERBOSE>=2) PCOUT<<"MPIAlltoAll:"<<std::endl;
  if(VERBOSE>=2)
    for(int h=0;h<howmany;h++)
      for(int id=0;id<nprocs;++id){
        if(procid==id)
          for(int i=0;i<local_n1;i++){
            std::cout<<std::endl;
            for(int j=0;j<N[0];j++){
              std::cout<<'\t'<<send_recv_cpu[h*odist+(i*N[0]+j)*n_tuples];//<<","<<send_recv[(i*N[0]+j)*n_tuples+1];
            }
          }
        std::cout<<'\n';
        MPI_Barrier(T_plan->comm);
      }




  //PCOUT<<" ============================================= "<<std::endl;
  //PCOUT<<" ============== 2nd Local Trnaspose ========== "<<std::endl;
  //PCOUT<<" ============================================= "<<std::endl;

  reshuffle_time-=MPI_Wtime();

  // int first_local_n0, last_local_n0;
  // first_local_n0=local_n0_proc[0]; last_local_n0=local_n0_proc[nprocs_1-1];

  //local_transpose(nprocs_1,local_n1,n_tuples*local_n0,send_recv,data );

  int last_ntuples=0,first_ntuples=T_plan->local_n0_proc[0]*n_tuples;
  if(local_n1!=0)
    last_ntuples=T_plan->last_recv_count/((int)local_n1);

  for(int h=0;h<howmany;h++){
    if(local_n1==1)
      cudaMemcpy(&data[h*odist],&send_recv[h*odist],T_plan->alloc_local/howmany ,cudaMemcpyDeviceToDevice); // you are done, no additional transpose is needed.
    else if(last_ntuples!=first_ntuples){
      local_transpose_cuda((nprocs_0-1),local_n1,first_ntuples,&send_recv[h*odist] );
      local_transpose_cuda(2,local_n1,(nprocs_0-1)*first_ntuples,last_ntuples,&send_recv[h*odist],&data[h*odist] );
    }
    else if(last_ntuples==first_ntuples){
      //local_transpose_cuda(nprocs_0,local_n1,n_tuples*local_n0,send_recv );
      local_transpose_cuda(nprocs_0,local_n1,first_ntuples,&send_recv[h*odist],&data[h*odist] );
    }
  }
  cudaDeviceSynchronize();

  if(Flags[1]==1){ // Transpose output
    for(int h=0;h<howmany;h++)
      local_transpose_cuda(local_n1,N[0],n_tuples,&data[h*odist] );
  }
  reshuffle_time+=MPI_Wtime();

  if(VERBOSE>=2) PCOUT<<"2nd Transpose"<<std::endl;
  if(VERBOSE>=2){
    cudaMemcpy(data_cpu, data, T_plan->alloc_local, cudaMemcpyDeviceToHost);
    for(int h=0;h<howmany;h++)
      for(int id=0;id<nprocs_1;++id){
        if(procid==id)
          for(int i=0;i<local_n1;i++){
            std::cout<<std::endl;
            for(int j=0;j<N[0];j++){
              std::cout<<'\t'<<data_cpu[h*odist+(i*N[0]+j)*n_tuples];
            }
          }
        std::cout<<'\n';
        MPI_Barrier(T_plan->comm);
      }
  }
  MPI_Barrier(T_plan->comm);
  if(VERBOSE>=1){
    PCOUT<<"Shuffle Time= "<<shuffle_time<<std::endl;
    PCOUT<<"Alltoall Time= "<<comm_time<<std::endl;
    PCOUT<<"Reshuffle Time= "<<reshuffle_time<<std::endl;
    PCOUT<<"Total Time= "<<(shuffle_time+comm_time+reshuffle_time)<<std::endl;
  }
  timings[0]+=MPI_Wtime();//timings[0]+=shuffle_time+comm_time+reshuffle_time;
  timings[1]+=shuffle_time;
  timings[2]+=comm_time;
  timings[3]+=reshuffle_time;
  return;
}
