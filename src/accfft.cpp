
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


int accfft_init(int nthreads){
  int threads_ok;
  if (threads_ok) threads_ok = fftw_init_threads();
  if (threads_ok) fftw_plan_with_nthreads(nthreads);
  return (!threads_ok);
}
void* accfft_alloc(ptrdiff_t size){
  void * ptr=fftw_malloc(size);
  return ptr;
}
void accfft_free(void * ptr){
  fftw_free(ptr);
  return;
}
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
int accfft_local_size_dft_r2c( int * n,int * isize, int * istart, int * osize, int *ostart,MPI_Comm c_comm, bool inplace){

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

  // 1D Decomposition
  if(plan->np[1]==1){
    unsigned fftw_flags=FFTW_MEASURE;
    int N0=n[0], N1=n[1], N2=n[2];

    int n_tuples_o,n_tuples_i;
    plan->inplace==true ? n_tuples_i=(N2/2+1)*2:  n_tuples_i=N2;
    n_tuples_o=(N2/2+1)*2;

    int isize[3],osize[3],istart[3],ostart[3];
    int alloc_max=accfft_local_size_dft_r2c(n,isize,istart,osize,ostart,c_comm,plan->inplace);
    plan->alloc_max=alloc_max;

    plan->Mem_mgr= new Mem_Mgr(N0,N1,n_tuples_o,c_comm);
    plan->T_plan_1= new T_Plan(N0,N1,n_tuples_o, plan->Mem_mgr, c_comm);
    plan->T_plan_1i= new T_Plan(N1,N0,n_tuples_o,plan->Mem_mgr, c_comm);

    plan->T_plan_1->alloc_local=alloc_max;
    plan->T_plan_1i->alloc_local=alloc_max;

    long int BATCH=plan->T_plan_1->local_n0;
    plan->fplan_0= fftw_plan_many_dft_r2c(2, &n[1],plan->T_plan_1->local_n0, //int rank, const int *n, int howmany
        data, NULL,					//double *in, const int *inembed,
        1, N1*n_tuples_i,			//int istride, int idist,
        (fftw_complex*)data_out, NULL,	//fftw_complex *out, const int *onembed,
        1, N1*n_tuples_o/2,				// int ostride, int odist,
        fftw_flags);
    if(plan->fplan_0==NULL) std::cout<<"!!! Forward Plan not Created !!!"<<std::endl;

    ptrdiff_t local_n0=plan->T_plan_1->local_n0;
    ptrdiff_t local_n1=plan->T_plan_1->local_n1;
    plan->fplan_1= fftw_plan_many_dft(1, &n[0],n_tuples_o/2*local_n1, //int rank, const int *n, int howmany
        (fftw_complex*)data_out, NULL,					//double *in, const int *inembed,
        local_n1*n_tuples_o/2,1,			//int istride, int idist,
        (fftw_complex*)data_out, NULL,	//fftw_complex *out, const int *onembed,
        local_n1*n_tuples_o/2,1,				// int ostride, int odist,
        FFTW_FORWARD,fftw_flags);
    if(plan->fplan_1==NULL) std::cout<<"!!! Forward Plan not Created !!!"<<std::endl;

    plan->iplan_0=fftw_plan_many_dft_c2r(2, &n[1],plan->T_plan_1->local_n0,	//int rank, const int *n, int howmany,
        (fftw_complex*)data_out, NULL,		//fftw_complex *in, const int *inembed,
        1, N1*n_tuples_o/2,					//int istride, int idist,
        data, NULL,						//double *out, const int *onembed,
        1, N1*n_tuples_i,				//int ostride, int odist,
        fftw_flags);
    if(plan->iplan_0==NULL) std::cout<<"!!! Inverse Plan not Created !!!"<<std::endl;

    plan->iplan_1= fftw_plan_many_dft(1, &n[0],n_tuples_o/2*local_n1, //int rank, const int *n, int howmany
        (fftw_complex*)data_out, NULL,					//double *in, const int *inembed,
        local_n1*n_tuples_o/2,1,			//int istride, int idist,
        (fftw_complex*)data_out, NULL,	//fftw_complex *out, const int *onembed,
        local_n1*n_tuples_o/2,1,				// int ostride, int odist,
        FFTW_BACKWARD,fftw_flags);
    if(plan->iplan_1==NULL) std::cout<<"!!! Inverse Plan2 not Created !!!"<<std::endl;

    static int method_static=0;
    static int kway_static_2=0;
    if(method_static==0){
      plan->T_plan_1->which_fast_method(plan->T_plan_1,data_out);
      method_static=plan->T_plan_1->method;
      kway_static_2=plan->T_plan_1->kway;
    }
    else{
      plan->T_plan_1->method=method_static;
      plan->T_plan_1->kway=kway_static_2;
    }
    plan->T_plan_1->method=plan->T_plan_1->method;
      plan->T_plan_1->kway=kway_static_2;
    plan->T_plan_1i->method=plan->T_plan_1->method;
      plan->T_plan_1i->kway=kway_static_2;
    plan->data=data;

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
    int alloc_max=0;
    int n_tuples_o,n_tuples_i;
    plan->inplace==true ? n_tuples_i=(n[2]/2+1)*2:  n_tuples_i=n[2];
    n_tuples_o=(n[2]/2+1)*2;

    int isize[3],osize[3],istart[3],ostart[3];
    alloc_max=accfft_local_size_dft_r2c(n,isize,istart,osize,ostart,c_comm,plan->inplace);

    dfft_get_local_size(n[0],n[1],n_tuples_o,osize_0,ostart_0,c_comm);
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

    // the reaseon for n_tuples/2 is to avoid splitting of imag and real parts of complex numbers
    plan->Mem_mgr= new Mem_Mgr(n[1],n_tuples_o/2,2,plan->row_comm,osize_0[0],alloc_max);
    plan->T_plan_1= new T_Plan(n[1],n_tuples_o/2,2, plan->Mem_mgr, plan->row_comm,osize_0[0]);
    plan->T_plan_2= new T_Plan(n[0],n[1],osize_2[2]*2, plan->Mem_mgr, plan->col_comm);
    plan->T_plan_2i= new T_Plan(n[1],n[0],osize_2i[2]*2, plan->Mem_mgr, plan->col_comm);
    plan->T_plan_1i= new T_Plan(n_tuples_o/2,n[1],2, plan->Mem_mgr, plan->row_comm,osize_1i[0]);

    plan->T_plan_1->alloc_local=plan->alloc_max;
    plan->T_plan_2->alloc_local=plan->alloc_max;
    plan->T_plan_2i->alloc_local=plan->alloc_max;
    plan->T_plan_1i->alloc_local=plan->alloc_max;

    {
      unsigned fftw_flags=FFTW_MEASURE;
      plan->fplan_0= fftw_plan_many_dft_r2c(1, &n[2],osize_0[0]*osize_0[1], //int rank, const int *n, int howmany
          data, NULL,					//double *in, const int *inembed,
          1, n_tuples_i,			//int istride, int idist,
          (fftw_complex*)data_out, NULL,	//fftw_complex *out, const int *onembed,
          1, n_tuples_o/2,				// int ostride, int odist,
          fftw_flags);
      if(plan->fplan_0==NULL) std::cout<<"!!! fplan_0 not Created !!!"<<std::endl;

      plan->iplan_0= fftw_plan_many_dft_c2r(1, &n[2],osize_0[0]*osize_0[1], //int rank, const int *n, int howmany
          (fftw_complex*)data_out, NULL,					//double *in, const int *inembed,
          1, n_tuples_o/2,			//int istride, int idist,
          data, NULL,	//fftw_complex *out, const int *onembed,
          1, n_tuples_i,				// int ostride, int odist,
          fftw_flags);
      if(plan->iplan_0==NULL) std::cout<<"!!! iplan_0 not Created !!!"<<std::endl;

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
      if(plan->fplan_1==NULL) std::cout<<"!!! fplan1 not Created !!!"<<std::endl;
      plan->iplan_1=fftw_plan_guru_dft(
          1, &dims,
          2,howmany_dims,
          (fftw_complex*)data_out, (fftw_complex*)data_out,
          1,fftw_flags);
      if(plan->iplan_1==NULL) std::cout<<"!!! fplan1 not Created !!!"<<std::endl;

      // ----
      dims.n=n[0];
      dims.is=osize_2[2];
      dims.os=osize_2[2];

      howmany_dims[0].n=osize_2[2];
      howmany_dims[0].is=1;
      howmany_dims[0].os=1;

      howmany_dims[1].n=osize_2[0];
      howmany_dims[1].is=osize_2[1]*osize_2[2];
      howmany_dims[1].os=osize_2[1]*osize_2[2];

      plan->fplan_2= fftw_plan_many_dft(1, &n[0],osize_2[2]*osize_2[1], //int rank, const int *n, int howmany
          (fftw_complex*)data_out, NULL,					//double *in, const int *inembed,
          osize_2[2]*osize_2[1],1,			//int istride, int idist,
          (fftw_complex*)data_out, NULL,	//fftw_complex *out, const int *onembed,
          osize_2[2]*osize_2[1],1,				// int ostride, int odist,
          FFTW_FORWARD,fftw_flags);
      if(plan->fplan_2==NULL) std::cout<<"!!! fplan2 not Created !!!"<<std::endl;

      plan->iplan_2= fftw_plan_many_dft(1, &n[0],osize_2[2]*osize_2[1], //int rank, const int *n, int howmany
          (fftw_complex*)data_out, NULL,					//double *in, const int *inembed,
          osize_2[2]*osize_2[1],1,			//int istride, int idist,
          (fftw_complex*)data_out, NULL,	//fftw_complex *out, const int *onembed,
          osize_2[2]*osize_2[1],1,				// int ostride, int odist,
          FFTW_BACKWARD,fftw_flags);
      if(plan->iplan_2==NULL) std::cout<<"!!! fplan2 not Created !!!"<<std::endl;
    }

    static int method_static_2=0;
    static int kway_static_2=0;
    if(method_static_2==0){
      if(coord[0]==0){
        plan->T_plan_1->which_fast_method(plan->T_plan_1,data_out);
        method_static_2=plan->T_plan_1->method;
        kway_static_2=plan->T_plan_1->kway;
      }
      MPI_Bcast(&method_static_2,1, MPI_INT,0, c_comm );
      MPI_Bcast(&kway_static_2,1, MPI_INT,0, c_comm );
      MPI_Barrier(c_comm);
    }
    plan->T_plan_1->method=method_static_2;
    plan->T_plan_2->method=method_static_2;
    plan->T_plan_2i->method=method_static_2;
    plan->T_plan_1i->method=method_static_2;
    plan->T_plan_1->kway=kway_static_2;
    plan->T_plan_2->kway=kway_static_2;
    plan->T_plan_2i->kway=kway_static_2;
    plan->T_plan_1i->kway=kway_static_2;
    plan->data=data;
  }
  return plan;

}

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

  // 1D Decomposition
  if (plan->np[1]==1){
    int NX=n[0],NY=n[1],NZ=n[2];
    plan->Mem_mgr= new Mem_Mgr(NX,NY,(NZ)*2,c_comm);
    plan->T_plan_1= new T_Plan(NX,NY,(NZ)*2, plan->Mem_mgr,c_comm);
    plan->T_plan_1i= new T_Plan(NY,NX,NZ*2, plan->Mem_mgr,c_comm);

    int isize[3],osize[3],istart[3],ostart[3];
    int alloc_max=accfft_local_size_dft_c2c(n,isize,istart,osize,ostart,c_comm);
    plan->alloc_max=alloc_max;
    plan->T_plan_1->alloc_local=alloc_max;
    plan->T_plan_1i->alloc_local=alloc_max;


    ptrdiff_t local_n0=plan->T_plan_1->local_n0;
    ptrdiff_t local_n1=plan->T_plan_1->local_n1;
    int N0=NX, N1=NY, N2=NZ;

    unsigned fftw_flags=FFTW_MEASURE;
    plan->fplan_0= fftw_plan_many_dft(2, &n[1],plan->T_plan_1->local_n0, //int rank, const int *n, int howmany
        data, NULL,					//double *in, const int *inembed,
        1, N1*N2,			//int istride, int idist,
        data_out, NULL,	//fftw_complex *out, const int *onembed,
        1, N1*N2,				// int ostride, int odist,
        FFTW_FORWARD,fftw_flags);
    if(plan->fplan_0==NULL) std::cout<<"!!! Forward Plan not Created !!!"<<std::endl;
    plan->iplan_0= fftw_plan_many_dft(2, &n[1],plan->T_plan_1->local_n0, //int rank, const int *n, int howmany
        data_out, NULL,					//double *in, const int *inembed,
        1, N1*N2,			//int istride, int idist,
        data, NULL,	//fftw_complex *out, const int *onembed,
        1, N1*N2,				// int ostride, int odist,
        FFTW_BACKWARD,fftw_flags);
    if(plan->iplan_0==NULL) std::cout<<"!!! Forward Plan not Created !!!"<<std::endl;

    MPI_Barrier(c_comm);

    plan->fplan_1= fftw_plan_many_dft(1, &n[0],plan->T_plan_1->local_n1*NZ, //int rank, const int *n, int howmany
        data_out, NULL,					//double *in, const int *inembed,
        plan->T_plan_1->local_n1*NZ, 1,			//int istride, int idist,
        data_out, NULL,	//fftw_complex *out, const int *onembed,
        plan->T_plan_1->local_n1*NZ, 1,				// int ostride, int odist,
        FFTW_FORWARD,fftw_flags);
    if(plan->fplan_1==NULL) std::cout<<"!!! Forward Plan not Created !!!"<<std::endl;

    plan->iplan_1= fftw_plan_many_dft(1, &n[0],plan->T_plan_1->local_n1*NZ, //int rank, const int *n, int howmany
        data_out, NULL,					//double *in, const int *inembed,
        plan->T_plan_1->local_n1*NZ, 1,			//int istride, int idist,
        data_out, NULL,	//fftw_complex *out, const int *onembed,
        plan->T_plan_1->local_n1*NZ, 1,				// int ostride, int odist,
        FFTW_BACKWARD,fftw_flags);
    if(plan->iplan_1==NULL) std::cout<<"!!! Forward Plan not Created !!!"<<std::endl;


    plan->T_plan_1->which_fast_method(plan->T_plan_1,(double*)data_out);
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




    // the reaseon for n_tuples/2 is to avoid splitting of imag and real parts of complex numbers

    plan->Mem_mgr=  new Mem_Mgr(n[1],n[2],2,plan->row_comm,osize_0[0],alloc_max);
    plan->T_plan_1= new T_Plan(n[1],n[2],2, plan->Mem_mgr, plan->row_comm,osize_0[0]);
    plan->T_plan_2= new T_Plan(n[0],n[1],2*osize_2[2], plan->Mem_mgr, plan->col_comm);
    plan->T_plan_2i=new T_Plan(n[1],n[0],2*osize_2i[2], plan->Mem_mgr, plan->col_comm);
    plan->T_plan_1i=new T_Plan(n[2],n[1],2, plan->Mem_mgr, plan->row_comm,osize_1i[0]);

    plan->T_plan_1->alloc_local=plan->alloc_max;
    plan->T_plan_2->alloc_local=plan->alloc_max;
    plan->T_plan_2i->alloc_local=plan->alloc_max;
    plan->T_plan_1i->alloc_local=plan->alloc_max;

    {
      // fplan_0
      unsigned fftw_flags=FFTW_MEASURE;
      plan->fplan_0= fftw_plan_many_dft(1, &n[2],osize_0[0]*osize_0[1], //int rank, const int *n, int howmany
          data, NULL,					//double *in, const int *inembed,
          1, n[2],			//int istride, int idist,
          data_out, NULL,	//fftw_complex *out, const int *onembed,
          1, n[2],				// int ostride, int odist,
          FFTW_FORWARD,fftw_flags);
      if(plan->fplan_0==NULL) std::cout<<"!!! Forward Plan not Created !!!"<<std::endl;
      plan->iplan_0= fftw_plan_many_dft(1, &n[2],osize_0[0]*osize_0[1], //int rank, const int *n, int howmany
          data_out, NULL,					//double *in, const int *inembed,
          1, n[2],			//int istride, int idist,
          data, NULL,	//fftw_complex *out, const int *onembed,
          1, n[2],				// int ostride, int odist,
          FFTW_BACKWARD,fftw_flags);
      if(plan->iplan_0==NULL) std::cout<<"!!! Forward Plan not Created !!!"<<std::endl;


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
      if(plan->fplan_1==NULL) std::cout<<"!!! fplan1 not Created !!!"<<std::endl;
      plan->iplan_1=fftw_plan_guru_dft(
          1, &dims,
          2,howmany_dims,
          (fftw_complex*)data_out, (fftw_complex*)data_out,
          1,fftw_flags);
      if(plan->iplan_1==NULL) std::cout<<"!!! iplan1 not Created !!!"<<std::endl;


      // fplan_2
      plan->fplan_2= fftw_plan_many_dft(1, &n[0],osize_2[1]*osize_2[2], //int rank, const int *n, int howmany
          data_out, NULL,					//double *in, const int *inembed,
          osize_2[1]*osize_2[2],1 ,			//int istride, int idist,
          data_out, NULL,	//fftw_complex *out, const int *onembed,
          osize_2[1]*osize_2[2],1 ,				// int ostride, int odist,
          FFTW_FORWARD,fftw_flags);
      if(plan->fplan_2==NULL) std::cout<<"!!! Forward Plan not Created !!!"<<std::endl;
      plan->iplan_2= fftw_plan_many_dft(1, &n[0],osize_2[1]*osize_2[2], //int rank, const int *n, int howmany
          data_out, NULL,					//double *in, const int *inembed,
          osize_2[1]*osize_2[2],1 ,			//int istride, int idist,
          data_out, NULL,	//fftw_complex *out, const int *onembed,
          osize_2[1]*osize_2[2],1 ,				// int ostride, int odist,
          FFTW_BACKWARD,fftw_flags);
      if(plan->iplan_2==NULL) std::cout<<"!!! Forward Plan not Created !!!"<<std::endl;

    }

    static int method_static_2=0;
    static int kway_static_2=0;
    if(method_static_2==0){
      if(coord[0]==0){
        plan->T_plan_1->which_fast_method(plan->T_plan_1,(double*)data_out);
        method_static_2=plan->T_plan_1->method;
        kway_static_2=plan->T_plan_1->kway;
      }
      MPI_Bcast(&method_static_2,1, MPI_INT,0, c_comm );
      MPI_Bcast(&kway_static_2,1, MPI_INT,0, c_comm );
      MPI_Barrier(c_comm);
    }
    plan->T_plan_1->method=method_static_2;
    plan->T_plan_2->method=method_static_2;
    plan->T_plan_2i->method=method_static_2;
    plan->T_plan_1i->method=method_static_2;
    plan->T_plan_1->kway=kway_static_2;
    plan->T_plan_2->kway=kway_static_2;
    plan->T_plan_2i->kway=kway_static_2;
    plan->T_plan_1i->kway=kway_static_2;
  }
  return plan;

}

void accfft_execute_r2c(accfft_plan* plan, double * data,Complex * data_out, double * timer){
  accfft_execute(plan,-1,data,(double*)data_out,timer);

  return;
}
void accfft_execute_c2r(accfft_plan* plan, Complex * data,double * data_out, double * timer){
  accfft_execute(plan,1,(double*)data,data_out,timer);

  return;
}

void accfft_execute(accfft_plan* plan, int direction,double * data,double * data_out, double * timer){

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
  if(plan->np[1]==1){
    if(direction==-1){

      /**************************************************************/
      /*******************  N0/P0 x N1 x N2 *************************/
      /**************************************************************/
      fft_time-=MPI_Wtime();
      fftw_execute_dft_r2c(plan->fplan_0,(double*)data,(fftw_complex*)data_out);
      fft_time+=MPI_Wtime();

      MPI_Barrier(plan->c_comm);
      plan->T_plan_1->execute(plan->T_plan_1,data_out,timings,2);
      /**************************************************************/
      /*******************  N1 x N0/P0 x N2 *************************/
      /**************************************************************/
      fft_time-=MPI_Wtime();
      fftw_execute_dft(plan->fplan_1,(fftw_complex*)data_out,(fftw_complex*)data_out);
      fft_time+=MPI_Wtime();
    }

    if(direction==1){
      /* Now Perform the inverse transform  */
      fft_time-=MPI_Wtime();
      fftw_execute_dft(plan->iplan_1,(fftw_complex*)data,(fftw_complex*)data);
      fft_time+=MPI_Wtime();

      plan->T_plan_1i->execute(plan->T_plan_1i,data,timings,1);
      /**************************************************************/
      /*******************  N0/P0 x N1 x N2 *************************/
      /**************************************************************/

      fft_time-=MPI_Wtime();
      fftw_execute_dft_c2r(plan->iplan_0,(fftw_complex*)data,(double*)data_out);
      fft_time+=MPI_Wtime();

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
      fft_time-=MPI_Wtime();
      fftw_execute_dft_r2c(plan->fplan_0,(double*)data,(fftw_complex*)data_out);
      fft_time+=MPI_Wtime();

      // Perform N0/P0 transpose


      plan->T_plan_1->execute(plan->T_plan_1,data_out,timings,2,osize_0[0],coords[0]);
      /**************************************************************/
      /*******************  N0/P0 x N1 x N2/P1 **********************/
      /**************************************************************/
      fft_time-=MPI_Wtime();
      fftw_execute_dft(plan->fplan_1,(fftw_complex*)data_out,(fftw_complex*)data_out);
      fft_time+=MPI_Wtime();



      plan->T_plan_2->execute(plan->T_plan_2,plan->data_out,timings,2,1,coords[1]);
      /**************************************************************/
      /*******************  N0 x N1/P0 x N2/P1 **********************/
      /**************************************************************/
      fft_time-=MPI_Wtime();
      fftw_execute_dft(plan->fplan_2,(fftw_complex*)data_out,(fftw_complex*)data_out);
      fft_time+=MPI_Wtime();
    }
    else if (direction==1){
      fft_time-=MPI_Wtime();
      fftw_execute_dft(plan->iplan_2,(fftw_complex*)data,(fftw_complex*)data);
      fft_time+=MPI_Wtime();


      plan->T_plan_2i->execute(plan->T_plan_2i,data,timings,1,1,coords[1]);
      MPI_Barrier(plan->c_comm);
      /**************************************************************/
      /*******************  N0/P0 x N1 x N2/P1 **********************/
      /**************************************************************/
      fft_time-=MPI_Wtime();
      fftw_execute_dft(plan->iplan_1,(fftw_complex*)data,(fftw_complex*)data);
      fft_time+=MPI_Wtime();


      plan->T_plan_1i->execute(plan->T_plan_1i,data,timings,1,osize_1i[0],coords[0]);
      MPI_Barrier(plan->c_comm);
      /**************************************************************/
      /*******************  N0/P0 x N1/P1 x N2 **********************/
      /**************************************************************/

      // IFFT in Z direction
      fft_time-=MPI_Wtime();
      fftw_execute_dft_c2r(plan->iplan_0,(fftw_complex*)data,(double*)data_out);
      fft_time+=MPI_Wtime();
      MPI_Barrier(plan->c_comm);

    }
  }
  timings[4]=fft_time;
  if(timer==NULL){
    delete [] timings;
  }
  MPI_Barrier(plan->c_comm);

  return;
}
void accfft_execute_c2c(accfft_plan* plan, int direction,Complex * data, Complex * data_out, double * timer){

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



  // 1D Decomposition
  if(plan->np[1]==1){
    if(direction==-1){

      /**************************************************************/
      /*******************  N0/P0 x N1 x N2 *************************/
      /**************************************************************/
      fft_time-=MPI_Wtime();
      fftw_execute_dft(plan->fplan_0,data,data_out);
      fft_time+=MPI_Wtime();

      MPI_Barrier(plan->c_comm);
      plan->T_plan_1->execute(plan->T_plan_1,(double*)data_out,timings,2);
      /**************************************************************/
      /*******************  N1 x N0/P0 x N2 *************************/
      /**************************************************************/
      fft_time-=MPI_Wtime();
      fftw_execute_dft(plan->fplan_1,(fftw_complex*)data_out,(fftw_complex*)data_out);
      fft_time+=MPI_Wtime();
    }

    if(direction==1){
      /* Now Perform the inverse transform  */
      fft_time-=MPI_Wtime();
      fftw_execute_dft(plan->iplan_1,(fftw_complex*)data,(fftw_complex*)data);
      fft_time+=MPI_Wtime();

      plan->T_plan_1i->execute(plan->T_plan_1i,(double*)data,timings,1);
      /**************************************************************/
      /*******************  N0/P0 x N1 x N2 *************************/
      /**************************************************************/

      fft_time-=MPI_Wtime();
      fftw_execute_dft(plan->iplan_0,data,data_out);
      fft_time+=MPI_Wtime();

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
      fft_time-=MPI_Wtime();
      fftw_execute_dft(plan->fplan_0,data,data_out);
      fft_time+=MPI_Wtime();



      plan->T_plan_1->execute(plan->T_plan_1,(double*)data_out,timings,2,osize_0[0],coords[0]);
      /**************************************************************/
      /*******************  N0/P0 x N1 x N2/P1 **********************/
      /**************************************************************/
      fft_time-=MPI_Wtime();
      fftw_execute_dft(plan->fplan_1,(fftw_complex*)data_out,(fftw_complex*)data_out);
      fft_time+=MPI_Wtime();



      plan->T_plan_2->execute(plan->T_plan_2,(double*)data_out,timings,2,1,coords[1]);
      /**************************************************************/
      /*******************  N0 x N1/P0 x N2/P1 **********************/
      /**************************************************************/
      fft_time-=MPI_Wtime();
      fftw_execute_dft(plan->fplan_2,(fftw_complex*)data_out,(fftw_complex*)data_out);
      fft_time+=MPI_Wtime();
    }
    else if (direction==1){
      fft_time-=MPI_Wtime();
      fftw_execute_dft(plan->iplan_2,(fftw_complex*)data,(fftw_complex*)data);
      fft_time+=MPI_Wtime();


      plan->T_plan_2i->execute(plan->T_plan_2i,(double*)data,timings,1,1,coords[1]);
      /**************************************************************/
      /*******************  N0/P0 x N1 x N2/P1 **********************/
      /**************************************************************/
      fft_time-=MPI_Wtime();
      fftw_execute_dft(plan->iplan_1,(fftw_complex*)data,(fftw_complex*)data);
      fft_time+=MPI_Wtime();


      plan->T_plan_1i->execute(plan->T_plan_1i,(double*)data,timings,1,osize_1i[0],coords[0]);
      MPI_Barrier(plan->c_comm);
      /**************************************************************/
      /*******************  N0/P0 x N1/P1 x N2 **********************/
      /**************************************************************/

      // IFFT in Z direction
      fft_time-=MPI_Wtime();
      fftw_execute_dft(plan->iplan_0,data,data_out);
      fft_time+=MPI_Wtime();

    }
  }
  timings[4]=fft_time;
  if(timer==NULL){
    delete [] timings;
  }
  MPI_Barrier(plan->c_comm);

  return;
}
void accfft_destroy_plan(accfft_plan * plan){

  if(plan->T_plan_1!=NULL)delete(plan->T_plan_1);
  if(plan->T_plan_1i!=NULL)delete(plan->T_plan_1i);
  if(plan->T_plan_2!=NULL)delete(plan->T_plan_2);
  if(plan->T_plan_2i!=NULL)delete(plan->T_plan_2i);
  if(plan->Mem_mgr!=NULL)delete(plan->Mem_mgr);
  if(plan->fplan_0!=NULL)fftw_destroy_plan(plan->fplan_0);
  if(plan->fplan_1!=NULL)fftw_destroy_plan(plan->fplan_1);
  if(plan->fplan_2!=NULL)fftw_destroy_plan(plan->fplan_2);
  if(plan->iplan_0!=NULL)fftw_destroy_plan(plan->iplan_0);
  if(plan->iplan_1!=NULL)fftw_destroy_plan(plan->iplan_1);
  if(plan->iplan_2!=NULL)fftw_destroy_plan(plan->iplan_2);

  MPI_Comm_free(&plan->row_comm);
  MPI_Comm_free(&plan->col_comm);
}
