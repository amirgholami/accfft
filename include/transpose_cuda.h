/*
 *  Copyright (c) 2014-2015, George Biros
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

#ifndef _TRANSPOSE_CUDA_
#define _TRANSPOSE_CUDA_

//#include "transpose.h"
#include <cuda.h>
#include <vector>

template <typename T>
class Mem_Mgr_gpu{
	public:
		Mem_Mgr_gpu(int N0, int N1, int n_tuples, MPI_Comm Comm, int howmany=1, int specified_alloc_local=0);

    ptrdiff_t N[2];
    ptrdiff_t n_tuples;
    //int procid,nprocs;
    ptrdiff_t alloc_local;

    ptrdiff_t local_n0;
    ptrdiff_t local_n1;

    bool PINNED;

    T* buffer;
    T* buffer_2;
    T* buffer_d;
    T* buffer_d2;
    //T * data_cpu;
    //MPI_Comm comm;
		// Deconstructor
		~Mem_Mgr_gpu();
};

template <typename T>
class T_Plan_gpu{
	public:
		T_Plan_gpu(int N0, int N1, int n_tuples,Mem_Mgr_gpu<T> * Mem_mgr,MPI_Comm, int howmany=1);
    void which_method_gpu(T_Plan_gpu<T>*,T*);
    void which_fast_method_gpu(T_Plan_gpu<T>*,T*, unsigned flags=2, int howmany=1, int tag=0);
    void execute_gpu(T_Plan_gpu<T>*,T*,double*, unsigned flags=0, int howmany=1, int tag=0);

    ptrdiff_t N[2];
    ptrdiff_t n_tuples;
    // Other procs local distribution
    ptrdiff_t *local_n0_proc;
    ptrdiff_t *local_n1_proc;
    ptrdiff_t *local_0_start_proc;
    ptrdiff_t *local_1_start_proc;

    // My local distribution
    ptrdiff_t local_n0;
    ptrdiff_t local_n1;
    ptrdiff_t local_0_start;
    ptrdiff_t local_1_start;

    int* scount_proc;
    int* rcount_proc;
    int* soffset_proc;
    int* roffset_proc;
    int* scount_proc_w; // for transpose_v8
    int* rcount_proc_w;
    int* soffset_proc_w;
    int* roffset_proc_w;
    int* scount_proc_f; // for fast transposes
    int* rcount_proc_f;
    int* soffset_proc_f;
    int* roffset_proc_f;

    int* scount_proc_v8;
    int* rcount_proc_v8;
    int* soffset_proc_v8;
    int* roffset_proc_v8;

    int last_recv_count;
    int last_local_n0; //the last non zero local_n0
    int last_local_n1; //the last non zero local_n1

    int procid,nprocs;
    int nprocs_0; // number of effective processes involved in the starting phase
    int nprocs_1; // number of effective processes involved in the transposed phase
    int method;   // Transpose method
    int kway;   //   Transpose_v7
    bool kway_async;   //   Transpose_v7

    ptrdiff_t alloc_local;
    bool PINNED;
    bool is_evenly_distributed;

    std::vector< std::pair<int,double> >* pwhich_f_time;

    MPI_Comm comm;
    MPI_Datatype MPI_T;
    MPI_Datatype *stype;
    MPI_Datatype *rtype;
    MPI_Datatype *stype_v8;
    MPI_Datatype *rtype_v8;
    MPI_Datatype *rtype_v8_;

    T * buffer;
    T * buffer_2;
    T * buffer_d;
    T * buffer_d2;
    //T * data_cpu;
    // Deconstructor
    ~T_Plan_gpu();
};

template <typename T>
void local_transpose_cuda(int r, int c, int n_tuples, T * in, T *out );

template <typename T>
void local_transpose_cuda(int r, int c, int n_tuples, int n_tuples2, T* in, T* out );

template <typename T>
void local_transpose_cuda(int r, int c, int n_tuples, T* A);

template<typename T> void     fast_transpose_cuda_v1(T_Plan_gpu<T>* T_plan, T * data, double *timings, unsigned flags=0, int howmany=1, int tag=0 );
template<typename T> void   fast_transpose_cuda_v1_2(T_Plan_gpu<T>* T_plan, T * data, double *timings, unsigned flags=0, int howmany=1, int tag=0 );
template<typename T> void   fast_transpose_cuda_v1_3(T_Plan_gpu<T>* T_plan, T * data, double *timings, unsigned flags=0, int howmany=1, int tag=0 );
template<typename T> void     fast_transpose_cuda_v2(T_Plan_gpu<T>* T_plan, T * data, double *timings, unsigned flags=0, int howmany=1, int tag=0 );
template<typename T> void     fast_transpose_cuda_v3(T_Plan_gpu<T>* T_plan, T * data, double *timings, int kway, unsigned flags=0, int howmany=1 , int tag=0);
template<typename T> void   fast_transpose_cuda_v3_2(T_Plan_gpu<T>* T_plan, T * data, double *timings, int kway, unsigned flags=0, int howmany=1 , int tag=0);
template<typename T> void   fast_transpose_cuda_v1_h(T_Plan_gpu<T>* T_plan, T * data, double *timings, unsigned flags, int howmany, int tag=0 );
template<typename T> void fast_transpose_cuda_v1_2_h(T_Plan_gpu<T>* T_plan, T * data, double *timings, unsigned flags, int howmany, int tag=0 );
template<typename T> void fast_transpose_cuda_v1_3_h(T_Plan_gpu<T>* T_plan, T * data, double *timings, unsigned flags, int howmany, int tag=0 );
template<typename T> void   fast_transpose_cuda_v2_h(T_Plan_gpu<T>* T_plan, T * data, double *timings, unsigned flags, int howmany, int tag=0);
template<typename T> void   fast_transpose_cuda_v3_h(T_Plan_gpu<T>* T_plan, T * data, double *timings, int kway, unsigned flags, int howmany, int tag=0 );
template<typename T> void          transpose_cuda_v5(T_Plan_gpu<T>* T_plan, T * inout, double *timings, unsigned flags=0,int howmany=1, int tag=0 ); // INPLACE local transpose + mpiIsendIrecv+local transpose
template<typename T> void         transpose_cuda_v5_2(T_Plan_gpu<T>* T_plan, T * inout, double *timings, unsigned flags=0,int howmany=1, int tag=0  );// INPLACE local transpose + multiple memcpy/mpiIsendIrecv+local transpose
template<typename T> void         transpose_cuda_v5_3(T_Plan_gpu<T>* T_plan, T * inout, double *timings, unsigned flags=0,int howmany=1, int tag=0  );// INPLACE local transpose + multiple memcpy/mpiIsendIrecv+local transpose
template<typename T> void          transpose_cuda_v6(T_Plan_gpu<T>* T_plan, T * inout, double *timings, unsigned flags=0,int howmany=1, int tag=0  ); // INPLACE local transpose + mpialltoallv+local transpose
template<typename T> void          transpose_cuda_v7(T_Plan_gpu<T>* T_plan, T * inout, double *timings, int kway , unsigned flags=0,int howmany=1, int tag=0 ); // INPLACE local transpose + paralltoallv+local transpose
template<typename T> void        transpose_cuda_v7_2(T_Plan_gpu<T>* T_plan, T * inout, double *timings, int kway , unsigned flags=0,int howmany=1, int tag=0 ); // INPLACE local transpose + paralltoallv+local transpose


template<typename T> void     fast_transpose_cuda_v(T_Plan_gpu<T>* T_plan, T * data, double *timings, unsigned flags=0, int howmany=1, int tag=0 );  // handles only the cases where Flag=1
template<typename T> void     fast_transpose_cuda_v_h(T_Plan_gpu<T>* T_plan, T * data, double *timings, unsigned flags=0, int howmany=1, int tag=0 );  // handles only the cases where Flag=1
template<typename T> void     fast_transpose_cuda_v_i(T_Plan_gpu<T>* T_plan, T * data, double *timings, unsigned flags=0, int howmany=1, int tag=0 );  // handles only the cases where Flag=1
template<typename T> void     fast_transpose_cuda_v_hi(T_Plan_gpu<T>* T_plan, T * data, double *timings, unsigned flags=0, int howmany=1, int tag=0 );  // handles only the cases where Flag=1
template<typename T> void     fast_transpose_cuda_v1_2i(T_Plan_gpu<T>* T_plan, T * data, double *timings, unsigned flags=0, int howmany=1, int tag=0 );  // handles only the cases where Flag=1
template<typename T> void     fast_transpose_cuda_v1_3i(T_Plan_gpu<T>* T_plan, T * data, double *timings, unsigned flags=0, int howmany=1, int tag=0 );  // handles only the cases where Flag=1

// outplace local transpose multiple n_tuples for the last col
template <typename T>
void local_transpose_col_cuda(int r, int c, int n_tuples, int n_tuples2, T* in, T* out );

template <typename T>
void memcpy_v1_h1(int nprocs_1,int howmany, int local_n0, int n_tuples,ptrdiff_t* local_n1_proc, T* send_recv_d, T* data,int idist,int N1,ptrdiff_t* local_1_start_proc);

template <typename T>
void memcpy_v1_h2( int nprocs_0,int howmany,ptrdiff_t* local_0_start_proc,ptrdiff_t* local_n0_proc, T* data,int odist,int local_n1,int n_tuples,    T* send_recv_cpu);


//#include <../src/transpose_cuda.cu>
#endif
