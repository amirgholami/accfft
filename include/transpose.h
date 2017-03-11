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

#ifndef _TRANSPOSE_
#define _TRANSPOSE_
#include <vector>
template<typename T>
class Mem_Mgr {
public:
	Mem_Mgr(int N0, int N1, int n_tuples, MPI_Comm Comm, int howmany = 1,
			ptrdiff_t specified_alloc_local = 0);

	ptrdiff_t N[2];
	ptrdiff_t n_tuples;
	//int procid,nprocs;
	ptrdiff_t alloc_local;

	ptrdiff_t local_n0;
	ptrdiff_t local_n1;

	bool PINNED;

	T * buffer;
	T * buffer_2;
	T * buffer_3; // used for fast operators
	T * buffer_d;
	//T * data_cpu;
	//MPI_Comm comm;
	// Deconstructor
	~Mem_Mgr();
};

template<typename T>
class T_Plan {
public:
	T_Plan() {
	}
	;
	T_Plan(int N0, int N1, int n_tuples, Mem_Mgr<T> * Mem_mgr, MPI_Comm,
			int howmany = 1);
	void which_method(T*);
	void which_fast_method(T_Plan<T>*, T*, unsigned flags = 2, int howmany = 1,
			int tag = 0);
	void execute(T_Plan<T>*, T*, double*, unsigned flags = 0, int howmany = 1,
			int tag = 0);

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
	int last_local_n1; //the last non zero local_n1
	int last_local_n0; //the last non zero local_n0

	int procid, nprocs; // note that nprocs is for col or row comms
	int nprocs_0; // number of effective processes involved in the starting phase
	int nprocs_1; // number of effective processes involved in the transposed phase
	int method;   // Transpose method
	int kway;
	bool kway_async;

	std::vector<std::pair<int, double> > *pwhich_f_time;

	ptrdiff_t alloc_local;
	bool PINNED;
	bool is_evenly_distributed;

	MPI_Comm comm; // this is either row or column communicator
	MPI_Datatype MPI_T;
	MPI_Datatype *stype;
	MPI_Datatype *rtype;
	MPI_Datatype *stype_v8;
	MPI_Datatype *rtype_v8;
	MPI_Datatype *rtype_v8_;

	T * buffer;
	T * buffer_2;
	T * buffer_d;
	//T * data_cpu;
	// Deconstructor
	~T_Plan();
};

void mytestfunctiondouble();
template<typename T> void transpose_v5(T_Plan<T>* T_plan, T * inout,
		double *timings, unsigned flags = 0, int howmany = 1, int tag = 0); // INPLACE local transpose + mpiIsendIrecv+local transpose
template<typename T> void transpose_v6(T_Plan<T>* T_plan, T * inout,
		double *timings, unsigned flags = 0, int howmany = 1); // INPLACE local transpose + alltoallv+local transpose
// note that tag is not needed here as the plan comm with mpialltoall is used
template<typename T> void transpose_v7(T_Plan<T>* T_plan, T * inout,
		double *timings, int kway, unsigned flags = 0, int howmany = 1); // INPLACE local transpose + paralltoallv+local transpose
// note that tag is not needed here as the plan comm with mpialltoall is used
template<typename T> void transpose_v8(T_Plan<T>* T_plan, T * inout,
		double *timings, unsigned flags = 0, int howmany = 1, int tag = 0); // INPLACE local transpose + mpiIsendIrecv+local transpose
template<typename T> void fast_transpose_v1(T_Plan<T>* T_plan, T * inout,
		double *timings, unsigned flags = 0, int howmany = 1, int tag = 0); // INPLACE local transpose + mpiIsendIrecv+local transpose
template<typename T> void fast_transpose_v2(T_Plan<T>* T_plan, T * inout,
		double *timings, unsigned flags = 0, int howmany = 1, int tag = 0); // INPLACE local transpose + mpiIsendIrecv+local transpose
template<typename T> void fast_transpose_v3(T_Plan<T>* T_plan, T * inout,
		double *timings, int kway, unsigned flags = 0, int howmany = 1,
		int tag = 0); // INPLACE local transpose + mpiIsendIrecv+local transpose
template<typename T> void fast_transpose_v1_h(T_Plan<T>* T_plan, T * inout,
		double *timings, unsigned flags = 0, int howmany = 1, int tag = 0); // INPLACE local transpose + mpiIsendIrecv+local transpose
template<typename T> void fast_transpose_v2_h(T_Plan<T>* T_plan, T * inout,
		double *timings, unsigned flags = 0, int howmany = 1, int tag = 0); // INPLACE local transpose + mpiIsendIrecv+local transpose
template<typename T> void fast_transpose_v3_h(T_Plan<T>* T_plan, T * inout,
		double *timings, int kway, unsigned flags = 0, int howmany = 1,
		int tag = 0); // INPLACE local transpose + mpiIsendIrecv+local transpose

template<typename T> void fast_transpose_v(T_Plan<T>* T_plan, T * inout,
		double *timings, int kway = 2, unsigned flags = 0, int howmany = 1,
		int tag = 0, int method = 1, int comm_test = 0);
template<typename T> void fast_transpose_v_h(T_Plan<T>* T_plan, T * inout,
		double *timings, int kway = 2, unsigned flags = 0, int howmany = 1,
		int tag = 0, int method = 1, int comm_test = 0);
template<typename T> void fast_transpose_v_i(T_Plan<T>* T_plan, T * inout,
		double *timings, int kway = 2, unsigned flags = 0, int howmany = 1,
		int tag = 0, int method = -1); // INPLACE local transpose + mpiIsendIrecv+local transpose
template<typename T> void fast_transpose_v_hi(T_Plan<T>* T_plan, T * inout,
		double *timings, int kway = 2, unsigned flags = 0, int howmany = 1,
		int tag = 0, int method = -1); // INPLACE local transpose + mpiIsendIrecv+local transpose
#endif
