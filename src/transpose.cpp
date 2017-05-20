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
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <cmath>
#include <string.h>
#include <fftw3.h>
#include <vector>
#include <bitset>
#include "transpose.h"
#include "parUtils.h"
#include <assert.h>

#define PCOUT if(procid == 0) std::cout
//#define VERBOSE2
#define VERBOSE 0

static bool IsPowerOfTwo(unsigned long x) {
	return (x & (x - 1)) == 0;
}
static bool IsPowerOfN(unsigned long x, int n) {
	if (x == 0)
		return false;
	while (x % n == 0) {
		x /= n;
	}
	return x == 1;
}

static int intpow(int a, int b) {
	return ((int) std::pow((double) a, b));
}

template<typename T>
Mem_Mgr<T>::Mem_Mgr(int N0, int N1, int tuples, MPI_Comm Comm, int howmany,
		ptrdiff_t specified_alloc_local) {

	N[0] = N0;
	N[1] = N1;
	n_tuples = tuples;
	int procid, nprocs;
	MPI_Comm_rank(Comm, &procid);
	MPI_Comm_size(Comm, &nprocs);

	// Determine local_n0/n1 of each processor
	if (specified_alloc_local == 0) {
		{
			ptrdiff_t * local_n0_proc = (ptrdiff_t*) malloc(
					sizeof(ptrdiff_t) * nprocs);
			ptrdiff_t * local_n1_proc = (ptrdiff_t*) malloc(
					sizeof(ptrdiff_t) * nprocs);
			for (int proc = 0; proc < nprocs; ++proc) {
				local_n0_proc[proc] = ceil(N[0] / (double) nprocs);
				local_n1_proc[proc] = ceil(N[1] / (double) nprocs);

				if ((N[0] - local_n0_proc[proc] * proc) < local_n0_proc[proc]) {
					local_n0_proc[proc] = N[0] - local_n0_proc[proc] * proc;
					local_n0_proc[proc] *= (int) local_n0_proc[proc] > 0;
				}
				if ((N[1] - local_n1_proc[proc] * proc) < local_n1_proc[proc]) {
					local_n1_proc[proc] = N[1] - local_n1_proc[proc] * proc;
					local_n1_proc[proc] *= (int) local_n1_proc[proc] > 0;
				}

			}

			local_n0 = local_n0_proc[procid];
			local_n1 = local_n1_proc[procid];
			free(local_n0_proc);
			free(local_n1_proc);
		}

		// Determine alloc local based on maximum size of input and output distribution
		alloc_local = local_n0 * N[1] * n_tuples * sizeof(T);
		if (alloc_local < local_n1 * N[0] * n_tuples * sizeof(T))
			alloc_local = local_n1 * N[0] * n_tuples * sizeof(T);

		alloc_local *= howmany;
	} else {
		alloc_local = specified_alloc_local;
	}
	if (alloc_local <= 1.05 * intpow(2, 30))
		PINNED = 1;
	else
		PINNED = 0;

	int err;
	err = posix_memalign((void **) &buffer, 64, alloc_local);
	err = posix_memalign((void **) &buffer_2, 64, alloc_local);
	err = posix_memalign((void **) &buffer_3, 64, alloc_local);
	err = posix_memalign((void **) &operator_buffer_1, 64, alloc_local);
	err = posix_memalign((void **) &operator_buffer_2, 64, alloc_local); // exclusive to div
	assert(err == 0 && "posix_memalign failed to allocate memory in Mem_Mgr");
	memset(buffer, 0, alloc_local);
	memset(buffer_2, 0, alloc_local);
	memset(buffer_3, 0, alloc_local);
	memset(operator_buffer_1, 0, alloc_local);
	memset(operator_buffer_2, 0, alloc_local);

}

template<typename T>
Mem_Mgr<T>::~Mem_Mgr() {

	free(buffer);
	free(buffer_2);
	free(buffer_3);
	free(operator_buffer_1);
	free(operator_buffer_2);
}

void mytestfunctiondouble() {
	MPI_Comm comm;
	T_Plan<double> *T = new T_Plan<double>(0, 0, 0, NULL, comm, 0);
	T->which_method(NULL);
}

template<typename T>
T_Plan<T>::T_Plan(int N0, int N1, int tuples, Mem_Mgr<T> * Mem_mgr,
		MPI_Comm Comm, int howmany) {

	N[0] = N0;
	N[1] = N1;
	n_tuples = tuples;
	MPI_Comm_rank(Comm, &procid);
	MPI_Comm_size(Comm, &nprocs);

	local_n0_proc = (ptrdiff_t*) calloc(nprocs, sizeof(ptrdiff_t));
	local_n1_proc = (ptrdiff_t*) calloc(nprocs, sizeof(ptrdiff_t));
	local_0_start_proc = (ptrdiff_t*) calloc(nprocs, sizeof(ptrdiff_t));
	local_1_start_proc = (ptrdiff_t*) calloc(nprocs, sizeof(ptrdiff_t));

	// Determine local_n0/n1 of each processor

	local_0_start_proc[0] = 0;
	local_1_start_proc[0] = 0;
	for (int proc = 0; proc < nprocs; ++proc) {
		local_n0_proc[proc] = ceil(N[0] / (double) nprocs);
		local_n1_proc[proc] = ceil(N[1] / (double) nprocs);

		if ((N[0] - local_n0_proc[proc] * proc) < local_n0_proc[proc]) {
			local_n0_proc[proc] = N[0] - local_n0_proc[proc] * proc;
			local_n0_proc[proc] *= (int) local_n0_proc[proc] > 0;
		}
		if ((N[1] - local_n1_proc[proc] * proc) < local_n1_proc[proc]) {
			local_n1_proc[proc] = N[1] - local_n1_proc[proc] * proc;
			local_n1_proc[proc] *= (int) local_n1_proc[proc] > 0;
		}

		if (proc != 0) {
			local_0_start_proc[proc] = local_0_start_proc[proc - 1]
					+ local_n0_proc[proc - 1];
			local_1_start_proc[proc] = local_1_start_proc[proc - 1]
					+ local_n1_proc[proc - 1];
		}

	}

	local_n0 = local_n0_proc[procid];
	local_n1 = local_n1_proc[procid];
	local_0_start = local_0_start_proc[procid];
	local_1_start = local_1_start_proc[procid];

	alloc_local = Mem_mgr->alloc_local;
	// Determine effective processors taking part in each transpose phase
	nprocs_0 = 0;
	nprocs_1 = 0;
	for (int proc = 0; proc < nprocs; ++proc) {
		if (local_n0_proc[proc] != 0)
			nprocs_0 += 1;
		if (local_n1_proc[proc] != 0)
			nprocs_1 += 1;
	}

	// Set send recv counts for communication part
	scount_proc = (int*) calloc(nprocs, sizeof(int));
	rcount_proc = (int*) calloc(nprocs, sizeof(int));
	soffset_proc = (int*) calloc(nprocs, sizeof(int));
	roffset_proc = (int*) calloc(nprocs, sizeof(int));

	scount_proc_f = (int*) calloc(nprocs, sizeof(int));
	rcount_proc_f = (int*) calloc(nprocs, sizeof(int));
	soffset_proc_f = (int*) calloc(nprocs, sizeof(int));
	roffset_proc_f = (int*) calloc(nprocs, sizeof(int));

	scount_proc_w = (int*) calloc(nprocs, sizeof(int));
	rcount_proc_w = (int*) calloc(nprocs, sizeof(int));
	soffset_proc_w = (int*) calloc(nprocs, sizeof(int));
	roffset_proc_w = (int*) calloc(nprocs, sizeof(int));

	last_recv_count = 0; // Will store the n_tuples of the last received data. In general ~=n_tuples
	if (nprocs_1 > nprocs_0)
		for (int proc = 0; proc < nprocs; ++proc) {

			scount_proc[proc] = local_n1_proc[proc] * local_n0 * n_tuples;

			if (scount_proc[proc] != 0)
				rcount_proc[proc] = local_n1_proc[proc] * local_n0_proc[proc]
						* n_tuples; //scount_proc[proc];
			else
				rcount_proc[proc] = local_n1 * local_n0_proc[proc] * n_tuples; //local_n0_proc[proc]*n_tuples; //

			soffset_proc[proc] = 0;
			roffset_proc[proc] = 0;
			if (proc == 0) {
				soffset_proc[proc] = 0;
				roffset_proc[proc] = 0;
			} else {
				soffset_proc[proc] = soffset_proc[proc - 1]
						+ scount_proc[proc - 1];
				roffset_proc[proc] = roffset_proc[proc - 1]
						+ rcount_proc[proc - 1];
			}

			if (proc >= nprocs_1) { // in case you have requested too many processes
				rcount_proc[proc] = 0;
				scount_proc[proc] = 0;
			}
			if (scount_proc[proc] == 0)
				soffset_proc[proc] = 0; //local_n0*N[1]*n_tuples-1;

			if (proc >= nprocs_0) {
				rcount_proc[proc] = 0;
				roffset_proc[proc] = 0;
			}
			if (rcount_proc[proc] != 0)
				last_recv_count = rcount_proc[proc];
			if (local_n1_proc[proc] != 0)
				last_local_n1 = local_n1_proc[proc];
			if (local_n0_proc[proc] != 0)
				last_local_n0 = local_n0_proc[proc];
		}
	else if (nprocs_1 <= nprocs_0)
		for (int proc = 0; proc < nprocs; ++proc) {

			scount_proc[proc] = local_n1_proc[proc] * local_n0 * n_tuples;
			rcount_proc[proc] = local_n1 * local_n0_proc[proc] * n_tuples; //scount_proc[proc];

			soffset_proc[proc] = 0;
			roffset_proc[proc] = 0;
			if (proc == 0) {
				soffset_proc[proc] = 0;
				roffset_proc[proc] = 0;
			} else {
				soffset_proc[proc] = soffset_proc[proc - 1]
						+ scount_proc[proc - 1];
				roffset_proc[proc] = roffset_proc[proc - 1]
						+ rcount_proc[proc - 1];
			}

			if (proc >= nprocs_0) { // in case you have requested too many processes
				rcount_proc[proc] = 0;
				scount_proc[proc] = 0;
				roffset_proc[proc] = 0;
				soffset_proc[proc] = 0;
			}

			if (scount_proc[proc] == 0)
				soffset_proc[proc] = 0; //local_n0*N[1]*n_tuples-1;

			if (proc >= nprocs_1) {
				scount_proc[proc] = 0;
				soffset_proc[proc] = 0;
			}
			if (rcount_proc[proc] != 0)
				last_recv_count = rcount_proc[proc];

			if (local_n1_proc[proc] != 0)
				last_local_n1 = local_n1_proc[proc];
			if (local_n0_proc[proc] != 0)
				last_local_n0 = local_n0_proc[proc];
		}

	is_evenly_distributed = 0; // use alltoallv
	if ((local_n0 * nprocs_0 - N[0]) == 0 && (local_n1 * nprocs_1 - N[1]) == 0
			&& nprocs_0 == nprocs_1 && nprocs_0 == nprocs) {
		is_evenly_distributed = 1; // use alltoall
	}

	method = 1;
	kway = nprocs;
	kway_async = true;

	this->pwhich_f_time = new std::vector<std::pair<int, double> >;
	MPI_Type_contiguous(sizeof(T), MPI_BYTE, &MPI_T);
	MPI_Type_commit(&MPI_T);

	stype = new MPI_Datatype[nprocs];
	rtype = new MPI_Datatype[nprocs];
	//stype_v8=new MPI_Datatype[nprocs];
	//rtype_v8_=new MPI_Datatype[nprocs];
	//rtype_v8=new MPI_Datatype[nprocs];

	for (int i = 0; i < nprocs; i++) {
		MPI_Type_vector(howmany, scount_proc[i], local_n0 * N[1] * n_tuples,
				MPI_T, &stype[i]);
		MPI_Type_vector(howmany, rcount_proc[i], local_n1 * N[0] * n_tuples,
				MPI_T, &rtype[i]);

		MPI_Type_commit(&stype[i]);
		MPI_Type_commit(&rtype[i]);

		soffset_proc_w[i] = soffset_proc[i] * sizeof(T);
		roffset_proc_w[i] = roffset_proc[i] * sizeof(T);
		scount_proc_w[i] = 1;
		rcount_proc_w[i] = 1;

		soffset_proc_f[i] = soffset_proc[i] * howmany;
		roffset_proc_f[i] = roffset_proc[i] * howmany;
		scount_proc_f[i] = scount_proc[i] * howmany;
		rcount_proc_f[i] = rcount_proc[i] * howmany;

		/*
		 if(local_n0!=0){
		 soffset_proc_v8[i]=soffset_proc[i]/local_n0;
		 scount_proc_v8[i]=scount_proc[i]/local_n0;
		 }
		 if(local_n1!=0){
		 roffset_proc_v8[i]=roffset_proc[i]/local_n1;
		 rcount_proc_v8[i]=rcount_proc[i]/local_n1;
		 }

		 MPI_Type_vector(local_n0,scount_proc_v8[i],N[1]*n_tuples, MPI_T, &stype_v8[i]);
		 MPI_Type_vector(local_n1,1*n_tuples,N[0]*n_tuples, MPI_T, &rtype_v8_[i]);
		 MPI_Type_hvector(rcount_proc_v8[i],1,n_tuples*sizeof(double),rtype_v8_[i], &rtype_v8[i]);

		 MPI_Type_commit(&stype_v8[i]);
		 MPI_Type_commit(&rtype_v8[i]);
		 */

	}

	comm = Comm; // MPI Communicator
	buffer = Mem_mgr->buffer;
	buffer_2 = Mem_mgr->buffer_2;
	buffer_d = Mem_mgr->buffer_d;
	//data_cpu=Mem_mgr->data_cpu;

}

template<typename T>
void T_Plan<T>::which_method(T* data) {

	T_Plan<T>* T_plan = this;
	double dummy[4] = { 0 };
	size_t temp = 2 * (int) log2(nprocs) + 3;
	double * time = (double*) malloc(sizeof(double) * temp);
	double * g_time = (double*) malloc(sizeof(double) * temp);
	for (int i = 0; i < temp; i++)
		time[i] = 1000;

	transpose_v5(T_plan, (T*) data, dummy);  // Warmup
	time[0] = -MPI_Wtime();
	transpose_v5(T_plan, (T*) data, dummy);
	time[0] += MPI_Wtime();

	transpose_v6(T_plan, (T*) data, dummy);  // Warmup
	time[1] = -MPI_Wtime();
	transpose_v6(T_plan, (T*) data, dummy);
	time[1] += MPI_Wtime();

	if (IsPowerOfTwo(nprocs) && nprocs > 511) {
		kway_async = true;
#ifndef TORUS_TOPOL
		for (int i = 0; i < 6; i++) {
			kway = nprocs / intpow(2, i);
			MPI_Barrier(T_plan->comm);
			transpose_v7(T_plan, (T*) data, dummy, kway);  // Warmup
			time[2 + i] = -MPI_Wtime();
			transpose_v7(T_plan, (T*) data, dummy, kway);
			time[2 + i] += MPI_Wtime();
		}
#endif

#ifdef TORUS_TOPOL
		kway_async=false;
		for (int i=0;i<6;i++) {
			kway=nprocs/intpow(2,i);
			MPI_Barrier(T_plan->comm);
			transpose_v7(T_plan,(T*)data,dummy,kway);  // Warmup
			time[2+(int)log2(nprocs)+i]=-MPI_Wtime();
			transpose_v7(T_plan,(T*)data,dummy,kway);
			time[2+(int)log2(nprocs)+i]+=MPI_Wtime();
		}
#endif
	}

	//transpose_v8(T_plan,(T*)data,dummy);  // Warmup
	//time[2*(int)log2(nprocs)+2]=-MPI_Wtime();
	//transpose_v8(T_plan,(T*)data,dummy);
	//time[2*(int)log2(nprocs)+2]+=MPI_Wtime();

	MPI_Allreduce(time, g_time, (2 * (int) log2(nprocs) + 3), MPI_DOUBLE,
			MPI_MAX, T_plan->comm);
	if (VERBOSE >= 1)
		if (T_plan->procid == 0) {
			for (int i = 0; i < 2 * (int) log2(nprocs) + 3; ++i)
				std::cout << " time[" << i << "]= " << g_time[i] << " , ";
			std::cout << '\n';
		}

	double smallest = 1000;
	for (int i = 0; i < 2 * (int) log2(nprocs) + 3; i++)
		smallest = std::min(smallest, g_time[i]);

	if (g_time[0] == smallest) {
		T_plan->method = 5;
	} else if (g_time[1] == smallest) {
		T_plan->method = 6;
	} else if (g_time[2 * (int) log2(nprocs) + 2] == smallest) {
		T_plan->method = 8;
	} else {
		for (int i = 0; i < (int) log2(nprocs); i++)
			if (g_time[2 + i] == smallest) {
				T_plan->method = 7;
				T_plan->kway = nprocs / intpow(2, i);
				T_plan->kway_async = true;
				break;
			}
		for (int i = 0; i < (int) log2(nprocs); i++)
			if (g_time[2 + (int) log2(nprocs) + i] == smallest) {
				T_plan->method = 7;
				T_plan->kway = nprocs / intpow(2, i);
				T_plan->kway_async = false;
				break;
			}
	}

	if (VERBOSE >= 1) {
		PCOUT << "smallest= " << smallest << std::endl;
		PCOUT << "Using transpose v" << method << " kway= " << T_plan->kway
				<< " kway_async=" << T_plan->kway_async << std::endl;
	}
	free(time);
	free(g_time);
	MPI_Barrier(T_plan->comm);

	return;
}

#include <vector>
struct accfft_sort_pred {
	bool operator()(const std::pair<int, double> &left,
			const std::pair<int, double> &right) {
		return left.second < right.second;
	}
};
template<typename T>
void T_Plan<T>::which_fast_method(T_Plan* T_plan, T* data, unsigned flags,
		int howmany, int tag) {

	double dummy[5] = { 0 };
	double timings[5] = { 0 };

	double tmp;
	int factor;
	if (IsPowerOfTwo(nprocs))
		factor = 2;
	else if (IsPowerOfN(nprocs, 3))
		factor = 0; // support will be added in near future
	else if (IsPowerOfN(nprocs, 5))
		factor = 0; // support will be added in near future
	else
		factor = 0;

	if (factor > 0 && nprocs > 7) {
		kway_async = true;
		kway = nprocs / factor;
		do {
			MPI_Barrier(T_plan->comm);
			fast_transpose_v_h(T_plan, (T*) data, timings, kway, flags, howmany,
					tag, 3, 1);
			tmp = -MPI_Wtime();
			fast_transpose_v_h(T_plan, (T*) data, timings, kway, flags, howmany,
					tag, 3, 1);
			tmp += MPI_Wtime();
			pwhich_f_time->push_back(std::make_pair(kway, tmp));
			kway = kway / factor;
		} while (kway > 4);
		MPI_Barrier(comm);

		kway_async = false;
		kway = nprocs / factor;
		do {
			fast_transpose_v_h(T_plan, (T*) data, timings, kway, flags, howmany,
					tag, 3, 1);
			MPI_Barrier(T_plan->comm);
			tmp = -MPI_Wtime();
			fast_transpose_v_h(T_plan, (T*) data, timings, kway, flags, howmany,
					tag, 3, 1);
			tmp += MPI_Wtime();
			pwhich_f_time->push_back(std::make_pair(-kway, tmp));
			kway = kway / factor;
		} while (kway > 4);
	}

	fast_transpose_v_h(T_plan, (T*) data, timings, kway, flags, howmany, tag, 1,
			1);
	MPI_Barrier(comm);
	tmp = -MPI_Wtime();
	fast_transpose_v_h(T_plan, (T*) data, timings, kway, flags, howmany, tag, 1,
			1);
	tmp += MPI_Wtime();
	pwhich_f_time->push_back(std::make_pair(nprocs, tmp));

	fast_transpose_v_h(T_plan, (T*) data, timings, kway, flags, howmany, tag, 2,
			1);
	MPI_Barrier(comm);
	tmp = -MPI_Wtime();
	fast_transpose_v_h(T_plan, (T*) data, timings, kway, flags, howmany, tag, 2,
			1);
	tmp += MPI_Wtime();
	pwhich_f_time->push_back(std::make_pair(2, tmp));

	//if (comm!=T_plan->comm)

	//transpose_v8(T_plan,(T*)data,dummy);  // Warmup
	//time[2*(int)log2(nprocs)+2]=-MPI_Wtime();
	//transpose_v8(T_plan,(T*)data,dummy);
	//time[2*(int)log2(nprocs)+2]+=MPI_Wtime();

	for (std::vector<std::pair<int, double> >::iterator it =
			pwhich_f_time->begin(); it != pwhich_f_time->end(); ++it) {
		MPI_Allreduce(&it->second, &tmp, 1, MPI_DOUBLE, MPI_MAX, comm);
		it->second = tmp;
	}

	std::sort(pwhich_f_time->begin(), pwhich_f_time->end(), accfft_sort_pred());
	double min_time = pwhich_f_time->front().second;
	if (pwhich_f_time->front().first == nprocs) {
		T_plan->method = 1;
		T_plan->kway = nprocs;
		T_plan->kway_async = 1;
	} else if (pwhich_f_time->front().first == 2) {
		T_plan->method = 2;
		T_plan->kway = 2;
		T_plan->kway_async = 0;
	} else {
		T_plan->method = 3;
		T_plan->kway = std::abs(pwhich_f_time->front().first);
		T_plan->kway_async = (pwhich_f_time->front().first > 0);
	}

	if (VERBOSE >= 1) {
		std::sort(pwhich_f_time->begin(), pwhich_f_time->end());
		for (std::vector<std::pair<int, double> >::iterator it =
				pwhich_f_time->begin(); it != pwhich_f_time->end(); ++it) {
			PCOUT << it->first << '\t' << it->second << std::endl;
		}
		PCOUT << "Min time= " << min_time << std::endl;
		PCOUT << "Using transpose v" << method << " kway= " << T_plan->kway
				<< " kway_async=" << T_plan->kway_async << std::endl;
	}
	MPI_Barrier(T_plan->comm);

	return;
}
template<typename T>
void T_Plan<T>::execute(T_Plan* T_plan, T* data, double *timings,
		unsigned flags, int howmany, int tag, T* data_out) {

	if (howmany == 1) {
		if (method == 1 || method == 2 || method == 3)
			fast_transpose_v(T_plan, (T*) data, timings, kway, flags, howmany,
					tag, method, 0, (T*) data_out);
		//if(method==1)
		//  fast_transpose_v1(T_plan,(T*)data,timings,flags,howmany,tag, (T*) data_out);
		//if(method==2)
		//  fast_transpose_v2(T_plan,(T*)data,timings,flags,howmany,tag, (T*) data_out);
		//if(method==3)
		//  fast_transpose_v3(T_plan,(T*)data,timings,kway,flags,howmany,tag, (T*) data_out);
		if (method == -1 || method == -2 || method == -3)
			fast_transpose_vi(T_plan, (T*) data, timings, kway, flags, howmany,
					tag, method, (T*) data_out);
	} else {
		if (method == 1 || method == 2 || method == 3)    //snafu
			fast_transpose_v_h(T_plan, (T*) data, timings, kway, flags, howmany,
					tag, method, 0, (T*) data_out);
		//if(method==1)
		//  fast_transpose_v1_h(T_plan,(T*)data,timings,flags,howmany,tag, (T*) data_out);
		//if(method==2)
		//  fast_transpose_v2_h(T_plan,(T*)data,timings,flags,howmany,tag, (T*) data_out);
		//if(method==3)
		//  fast_transpose_v3_h(T_plan,(T*)data,timings,kway,flags,howmany,tag, (T*) data_out);
		if (method == -1 || method == -2 || method == -3)
			fast_transpose_v_hi(T_plan, (T*) data, timings, kway, flags,
					howmany, tag, method, (T*) data_out);
	}
	if (method == 5)
		transpose_v5(T_plan, (T*) data, timings, flags, howmany, tag, (T*) data_out);
	if (method == 6)
		transpose_v6(T_plan, (T*) data, timings, flags, howmany, (T*) data_out);
	if (method == 7)
		transpose_v7(T_plan, (T*) data, timings, kway, flags, howmany, (T*) data_out);
	if (method == 8)
		transpose_v8(T_plan, (T*) data, timings, flags, howmany, tag, (T*) data_out);

	return;
}

template<typename T>
T_Plan<T>::~T_Plan() {
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
	for (int i = 0; i < nprocs; i++) {
		MPI_Type_free(&stype[i]);
		MPI_Type_free(&rtype[i]);
	}

	//std::vector< std::pair<int,double> >temp().swap(*pwhich_f_time);

	delete[] stype;
	delete[] rtype;
	delete (pwhich_f_time);
	MPI_Type_free(&MPI_T);
	//delete [] stype_v8;
	//delete [] rtype_v8;
	//delete [] rtype_v8_;
}

// outplace local transpose
template<typename T>
void local_transpose(int r, int c, int n_tuples, T * __restrict in, T * __restrict out) {

	if (r == 0 || c == 0)
		return;
	int ptr = 0;
	for (int j = 0; j < c; j++) {
		for (int i = 0; i < r; i++) {
			memcpy(&out[ptr], &in[(i * c + j) * n_tuples],
					sizeof(T) * n_tuples);
			ptr += n_tuples;
		}
	}
	return;

}
// outplace local transpose multiple n_tuples for the last row
template<typename T>
void local_transpose(int r, int c, int n_tuples, int n_tuples2, T * __restrict in,
		T * __restrict out) {

	if (r == 0 || c == 0)
		return;
	int ptr = 0;
	for (int j = 0; j < c; j++) {
		for (int i = 0; i < r; i++) {

			if (i == r - 1) {
				memcpy(&out[ptr], &in[i * c * n_tuples + j * n_tuples2],
						sizeof(T) * n_tuples2);
				ptr += n_tuples2;
			} else {
				memcpy(&out[ptr], &in[(i * c + j) * n_tuples],
						sizeof(T) * n_tuples);
				ptr += n_tuples;
			}
		}
	}
	return;

}
// outplace local transpose multiple n_tuples for the last col
template<typename T>
void local_transpose_col(int r, int c, int n_tuples, int n_tuples2, T * __restrict in,
		T * __restrict out) {

	if (r == 0 || c == 0)
		return;
	int ptr = 0;
	for (int j = 0; j < c; j++) {
		for (int i = 0; i < r; i++) {

			if (j == c - 1) {
				memcpy(&out[ptr],
						&in[i * ((c - 1) * n_tuples + n_tuples2)
								+ (j) * n_tuples], sizeof(T) * n_tuples2);
				ptr += n_tuples2;
			} else {
				memcpy(&out[ptr],
						&in[i * ((c - 1) * n_tuples + n_tuples2) + j * n_tuples],
						sizeof(T) * n_tuples);
				ptr += n_tuples;
			}
		}
	}
	return;

}

// in place local transpose
template<typename T>
void local_transpose(int r, int c, int n_tuples, T* __restrict A) {
	{  // new CPU code
		size_t i_, j_;
		T* buff = (T*) malloc(n_tuples * sizeof(T));
		for (size_t i = 0; i < r; i++)
			for (size_t j = 0; j < c; j++) {
				size_t src = j + c * i;
				size_t trg = i + r * j;
				if (src == trg)
					continue; // nothing to be done

				size_t cycle_len = 0;
				while (src < trg) { // find cycle
					i_ = trg / c;
					j_ = trg % c;
					trg = i_ + r * j_;
					cycle_len++;
				}
				if (src != trg)
					continue;

				memcpy(buff, A + trg * n_tuples, n_tuples * sizeof(T));
				for (size_t k = 0; k < cycle_len; k++) { // reverse cycle
					j_ = trg / r;
					i_ = trg % r;
					src = j_ + c * i_;
					memcpy(A + trg * n_tuples, A + src * n_tuples,
							n_tuples * sizeof(T));
					trg = src;
				}
				memcpy(A + trg * n_tuples, buff, n_tuples * sizeof(T));
			}
		free(buff);
	}
}

template<typename T>
void fast_transpose_v_hi(T_Plan<T>* __restrict T_plan, T * __restrict data, double* __restrict timings, int kway,
		unsigned flags, int howmany, int tag, int method, T* __restrict data_out) {

	std::bitset < 8 > Flags(flags); // 1 Transposed in, 2 Transposed out
	if (Flags[1] == 1 && Flags[0] == 0 && T_plan->nprocs == 1) { // If Flags==Transposed_Out return
		return;
	}
  if(Flags[1] == 0 && data_out !=NULL) {
    std::cout << "operation not supported\n";
    return;
  }
	timings[0] -= MPI_Wtime();
	int nprocs, procid;
	int nprocs_0, nprocs_1;
	nprocs = T_plan->nprocs;
	procid = T_plan->procid;
	nprocs_0 = T_plan->nprocs_0;
	nprocs_1 = T_plan->nprocs_1;
	ptrdiff_t *N = T_plan->N;
	T * send_recv = T_plan->buffer;
	T * buffer_2 = T_plan->buffer_2;
	ptrdiff_t local_n0 = T_plan->local_n0;
	ptrdiff_t local_n1 = T_plan->local_n1;
	ptrdiff_t n_tuples = T_plan->n_tuples;

	int idist = N[1] * local_n0 * n_tuples;
	int odist = N[0] * local_n1 * n_tuples;

	double comm_time = 0, shuffle_time = 0, reshuffle_time = 0, total_time = 0;

#ifdef VERBOSE2
	if(VERBOSE>=2) PCOUT<<"INPUT:"<<std::endl;
	if(VERBOSE>=2)
	for(int h=0;h<howmany;h++)
	for(int id=0;id<nprocs;++id) {
		if(procid==id)
		for(int i=0;i<N[1];i++) {
			std::cout<<std::endl;
			for(int j=0;j<local_n0;j++) {
				std::cout<<'\t'<<data[h*idist+(i*local_n0+j)*n_tuples];
			}
		}
		std::cout<<'\n';
	}

#endif
	//PCOUT<<" ============================================= "<<std::endl;
	//PCOUT<<" ==============   Local Transpose============= "<<std::endl;
	//PCOUT<<" ============================================= "<<std::endl;
	int ptr = 0;
	shuffle_time -= MPI_Wtime();

	if (nprocs == 1 && Flags[0] == 1 && Flags[1] == 1) {
		local_transpose(local_n1, N[0], n_tuples, data);
	}
	if (nprocs == 1 && Flags[0] == 0 && Flags[1] == 0) {
		local_transpose(N[0], N[1], n_tuples, data);
	}
	if (nprocs == 1) { // Transpose is done!
		shuffle_time += MPI_Wtime();
		timings[0] += MPI_Wtime();
		timings[1] += shuffle_time;
		timings[2] += 0;
		timings[3] += 0;
		return;
	}

	// The idea is this: If The Output is to be transposed, then only one buffer is needed. The algorithm would be:
	// data --T-> buffer_2    ...  buffer_2 --alltoall--> send_recv ... send_recv -T-> data
	ptr = 0;
	for (int proc = 0; proc < nprocs_1; ++proc)
		for (int h = 0; h < howmany; ++h) {
			memcpy(&buffer_2[ptr],
					&data[h * idist + T_plan->soffset_proc[proc]],
					sizeof(T) * T_plan->scount_proc[proc]); // scount_proc[proc]=local_n1_proc[proc]*local_n0*n_tuples;
			ptr += T_plan->scount_proc[proc]; // pointer is going contiguous along buffer_2
		}
	shuffle_time += MPI_Wtime();

#ifdef VERBOSE2
	ptr=0;
	if(VERBOSE>=2) PCOUT<<"Local Transpose:"<<std::endl;
	if(VERBOSE>=2)
	for(int id=0;id<nprocs;++id) {
		if(procid==id)
		for(int i=0;i<N[1];i++) {
			for(int h=0;h<howmany;h++) {
				std::cout<<std::endl;
				for(int j=0;j<local_n0;j++) {
					std::cout<<'\t'<<buffer_2[ptr]; //data[h*idist+(i*local_n0+j)*n_tuples];
					ptr+=n_tuples;
				}
			}
		}
		std::cout<<'\n';
	}
#endif

	//PCOUT<<" ============================================= "<<std::endl;
	//PCOUT<<" ==============   MPIALLTOALL  =============== "<<std::endl;
	//PCOUT<<" ============================================= "<<std::endl;

	int* scount_proc = T_plan->scount_proc;
	int* rcount_proc = T_plan->rcount_proc;
	int* soffset_proc = T_plan->soffset_proc;
	int* roffset_proc = T_plan->roffset_proc;

	//PCOUT<<"nprocs_0= "<<nprocs_0<<" nprocs_1= "<<nprocs_1<<std::endl;
	comm_time -= MPI_Wtime();

	int soffset = 0, roffset = 0;
	MPI_Status ierr;
	int counter = 1;

	T *s_buf = buffer_2, *r_buf = send_recv;
	if (method == -1) {
		MPI_Request * request = new MPI_Request[2 * nprocs];
		int dst_r, dst_s;
		request[2 * procid] = MPI_REQUEST_NULL;
		request[2 * procid + 1] = MPI_REQUEST_NULL;
		// SEND
		for (int i = 0; i < nprocs; ++i) {
			dst_r = (procid + i) % nprocs;
			dst_s = (procid - i + nprocs) % nprocs;
			if (dst_r != procid && dst_s != procid) {
				soffset = soffset_proc[dst_s] * howmany;
				roffset = roffset_proc[dst_r] * howmany;
				MPI_Isend(&s_buf[soffset], scount_proc[dst_s] * howmany,
						T_plan->MPI_T, dst_s, tag, T_plan->comm,
						&request[2 * dst_s]);
				MPI_Irecv(&r_buf[roffset], rcount_proc[dst_r] * howmany,
						T_plan->MPI_T, dst_r, tag, T_plan->comm,
						&request[2 * dst_r + 1]);
			}
		}
		// Copy Your own part. See the note below for the if condition
		soffset = soffset_proc[procid] * howmany; //aoffset_proc[proc];//proc*count_proc[proc];
		roffset = roffset_proc[procid] * howmany;
		memcpy(&r_buf[roffset], &s_buf[soffset],
				sizeof(T) * scount_proc[procid] * howmany);

		MPI_Waitall(2 * nprocs, request, MPI_STATUSES_IGNORE);
		delete[] request;

	} else if (method == -2) {
		if (T_plan->is_evenly_distributed == 0)
			MPI_Alltoallv(s_buf, T_plan->scount_proc_f, T_plan->soffset_proc_f,
					T_plan->MPI_T, r_buf, T_plan->rcount_proc_f,
					T_plan->roffset_proc_f, T_plan->MPI_T, T_plan->comm);
		else
			MPI_Alltoall(s_buf, T_plan->scount_proc_f[0], T_plan->MPI_T, r_buf,
					T_plan->rcount_proc_f[0], T_plan->MPI_T, T_plan->comm);
	} else if (method == -3) {
		if (T_plan->kway_async)
			par::Mpi_Alltoallv_dense<T, true>(s_buf, T_plan->scount_proc_f,
					T_plan->soffset_proc_f, r_buf, T_plan->rcount_proc_f,
					T_plan->roffset_proc_f, T_plan->comm, kway);
		else
			par::Mpi_Alltoallv_dense<T, false>(s_buf, T_plan->scount_proc_f,
					T_plan->soffset_proc_f, r_buf, T_plan->rcount_proc_f,
					T_plan->roffset_proc_f, T_plan->comm, kway);

	}

	comm_time += MPI_Wtime();

#ifdef VERBOSE2
	ptr=0;
	if(VERBOSE>=2) PCOUT<<"MPIAlltoAll:"<<std::endl;
	if(VERBOSE>=2)
	for(int id=0;id<nprocs;++id) {
		if(procid==id)
		for(int i=0;i<local_n1;i++) {
			for(int h=0;h<howmany;h++) {
				std::cout<<std::endl;
				for(int j=0;j<N[0];j++) {
					std::cout<<'\t'<<send_recv[ptr]<<","<<send_recv[ptr+1];
					ptr+=n_tuples;
				}
			}
		}
		std::cout<<'\n';
	}
#endif

	//PCOUT<<" ============================================= "<<std::endl;
	//PCOUT<<" ============== 2nd Local Trnaspose ========== "<<std::endl;
	//PCOUT<<" ============================================= "<<std::endl;
	reshuffle_time -= MPI_Wtime();
	ptr = 0;
	for (int h = 0; h < howmany; ++h) {
		for (int proc = 0; proc < nprocs_0; ++proc) {
			memcpy(&buffer_2[ptr],
					&send_recv[h * T_plan->rcount_proc[proc]
							+ T_plan->roffset_proc[proc] * howmany],
					sizeof(T) * T_plan->rcount_proc[proc]); // scount_proc[proc]=local_n1_proc[proc]*local_n0*n_tuples;
			ptr += T_plan->rcount_proc[proc]; // pointer is going contiguous along buffer_2
		}
		local_transpose(nprocs_0, local_n1, n_tuples * T_plan->local_n0_proc[0],
				n_tuples * T_plan->last_local_n0, &buffer_2[h * odist],
				&data[h * odist]);
	}

#ifdef VERBOSE2
	if(VERBOSE>=2) PCOUT<<"2nd Transpose"<<std::endl;
	if(VERBOSE>=2)
	for(int id=0;id<nprocs_1;++id) {
		if(procid==id)
		for(int h=0;h<howmany;h++)
		for(int i=0;i<N[0];i++) {
			std::cout<<std::endl;
			for(int j=0;j<local_n1;j++) {
				ptr=h*odist+(i*local_n1+j)*n_tuples;
				std::cout<<'\t'<<data[ptr]<<","<<data[ptr+1];
			}
		}
		std::cout<<'\n';
	}
#endif

	reshuffle_time += MPI_Wtime();

#ifdef VERBOSE1
	if (VERBOSE >= 1) {
		PCOUT << "Shuffle Time= " << shuffle_time << std::endl;
		PCOUT << "Alltoall Time= " << comm_time << std::endl;
		PCOUT << "Reshuffle Time= " << reshuffle_time << std::endl;
		PCOUT << "Total Time= " << (shuffle_time + comm_time + reshuffle_time)
				<< std::endl;
	}
#endif
	timings[0] += MPI_Wtime(); //timings[0]+=shuffle_time+comm_time+reshuffle_time;
	timings[1] += shuffle_time;
	timings[2] += comm_time;
	timings[3] += reshuffle_time;
	return;
} // end fast_transpose_v1

template<typename T>
void fast_transpose_vi(T_Plan<T>* __restrict T_plan, T * __restrict data, double * __restrict timings, int kway,
		unsigned flags, int howmany, int tag, int method, T* __restrict data_out) {

	if (howmany > 1) {
		return fast_transpose_v_hi(T_plan, data, timings, kway, flags, howmany,
				tag, method);
	}
	std::bitset < 8 > Flags(flags); // 1 Transposed in, 2 Transposed out
	if (Flags[1] == 1 && Flags[0] == 0 && T_plan->nprocs == 1) { // If Flags==Transposed_Out return
		return;
	}
  if(Flags[1] == 0 && data_out !=NULL) {
    std::cout << "operation not supported\n";
    return;
  }
	timings[0] -= MPI_Wtime();
	int nprocs, procid;
	int nprocs_0, nprocs_1;
	nprocs = T_plan->nprocs;
	procid = T_plan->procid;
	nprocs_0 = T_plan->nprocs_0;
	nprocs_1 = T_plan->nprocs_1;
	ptrdiff_t *N = T_plan->N;
	T * send_recv = T_plan->buffer;
	T * buffer_2 = T_plan->buffer_2;
	ptrdiff_t local_n0 = T_plan->local_n0;
	ptrdiff_t local_n1 = T_plan->local_n1;
	ptrdiff_t n_tuples = T_plan->n_tuples;

	int idist = N[1] * local_n0 * n_tuples;
	int odist = N[0] * local_n1 * n_tuples;

	double comm_time = 0, shuffle_time = 0, reshuffle_time = 0, total_time = 0;

#ifdef VERBOSE2
	if(VERBOSE>=2) PCOUT<<"INPUT:"<<std::endl;
	if(VERBOSE>=2)
	for(int h=0;h<howmany;h++)
	for(int id=0;id<nprocs;++id) {
		if(procid==id)
		for(int i=0;i<N[1];i++) {
			std::cout<<std::endl;
			for(int j=0;j<local_n0;j++) {
				std::cout<<'\t'<<data[h*idist+(i*local_n0+j)*n_tuples];
			}
		}
		std::cout<<'\n';
	}
#endif

	//PCOUT<<" ============================================= "<<std::endl;
	//PCOUT<<" ==============   Local Transpose============= "<<std::endl;
	//PCOUT<<" ============================================= "<<std::endl;
	int ptr = 0;
	shuffle_time -= MPI_Wtime();

	if (nprocs == 1 && Flags[0] == 1 && Flags[1] == 1) {
		local_transpose(local_n1, N[0], n_tuples, data);
	}
	if (nprocs == 1 && Flags[0] == 0 && Flags[1] == 0) {
		local_transpose(N[0], N[1], n_tuples, data);
	}
	if (nprocs == 1) { // Transpose is done!
		shuffle_time += MPI_Wtime();
		timings[0] += MPI_Wtime();
		timings[1] += shuffle_time;
		timings[2] += 0;
		timings[3] += 0;
		return;
	}

	shuffle_time += MPI_Wtime();
#ifdef VERBOSE2
	ptr=0;
	if(VERBOSE>=2) PCOUT<<"Local Transpose:"<<std::endl;
	if(VERBOSE>=2)
	for(int h=0;h<howmany;h++)
	for(int id=0;id<nprocs;++id) {
		if(procid==id)
		for(int i=0;i<N[1];i++) {
			std::cout<<std::endl;
			for(int j=0;j<local_n0;j++) {
				std::cout<<'\t'<<T_plan->buffer_2[ptr]; //data[h*idist+(i*local_n0+j)*n_tuples];
				ptr+=n_tuples;
			}
		}
		std::cout<<'\n';
	}
#endif

	//PCOUT<<" ============================================= "<<std::endl;
	//PCOUT<<" ==============   MPIALLTOALL  =============== "<<std::endl;
	//PCOUT<<" ============================================= "<<std::endl;

	int* scount_proc = T_plan->scount_proc;
	int* rcount_proc = T_plan->rcount_proc;
	int* soffset_proc = T_plan->soffset_proc;
	int* roffset_proc = T_plan->roffset_proc;

	comm_time -= MPI_Wtime();

	int soffset = 0, roffset = 0;
	MPI_Status ierr;
	int counter = 1;

	T *s_buf = data, *r_buf = send_recv;
	// SEND
	if (method == -1) {
		MPI_Request * request = new MPI_Request[2 * nprocs];
		request[2 * procid] = MPI_REQUEST_NULL;
		request[2 * procid + 1] = MPI_REQUEST_NULL;
		int dst_r, dst_s;
		for (int i = 0; i < nprocs; ++i) {
			dst_r = (procid + i) % nprocs;
			dst_s = (procid - i + nprocs) % nprocs;
			if (dst_r != procid || dst_s != nprocs) {
				roffset = roffset_proc[dst_r];
				soffset = soffset_proc[dst_s];
				MPI_Isend(&s_buf[soffset], scount_proc[dst_s], T_plan->MPI_T,
						dst_s, tag, T_plan->comm, &request[2 * dst_s]);
				MPI_Irecv(&r_buf[roffset], rcount_proc[dst_r], T_plan->MPI_T,
						dst_r, tag, T_plan->comm, &request[2 * dst_r + 1]);
			}
		}
		// Copy Your own part. See the note below for the if condition
		soffset = soffset_proc[procid]; //aoffset_proc[proc];//proc*count_proc[proc];
		roffset = roffset_proc[procid];
		memcpy(&r_buf[roffset], &s_buf[soffset],
				sizeof(T) * scount_proc[procid]);
		MPI_Waitall(2 * nprocs, request, MPI_STATUSES_IGNORE);
		delete[] request;
	} else if (method == -2) {
		if (T_plan->is_evenly_distributed == 0)
			MPI_Alltoallv(s_buf, T_plan->scount_proc_f, T_plan->soffset_proc_f,
					T_plan->MPI_T, r_buf, T_plan->rcount_proc_f,
					T_plan->roffset_proc_f, T_plan->MPI_T, T_plan->comm);
		else
			MPI_Alltoall(s_buf, T_plan->scount_proc_f[0], T_plan->MPI_T, r_buf,
					T_plan->rcount_proc_f[0], T_plan->MPI_T, T_plan->comm);

	} else if (method == -3) {
		if (T_plan->kway_async)
			par::Mpi_Alltoallv_dense<T, true>(s_buf, T_plan->scount_proc_f,
					T_plan->soffset_proc_f, r_buf, T_plan->rcount_proc_f,
					T_plan->roffset_proc_f, T_plan->comm, kway);
		else
			par::Mpi_Alltoallv_dense<T, false>(s_buf, T_plan->scount_proc_f,
					T_plan->soffset_proc_f, r_buf, T_plan->rcount_proc_f,
					T_plan->roffset_proc_f, T_plan->comm, kway);

	}

	comm_time += MPI_Wtime();

#ifdef VERBOSE2
	ptr=0;
	if(VERBOSE>=2) PCOUT<<"MPIAlltoAll:"<<std::endl;
	if(VERBOSE>=2)
	for(int h=0;h<howmany;h++)
	for(int id=0;id<nprocs;++id) {
		if(procid==id)
		for(int i=0;i<local_n1;i++) {
			std::cout<<std::endl;
			for(int j=0;j<N[0];j++) {
				std::cout<<'\t'<<send_recv[ptr]; //send_recv[h*odist+(i*N[0]+j)*n_tuples];//<<","<<send_recv[(i*N[0]+j)*n_tuples+1];
				ptr+=n_tuples;
			}
		}
		std::cout<<'\n';
	}
#endif

	//PCOUT<<" ============================================= "<<std::endl;
	//PCOUT<<" ============== 2nd Local Trnaspose ========== "<<std::endl;
	//PCOUT<<" ============================================= "<<std::endl;
	reshuffle_time -= MPI_Wtime();
	ptr = 0;
	//local_transpose(nprocs_1,local_n0,n_tuples*T_plan->local_n0_proc[0], n_tuples*T_plan->last_local_n0,send_recv,data );
	local_transpose(nprocs_0, local_n1, n_tuples * T_plan->local_n0_proc[0],
			n_tuples * T_plan->last_local_n0, send_recv, data);

#ifdef VERBOSE2
	if(VERBOSE>=2) PCOUT<<"2nd Transpose"<<std::endl;
	if(VERBOSE>=2)
	for(int h=0;h<howmany;h++)
	for(int id=0;id<nprocs_1;++id) {
		if(procid==id)
		for(int i=0;i<local_n1;i++) {
			std::cout<<std::endl;
			for(int j=0;j<N[0];j++) {
				std::cout<<'\t'<<data[h*odist+(i*N[0]+j)*n_tuples];
			}
		}
		std::cout<<'\n';
	}
#endif

	reshuffle_time += MPI_Wtime();

#ifdef VERBOSE1
	if (VERBOSE >= 1) {
		PCOUT << "Shuffle Time= " << shuffle_time << std::endl;
		PCOUT << "Alltoall Time= " << comm_time << std::endl;
		PCOUT << "Reshuffle Time= " << reshuffle_time << std::endl;
		PCOUT << "Total Time= " << (shuffle_time + comm_time + reshuffle_time)
				<< std::endl;
	}
#endif
	timings[0] += MPI_Wtime(); //timings[0]+=shuffle_time+comm_time+reshuffle_time;
	timings[1] += shuffle_time;
	timings[2] += comm_time;
	timings[3] += reshuffle_time;
	return;
} // end fast_transpose_v1

template<typename T>
void fast_transpose_v(T_Plan<T>* __restrict T_plan, T * __restrict data, double * __restrict timings, int kway,
		unsigned flags, int howmany, int tag, int method, int comm_test, T* __restrict data_out) {

	if (howmany > 1) {
		return fast_transpose_v1_h(T_plan, data, timings, flags, howmany, tag);
	}
	std::bitset < 8 > Flags(flags); // 1 Transposed in, 2 Transposed out
	if (Flags[1] == 1 && Flags[0] == 0 && T_plan->nprocs == 1) { // If Flags==Transposed_Out return
		return;
	}
	if (Flags[0] == 1) { // If Flags==Transposed_In This function can not handle it, call other versions
		fast_transpose_v_hi(T_plan, (T*) data, timings, kway, flags, howmany,
				tag, method);
		return;
	}
	timings[0] -= MPI_Wtime();
	int nprocs, procid;
	int nprocs_0, nprocs_1;
	nprocs = T_plan->nprocs;
	procid = T_plan->procid;
	nprocs_0 = T_plan->nprocs_0;
	nprocs_1 = T_plan->nprocs_1;
	ptrdiff_t *N = T_plan->N;
	T * send_recv = T_plan->buffer;
	T * buffer_2 = T_plan->buffer_2;
	ptrdiff_t local_n0 = T_plan->local_n0;
	ptrdiff_t local_n1 = T_plan->local_n1;
	ptrdiff_t n_tuples = T_plan->n_tuples;

	int idist = N[1] * local_n0 * n_tuples;
	int odist = N[0] * local_n1 * n_tuples;

	double comm_time = 0, shuffle_time = 0, reshuffle_time = 0, total_time = 0;

#ifdef VERBOSE2
	if(VERBOSE>=2) PCOUT<<"INPUT:"<<std::endl;
	if(VERBOSE>=2) PCOUT<<"ln0:"<<local_n0<<std::endl;
	if(VERBOSE>=2) PCOUT<<"N[1]= :"<<N[1]<<std::endl;
	if(VERBOSE>=2)
	for(int h=0;h<howmany;h++)
	for(int id=0;id<nprocs;++id) {
		if(procid==id)
		for(int i=0;i<local_n0;i++) {
			std::cout<<std::endl;
			for(int j=0;j<N[1];j++) {
				std::cout<<'\t'<<data[h*idist+(i*N[1]+j)*n_tuples];
			}
		}
		std::cout<<'\n';
	}
#endif
	//PCOUT<<" ============================================= "<<std::endl;
	//PCOUT<<" ==============   Local Transpose============= "<<std::endl;
	//PCOUT<<" ============================================= "<<std::endl;
	int ptr = 0;
	shuffle_time -= MPI_Wtime();

	if (nprocs == 1 && Flags[0] == 1 && Flags[1] == 1) {
		local_transpose(local_n1, N[0], n_tuples, data);
	}
	if (nprocs == 1 && Flags[0] == 0 && Flags[1] == 0) {
    if(data_out == NULL)
		  local_transpose(N[0], N[1], n_tuples, data);
    else
		  local_transpose(N[0], N[1], n_tuples, data, data_out);
	}
	if (nprocs == 1) { // Transpose is done!
		shuffle_time += MPI_Wtime();
		timings[0] += MPI_Wtime();
		timings[1] += shuffle_time;
		timings[2] += 0;
		timings[3] += 0;
		return;
	}

	// The idea is this: If The Output is to be transposed, then only one buffer is needed. The algorithm would be:
	// data --T-> send_recv   ...  send_recv --alltoall--> data
	// Otherwise buffer2 needs to be used as well:
	// data --T-> buffer_2    ...  buffer_2 --alltoall--> send_recv ... send_recv -T-> data
	if (!comm_test)
		if (Flags[1] == 1) {
			local_transpose_col(local_n0, nprocs_1,
					n_tuples * T_plan->local_n1_proc[0],
					n_tuples * T_plan->last_local_n1, data, send_recv);
		} else if (Flags[0] == 0 && Flags[1] == 0) {
			local_transpose_col(local_n0, nprocs_1,
					n_tuples * T_plan->local_n1_proc[0],
					n_tuples * T_plan->last_local_n1, data, T_plan->buffer_2);
		}

	shuffle_time += MPI_Wtime();
#ifdef VERBOSE2
	ptr=0;
	if(VERBOSE>=2) PCOUT<<"Local Transpose:"<<std::endl;
	if(VERBOSE>=2)
	for(int h=0;h<howmany;h++)
	for(int id=0;id<nprocs;++id) {
		if(procid==id)
		for(int i=0;i<N[1];i++) {
			std::cout<<std::endl;
			for(int j=0;j<local_n0;j++) {
				std::cout<<'\t'<<T_plan->buffer_2[ptr]; //data[h*idist+(i*local_n0+j)*n_tuples];
				ptr+=n_tuples;
			}
		}
		std::cout<<'\n';
	}

#endif

	//PCOUT<<" ============================================= "<<std::endl;
	//PCOUT<<" ==============   MPIALLTOALL  =============== "<<std::endl;
	//PCOUT<<" ============================================= "<<std::endl;

	int* scount_proc = T_plan->scount_proc;
	int* rcount_proc = T_plan->rcount_proc;
	int* soffset_proc = T_plan->soffset_proc;
	int* roffset_proc = T_plan->roffset_proc;
	int* scount_proc_f = T_plan->scount_proc_f;
	int* rcount_proc_f = T_plan->rcount_proc_f;
	int* soffset_proc_f = T_plan->soffset_proc_f;
	int* roffset_proc_f = T_plan->roffset_proc_f;

	//PCOUT<<"nprocs_0= "<<nprocs_0<<" nprocs_1= "<<nprocs_1<<std::endl;
	comm_time -= MPI_Wtime();

	int soffset = 0, roffset = 0;
	MPI_Status ierr;
	int counter = 1;
	T *s_buf, *r_buf;
	if (Flags[1] == 1) {
		s_buf = send_recv;
		r_buf = data;
	} else if (Flags[0] == 0 && Flags[1] == 0) {
		s_buf = buffer_2;
		r_buf = send_recv;
	}

	if (method == 1) {
		MPI_Request * request = new MPI_Request[2 * nprocs];
		request[2 * procid] = MPI_REQUEST_NULL;
		request[2 * procid + 1] = MPI_REQUEST_NULL;
		// SEND
		int dst_r, dst_s;
		for (int i = 0; i < nprocs; ++i) {
			dst_r = (procid + i) % nprocs;
			dst_s = (procid - i + nprocs) % nprocs;
			if (dst_r != procid || dst_s != nprocs) {
				roffset = roffset_proc[dst_r];
				MPI_Irecv(&r_buf[roffset], rcount_proc[dst_r], T_plan->MPI_T,
						dst_r, tag, T_plan->comm, &request[2 * dst_r + 1]);
				soffset = soffset_proc[dst_s];
				MPI_Isend(&s_buf[soffset], scount_proc[dst_s], T_plan->MPI_T,
						dst_s, tag, T_plan->comm, &request[2 * dst_s]);
			}
		}
		// Copy Your own part. See the note below for the if condition
		soffset = soffset_proc[procid]; //aoffset_proc[proc];//proc*count_proc[proc];
		roffset = roffset_proc[procid];
		memcpy(&r_buf[roffset], &s_buf[soffset],
				sizeof(T) * scount_proc[procid]);
		MPI_Waitall(2 * nprocs, request, MPI_STATUSES_IGNORE);
		delete[] request;

	} else if (method == 2) {
		if (T_plan->is_evenly_distributed == 0)
			MPI_Alltoallv(s_buf, scount_proc_f, soffset_proc_f, T_plan->MPI_T,
					r_buf, rcount_proc_f, roffset_proc_f, T_plan->MPI_T,
					T_plan->comm);
		else
			MPI_Alltoall(s_buf, scount_proc_f[0], T_plan->MPI_T, r_buf,
					rcount_proc_f[0], T_plan->MPI_T, T_plan->comm);

	} else if (method == 3) {
		if (T_plan->kway_async)
			par::Mpi_Alltoallv_dense<T, true>(s_buf, scount_proc_f,
					soffset_proc_f, r_buf, rcount_proc_f, roffset_proc_f,
					T_plan->comm, kway);
		else
			par::Mpi_Alltoallv_dense<T, false>(s_buf, scount_proc_f,
					soffset_proc_f, r_buf, rcount_proc_f, roffset_proc_f,
					T_plan->comm, kway);
	}
	comm_time += MPI_Wtime();

#ifdef VERBOSE2
	ptr=0;
	if(VERBOSE>=2) PCOUT<<"MPIAlltoAll:"<<std::endl;
	if(VERBOSE>=2)
	for(int h=0;h<howmany;h++)
	for(int id=0;id<nprocs;++id) {
		if(procid==id)
		for(int i=0;i<local_n1;i++) {
			std::cout<<std::endl;
			for(int j=0;j<N[0];j++) {
				std::cout<<'\t'<<send_recv[ptr]; //send_recv[h*odist+(i*N[0]+j)*n_tuples];//<<","<<send_recv[(i*N[0]+j)*n_tuples+1];
				ptr+=n_tuples;
			}
		}
		std::cout<<'\n';
	}
#endif

	//PCOUT<<" ============================================= "<<std::endl;
	//PCOUT<<" ============== 2nd Local Trnaspose ========== "<<std::endl;
	//PCOUT<<" ============================================= "<<std::endl;
	reshuffle_time -= MPI_Wtime();
	ptr = 0;
	if (!comm_test)
		if (Flags[1] == 0 && data_out == NULL)
			local_transpose(N[0], local_n1, n_tuples, send_recv, data);
    else if (Flags[1] == 0 && data_out != NULL)
			local_transpose(N[0], local_n1, n_tuples, send_recv, data_out);
	reshuffle_time += MPI_Wtime();

#ifdef VERBOSE2
	if(VERBOSE>=2) PCOUT<<"2nd Transpose"<<std::endl;
	if(VERBOSE>=2)
	for(int h=0;h<howmany;h++)
	for(int id=0;id<nprocs_1;++id) {
		if(procid==id)
		for(int i=0;i<N[0];i++) {
			std::cout<<std::endl;
			for(int j=0;j<local_n1;j++) {
				std::cout<<'\t'<<data[h*odist+(i*local_n1+j)*n_tuples];
			}
		}
		std::cout<<'\n';
	}
#endif

#ifdef VERBOSE1
	PCOUT<<"Shuffle Time= "<<shuffle_time<<std::endl;
	PCOUT<<"Alltoall Time= "<<comm_time<<std::endl;
	PCOUT<<"Reshuffle Time= "<<reshuffle_time<<std::endl;
	PCOUT<<"Total Time= "<<(shuffle_time+comm_time+reshuffle_time)<<std::endl;
#endif
	timings[0] += MPI_Wtime(); //timings[0]+=shuffle_time+comm_time+reshuffle_time;
	timings[1] += shuffle_time;
	timings[2] += comm_time;
	timings[3] += reshuffle_time;
	return;
} // end fast_transpose_v

template<typename T>
void fast_transpose_v_h(T_Plan<T>* __restrict T_plan, T * __restrict data, double * __restrict timings, int kway,
		unsigned flags, int howmany, int tag, int method, int comm_test, T* __restrict data_out) {

	std::bitset < 8 > Flags(flags); // 1 Transposed in, 2 Transposed out
	if (Flags[1] == 1 && Flags[0] == 0 && T_plan->nprocs == 1) { // If Flags==Transposed_Out return
		return;
	}
	if (howmany == 1) {
		return fast_transpose_v1(T_plan, data, timings, flags, howmany, tag);
	}
	if (Flags[0] == 1) { // If Flags==Transposed_In This function can not handle it, call other versions
		fast_transpose_v_hi(T_plan, (T*) data, timings, kway, flags, howmany,
				tag, method);
		return;
	}
	timings[0] -= MPI_Wtime();
	int nprocs, procid;
	int nprocs_0, nprocs_1;
	nprocs = T_plan->nprocs;
	procid = T_plan->procid;
	nprocs_0 = T_plan->nprocs_0;
	nprocs_1 = T_plan->nprocs_1;
	ptrdiff_t *N = T_plan->N;
	T * send_recv = T_plan->buffer;
	T * buffer_2 = T_plan->buffer_2;
	ptrdiff_t local_n0 = T_plan->local_n0;
	ptrdiff_t local_n1 = T_plan->local_n1;
	ptrdiff_t n_tuples = T_plan->n_tuples;

	int idist = N[1] * local_n0 * n_tuples;
	int odist = N[0] * local_n1 * n_tuples;

	double comm_time = 0, shuffle_time = 0, reshuffle_time = 0, total_time = 0;

	int ptr = 0;
#ifdef VERBOSE2
	if(VERBOSE>=2) PCOUT<<"INPUT:"<<std::endl;
	if(VERBOSE>=2)
	for(int h=0;h<howmany;h++)
	for(int id=0;id<nprocs;++id) {
		if(procid==id)
		for(int i=0;i<local_n0;i++) {
			std::cout<<std::endl;
			for(int j=0;j<N[1];j++) {
				ptr=h*idist+(i*N[1]+j)*n_tuples;
				std::cout<<'\t'<<data[ptr]<<","<<data[ptr+1];
			}
		}
		std::cout<<'\n';
	}
#endif

	//PCOUT<<" ============================================= "<<std::endl;
	//PCOUT<<" ==============   Local Transpose============= "<<std::endl;
	//PCOUT<<" ============================================= "<<std::endl;
	ptr = 0;
	ptrdiff_t *local_n1_proc = &T_plan->local_n1_proc[0];
	ptrdiff_t *local_n0_proc = &T_plan->local_n0_proc[0];
	ptrdiff_t *local_0_start_proc = T_plan->local_0_start_proc;
	ptrdiff_t *local_1_start_proc = T_plan->local_1_start_proc;
	shuffle_time -= MPI_Wtime();
	if (nprocs == 1 && Flags[0] == 1 && Flags[1] == 1) {
		for (int h = 0; h < howmany; h++)
			local_transpose(local_n1, N[0], n_tuples, &data[h * idist]);
	}
	if (nprocs == 1 && Flags[0] == 0 && Flags[1] == 0) {
    if(data_out == NULL)
		  for (int h = 0; h < howmany; h++)
        local_transpose(N[0], N[1], n_tuples, &data[h * idist]);
    else if (data_out != NULL)
		  for (int h = 0; h < howmany; h++)
        local_transpose(N[0], N[1], n_tuples, &data[h * idist], &data_out[h * idist]);
	}
	if (nprocs == 1) { // Transpose is done!
		shuffle_time += MPI_Wtime();
		timings[0] += MPI_Wtime();
		timings[1] += shuffle_time;
		timings[2] += 0;
		timings[3] += 0;
		return;
	}

	// The idea is this: If The Output is to be transposed, then only one buffer is needed. The algorithm would be:
	// data --T-> send_recv   ...  send_recv --alltoall--> data
	// Otherwise buffer2 needs to be used as well:
	// data --T-> buffer_2    ...  buffer_2 --alltoall--> send_recv ... send_recv -T-> data
	ptr = 0;

	if (!comm_test)
		for (int proc = 0; proc < nprocs_1; ++proc)
			for (int h = 0; h < howmany; ++h) {
				for (int i = 0; i < local_n0; ++i) {
					//for(int j=local_1_start_proc[proc];j<local_1_start_proc[proc]+local_n1_proc[proc];++j){
					//  memcpy(&buffer_2[ptr],&data[h*idist+(i*N[1]+j)*n_tuples],sizeof(T)*n_tuples);
					//  //std::cout<<"proc= "<<proc<<" h= "<<h<<" (i,j)=("<<i<<","<<j<<")  data_ptr= "<<h*idist+(i*local_n1+j)*n_tuples<< " ptr= "<<ptr <<" cpy= "<<n_tuples*T_plan->local_n1_proc[0] <<std::endl;
					//  ptr+=n_tuples;
					//}
					memcpy(&buffer_2[ptr],
							&data[h * idist
									+ (i * N[1] + local_1_start_proc[proc])
											* n_tuples],
							sizeof(T) * n_tuples * local_n1_proc[proc]);
					ptr += n_tuples * local_n1_proc[proc]; // pointer is going contiguous along buffer_2
				}
			}

	shuffle_time += MPI_Wtime();
	ptr = 0;
#ifdef VERBOSE2
	if(VERBOSE>=2) PCOUT<<"Local Transpose:"<<std::endl;
	if(VERBOSE>=2)
	for(int id=0;id<nprocs;++id) {
		for(int h=0;h<howmany;h++)
		if(procid==id)
		for(int i=0;i<N[1];i++) {
			std::cout<<std::endl;
			for(int j=0;j<local_n0;j++) {
				std::cout<<'\t'<<buffer_2[ptr]<<","<<buffer_2[ptr+1];
				ptr+=n_tuples;
			}
		}
		std::cout<<'\n';
	}

#endif

	//PCOUT<<" ============================================= "<<std::endl;
	//PCOUT<<" ==============   MPIALLTOALL  =============== "<<std::endl;
	//PCOUT<<" ============================================= "<<std::endl;

	int* scount_proc = T_plan->scount_proc;
	int* rcount_proc = T_plan->rcount_proc;
	int* soffset_proc = T_plan->soffset_proc;
	int* roffset_proc = T_plan->roffset_proc;
	int* scount_proc_f = T_plan->scount_proc_f;
	int* rcount_proc_f = T_plan->rcount_proc_f;
	int* soffset_proc_f = T_plan->soffset_proc_f;
	int* roffset_proc_f = T_plan->roffset_proc_f;

	//PCOUT<<"nprocs_0= "<<nprocs_0<<" nprocs_1= "<<nprocs_1<<std::endl;
	comm_time -= MPI_Wtime();

	int soffset = 0, roffset = 0;
	MPI_Status ierr;

	int counter = 1;

	T *s_buf, *r_buf;
	s_buf = buffer_2;
	r_buf = send_recv;
	if (method == 1) {
		// SEND
		MPI_Request * request = new MPI_Request[2 * nprocs];

		int dst_r, dst_s;
		request[2 * procid] = MPI_REQUEST_NULL;
		request[2 * procid + 1] = MPI_REQUEST_NULL;
		for (int i = 0; i < nprocs; ++i) {
			dst_r = (procid + i) % nprocs;
			dst_s = (procid - i + nprocs) % nprocs;
			if (dst_r != procid && dst_s != procid) {
				roffset = roffset_proc[dst_r];
				MPI_Irecv(&r_buf[roffset * howmany],
						rcount_proc[dst_r] * howmany, T_plan->MPI_T, dst_r, tag,
						T_plan->comm, &request[2 * dst_r + 1]);
				soffset = soffset_proc[dst_s];
				MPI_Isend(&s_buf[soffset * howmany],
						scount_proc[dst_s] * howmany, T_plan->MPI_T, dst_s, tag,
						T_plan->comm, &request[2 * dst_s]);
			}
		}
		soffset = soffset_proc[procid]; //aoffset_proc[proc];//proc*count_proc[proc];
		roffset = roffset_proc[procid];
		memcpy(&r_buf[roffset * howmany], &s_buf[soffset * howmany],
				howmany * sizeof(T) * scount_proc[procid]);

		MPI_Waitall(2 * nprocs, request, MPI_STATUSES_IGNORE);
		delete[] request;

	} else if (method == 2) {
		if (T_plan->is_evenly_distributed == 0)
			MPI_Alltoallv(s_buf, scount_proc_f, soffset_proc_f, T_plan->MPI_T,
					r_buf, rcount_proc_f, roffset_proc_f, T_plan->MPI_T,
					T_plan->comm);
		else
			MPI_Alltoall(s_buf, scount_proc_f[0], T_plan->MPI_T, r_buf,
					rcount_proc_f[0], T_plan->MPI_T, T_plan->comm);
	} else if (method == 3) {
		if (T_plan->kway_async)
			par::Mpi_Alltoallv_dense<T, true>(s_buf, scount_proc_f,
					soffset_proc_f, r_buf, rcount_proc_f, roffset_proc_f,
					T_plan->comm, kway);
		else
			par::Mpi_Alltoallv_dense<T, false>(s_buf, scount_proc_f,
					soffset_proc_f, r_buf, rcount_proc_f, roffset_proc_f,
					T_plan->comm, kway);

	}
	comm_time += MPI_Wtime();

	ptr = 0;
#ifdef VERBOSE2
	if(VERBOSE>=2) PCOUT<<"MPIAlltoAll:"<<std::endl;
	if(VERBOSE>=2)
	for(int id=0;id<nprocs;++id) {
		if(procid==id)
		for(int h=0;h<howmany;h++)
		for(int i=0;i<local_n1;i++) {
			std::cout<<std::endl;
			for(int j=0;j<N[0];j++) {
				std::cout<<'\t'<<send_recv[ptr]<<","<<send_recv[ptr+1];
				ptr+=n_tuples;
			}
		}
		std::cout<<'\n';
	}
#endif

	//PCOUT<<" ============================================= "<<std::endl;
	//PCOUT<<" ============== 2nd Local Trnaspose ========== "<<std::endl;
	//PCOUT<<" ============================================= "<<std::endl;
	reshuffle_time -= MPI_Wtime();
	ptr = 0;
  T* alias = data;
  // if(Flags[1] == 0 && data_out != NULL)
    // alias = buffer_2;
	if (!comm_test && data_out==NULL)
    local_transpose(nprocs_0, howmany, n_tuples*local_n0_proc[0]*local_n1, n_tuples*T_plan->last_local_n0*local_n1, send_recv, alias);
		//for (int proc = 0; proc < nprocs_0; ++proc)
		//	for (int h = 0; h < howmany; ++h) {
		//		//for(int i=local_0_start_proc[proc];i<local_0_start_proc[proc]+local_n0_proc[proc];++i){
		//		//  memcpy(&data[h*odist+(i*local_n1)*n_tuples],&send_recv[ptr],local_n1*sizeof(T)*n_tuples);
		//		//  //std::cout<<"proc= "<<proc<<" h= "<<h<<" i=("<<i<<")  data_ptr= "<<h*odist+(i*local_n1)*n_tuples<< " ptr= "<<ptr <<" cpy= "<<n_tuples*local_n1_proc[proc] <<std::endl;
		//		//  ptr+=n_tuples*local_n1;
		//		//  //for(int j=0*local_1_start_proc[proc];j<0*local_1_start_proc[proc]+local_n1;++j){
		//		//  //  memcpy(&data[h*odist+(i*local_n1+j)*n_tuples],&send_recv[ptr],sizeof(T)*n_tuples);
		//		//  //  //std::cout<<"proc= "<<proc<<" h= "<<h<<" (i,j)=("<<i<<","<<j<<")  data_ptr= "<<h*idist+(i*local_n1+j)*n_tuples<< " ptr= "<<ptr <<" cpy= "<<n_tuples*T_plan->local_n1_proc[0] <<std::endl;
		//		//  //  ptr+=n_tuples;
		//		//  //}
		//		//}
		//		memcpy(
		//				&alias[h * odist
		//						+ local_0_start_proc[proc] * local_n1 * n_tuples],
		//				&send_recv[ptr],
		//				local_n1 * sizeof(T) * n_tuples * local_n0_proc[proc]);
		//		ptr += n_tuples * local_n1 * local_n0_proc[proc];

		//	}

	// Right now the data is in transposed out format.
	// If the user did not want this layout, transpose again.
	if (!comm_test)
		if (Flags[1] == 0 && data_out == NULL) {
			for (int h = 0; h < howmany; h++)
				local_transpose(N[0], local_n1, n_tuples, &data[h * odist]);
		}
    else if (Flags[1] == 0 && data_out != NULL) {
      // for (int h = 0; h < howmany; h++)
        // local_transpose(N[0], local_n1, n_tuples, &alias[h * odist], &data_out[h*odist]);
      ptr = 0;
	    for (int h = 0; h < howmany; ++h)
        for(int j = 0; j < local_n1; ++j){
		      for (int proc = 0; proc < nprocs_0; ++proc) {
            for(int i = 0; i < local_n0_proc[proc]; ++i){
              int off = (h)* local_n1 * local_n0_proc[proc]*n_tuples+roffset_proc[proc]*howmany;
              memcpy(&data_out[ptr], &send_recv[off+(i*local_n1+j)*n_tuples], n_tuples* sizeof(T));
              ptr += n_tuples;
            }
          }
        }
		}

	reshuffle_time += MPI_Wtime();
#ifdef VERBOSE2
	if(VERBOSE>=2) PCOUT<<"2nd Transpose"<<std::endl;
	if(VERBOSE>=2)
	for(int id=0;id<nprocs_1;++id) {
		if(procid==id)
		for(int h=0;h<howmany;h++)
		for(int i=0;i<N[0];i++) {
			std::cout<<std::endl;
			for(int j=0;j<local_n1;j++) {
				ptr=h*odist+(i*local_n1+j)*n_tuples;
				std::cout<<'\t'<<data[ptr]<<","<<data[ptr+1];
			}
		}
		std::cout<<'\n';
	}
#endif

#ifdef VERBOSE2
	if (VERBOSE >= 1) {
		PCOUT << "Shuffle Time= " << shuffle_time << std::endl;
		PCOUT << "Alltoall Time= " << comm_time << std::endl;
		PCOUT << "Reshuffle Time= " << reshuffle_time << std::endl;
		PCOUT << "Total Time= " << (shuffle_time + comm_time + reshuffle_time)
				<< std::endl;
	}
#endif
	timings[0] += MPI_Wtime(); //timings[0]+=shuffle_time+comm_time+reshuffle_time;
	timings[1] += shuffle_time;
	timings[2] += comm_time;
	timings[3] += reshuffle_time;
	return;
}  // end fast_transpose_v_h

template<typename T>
void fast_transpose_v1(T_Plan<T>* __restrict T_plan, T * __restrict data, double * __restrict timings,
		unsigned flags, int howmany, int tag, T* __restrict data_out) {

	if (howmany > 1) {
		return fast_transpose_v1_h(T_plan, data, timings, flags, howmany, tag);
	}
	std::bitset < 8 > Flags(flags); // 1 Transposed in, 2 Transposed out
	if (Flags[1] == 1 && Flags[0] == 0 && T_plan->nprocs == 1) { // If Flags==Transposed_Out return
		return;
	}
	if (Flags[0] == 1) { // If Flags==Transposed_In This function can not handle it, call other versions
		transpose_v5(T_plan, (T*) data, timings, flags, howmany, tag);
		return;
	}
  if(Flags[1] == 0 && data_out !=NULL) {
    std::cout << "operation not supported\n";
    return;
  }
	timings[0] -= MPI_Wtime();
	int nprocs, procid;
	int nprocs_0, nprocs_1;
	nprocs = T_plan->nprocs;
	procid = T_plan->procid;
	nprocs_0 = T_plan->nprocs_0;
	nprocs_1 = T_plan->nprocs_1;
	ptrdiff_t *N = T_plan->N;
	T * send_recv = T_plan->buffer;
	T * buffer_2 = T_plan->buffer_2;
	ptrdiff_t local_n0 = T_plan->local_n0;
	ptrdiff_t local_n1 = T_plan->local_n1;
	ptrdiff_t n_tuples = T_plan->n_tuples;

	int idist = N[1] * local_n0 * n_tuples;
	int odist = N[0] * local_n1 * n_tuples;

	double comm_time = 0, shuffle_time = 0, reshuffle_time = 0, total_time = 0;

#ifdef VERBOSE2
	if (VERBOSE >= 2)
		PCOUT << "INPUT:" << std::endl;
	if (VERBOSE >= 2)
		for (int h = 0; h < howmany; h++)
			for (int id = 0; id < nprocs; ++id) {
				if (procid == id)
					for (int i = 0; i < local_n0; i++) {
						std::cout << std::endl;
						for (int j = 0; j < N[1]; j++) {
							std::cout << '\t'
									<< data[h * idist
											+ (i * N[1] + j) * n_tuples];
						}
					}
				std::cout << '\n';
			}
#endif

	//PCOUT<<" ============================================= "<<std::endl;
	//PCOUT<<" ==============   Local Transpose============= "<<std::endl;
	//PCOUT<<" ============================================= "<<std::endl;
	int ptr = 0;
	shuffle_time -= MPI_Wtime();

	if (nprocs == 1 && Flags[0] == 1 && Flags[1] == 1) {
		for (int h = 0; h < howmany; h++)
			local_transpose(local_n1, N[0], n_tuples, &data[h * idist]);
	}
	if (nprocs == 1 && Flags[0] == 0 && Flags[1] == 0) {
		for (int h = 0; h < howmany; h++)
			local_transpose(N[0], N[1], n_tuples, &data[h * idist]);
	}
	if (nprocs == 1) { // Transpose is done!
		shuffle_time += MPI_Wtime();
		timings[0] += MPI_Wtime();
		timings[1] += shuffle_time;
		timings[2] += 0;
		timings[3] += 0;
		return;
	}

	// The idea is this: If The Output is to be transposed, then only one buffer is needed. The algorithm would be:
	// data --T-> send_recv   ...  send_recv --alltoall--> data
	// Otherwise buffer2 needs to be used as well:
	// data --T-> buffer_2    ...  buffer_2 --alltoall--> send_recv ... send_recv -T-> data
	if (Flags[1] == 1) {
		local_transpose_col(local_n0, nprocs_1,
				n_tuples * T_plan->local_n1_proc[0],
				n_tuples * T_plan->last_local_n1, data, send_recv);
	} else if (Flags[0] == 0 && Flags[1] == 0) {
		local_transpose_col(local_n0, nprocs_1,
				n_tuples * T_plan->local_n1_proc[0],
				n_tuples * T_plan->last_local_n1, data, T_plan->buffer_2);
	}

	shuffle_time += MPI_Wtime();
	ptr = 0;
#ifdef VERBOSE2
	if (VERBOSE >= 2)
		PCOUT << "Local Transpose:" << std::endl;
	if (VERBOSE >= 2)
		for (int h = 0; h < howmany; h++)
			for (int id = 0; id < nprocs; ++id) {
				if (procid == id)
					for (int i = 0; i < N[1]; i++) {
						std::cout << std::endl;
						for (int j = 0; j < local_n0; j++) {
							std::cout << '\t' << T_plan->buffer_2[ptr]; //data[h*idist+(i*local_n0+j)*n_tuples];
							ptr += n_tuples;
						}
					}
				std::cout << '\n';
			}
#endif

	//PCOUT<<" ============================================= "<<std::endl;
	//PCOUT<<" ==============   MPIALLTOALL  =============== "<<std::endl;
	//PCOUT<<" ============================================= "<<std::endl;

	int* scount_proc = T_plan->scount_proc;
	int* rcount_proc = T_plan->rcount_proc;
	int* soffset_proc = T_plan->soffset_proc;
	int* roffset_proc = T_plan->roffset_proc;


	//PCOUT<<"nprocs_0= "<<nprocs_0<<" nprocs_1= "<<nprocs_1<<std::endl;
	comm_time -= MPI_Wtime();

	int soffset = 0, roffset = 0;
	MPI_Status ierr;
	MPI_Request * s_request = new MPI_Request[nprocs];
	MPI_Request * request = new MPI_Request[nprocs];
	int counter = 1;

	s_request[procid] = MPI_REQUEST_NULL;
	request[procid] = MPI_REQUEST_NULL;
	T *s_buf, *r_buf;
	if (Flags[1] == 1) {
		s_buf = send_recv;
		r_buf = data;
	} else if (Flags[0] == 0 && Flags[1] == 0) {
		s_buf = buffer_2;
		r_buf = send_recv;
	}
	// SEND
	for (int proc = 0; proc < nprocs; ++proc) {
		if (proc != procid) {
			soffset = soffset_proc[proc];
			roffset = roffset_proc[proc];
			MPI_Isend(&s_buf[soffset], scount_proc[proc], T_plan->MPI_T, proc,
					tag, T_plan->comm, &s_request[proc]);
			MPI_Irecv(&r_buf[roffset], rcount_proc[proc], T_plan->MPI_T, proc,
					tag, T_plan->comm, &request[proc]);
		}
	}
	// Copy Your own part. See the note below for the if condition
	soffset = soffset_proc[procid]; //aoffset_proc[proc];//proc*count_proc[proc];
	roffset = roffset_proc[procid];
	for (int h = 0; h < howmany; h++)
		memcpy(&r_buf[h * odist + roffset], &s_buf[h * idist + soffset],
				sizeof(T) * scount_proc[procid]);
	for (int proc = 0; proc < nprocs; ++proc) {
		MPI_Wait(&request[proc], &ierr);
		MPI_Wait(&s_request[proc], &ierr);
	}


	comm_time += MPI_Wtime();

	ptr = 0;
#ifdef VERBOSE2
	if (VERBOSE >= 2)
		PCOUT << "MPIAlltoAll:" << std::endl;
	if (VERBOSE >= 2)
		for (int h = 0; h < howmany; h++)
			for (int id = 0; id < nprocs; ++id) {
				if (procid == id)
					for (int i = 0; i < local_n1; i++) {
						std::cout << std::endl;
						for (int j = 0; j < N[0]; j++) {
							std::cout << '\t' << send_recv[ptr]; //send_recv[h*odist+(i*N[0]+j)*n_tuples];//<<","<<send_recv[(i*N[0]+j)*n_tuples+1];
							ptr += n_tuples;
						}
					}
				std::cout << '\n';
			}
#endif

	//PCOUT<<" ============================================= "<<std::endl;
	//PCOUT<<" ============== 2nd Local Trnaspose ========== "<<std::endl;
	//PCOUT<<" ============================================= "<<std::endl;
	reshuffle_time -= MPI_Wtime();
	ptr = 0;
	if (Flags[1] == 0)
		local_transpose(N[0], local_n1, n_tuples, send_recv, data);

#ifdef VERBOSE2
	if (VERBOSE >= 2)
		PCOUT << "2nd Transpose" << std::endl;
	if (VERBOSE >= 2)
		for (int h = 0; h < howmany; h++)
			for (int id = 0; id < nprocs_1; ++id) {
				if (procid == id)
					for (int i = 0; i < local_n1; i++) {
						std::cout << std::endl;
						for (int j = 0; j < N[0]; j++) {
							std::cout << '\t'
									<< data[h * odist
											+ (i * N[0] + j) * n_tuples];
						}
					}
				std::cout << '\n';
			}
#endif

	reshuffle_time += MPI_Wtime();
	delete[] request;
	delete[] s_request;

#ifdef VERBOSE2
	if (VERBOSE >= 1) {
		PCOUT << "Shuffle Time= " << shuffle_time << std::endl;
		PCOUT << "Alltoall Time= " << comm_time << std::endl;
		PCOUT << "Reshuffle Time= " << reshuffle_time << std::endl;
		PCOUT << "Total Time= " << (shuffle_time + comm_time + reshuffle_time)
				<< std::endl;
	}
#endif
	timings[0] += MPI_Wtime(); //timings[0]+=shuffle_time+comm_time+reshuffle_time;
	timings[1] += shuffle_time;
	timings[2] += comm_time;
	timings[3] += reshuffle_time;
	return;
} // end fast_transpose_v1
template<typename T>
void fast_transpose_v2(T_Plan<T>* __restrict T_plan, T * __restrict data, double * __restrict timings,
		unsigned flags, int howmany, int tag, T* __restrict data_out) {

	if (howmany > 1) {
		return fast_transpose_v2_h(T_plan, data, timings, flags, howmany, tag);
	}
	std::bitset < 8 > Flags(flags); // 1 Transposed in, 2 Transposed out
	if (Flags[1] == 1 && Flags[0] == 0 && T_plan->nprocs == 1) { // If Flags==Transposed_Out return
		return;
	}
	if (Flags[0] == 1) { // If Flags==Transposed_In This function can not handle it, call other versions
		transpose_v6(T_plan, (T*) data, timings, flags, howmany);
		return;
	}
  if(Flags[1] == 0 && data_out !=NULL) {
    std::cout << "operation not supported\n";
    return;
  }
	timings[0] -= MPI_Wtime();
	int nprocs, procid;
	int nprocs_0, nprocs_1;
	nprocs = T_plan->nprocs;
	procid = T_plan->procid;
	nprocs_0 = T_plan->nprocs_0;
	nprocs_1 = T_plan->nprocs_1;
	ptrdiff_t *N = T_plan->N;
	T * send_recv = T_plan->buffer;
	T * buffer_2 = T_plan->buffer_2;
	ptrdiff_t local_n0 = T_plan->local_n0;
	ptrdiff_t local_n1 = T_plan->local_n1;
	ptrdiff_t n_tuples = T_plan->n_tuples;

	int idist = N[1] * local_n0 * n_tuples;
	int odist = N[0] * local_n1 * n_tuples;

	double comm_time = 0, shuffle_time = 0, reshuffle_time = 0, total_time = 0;

#ifdef VERBOSE2
	if (VERBOSE >= 2)
		PCOUT << "INPUT:" << std::endl;
	if (VERBOSE >= 2)
		for (int h = 0; h < howmany; h++)
			for (int id = 0; id < nprocs; ++id) {
				if (procid == id)
					for (int i = 0; i < local_n0; i++) {
						std::cout << std::endl;
						for (int j = 0; j < N[1]; j++) {
							std::cout << '\t'
									<< data[h * idist
											+ (i * N[1] + j) * n_tuples];
						}
					}
				std::cout << '\n';
			}
#endif

	//PCOUT<<" ============================================= "<<std::endl;
	//PCOUT<<" ==============   Local Transpose============= "<<std::endl;
	//PCOUT<<" ============================================= "<<std::endl;
	int ptr = 0;
	shuffle_time -= MPI_Wtime();

	if (nprocs == 1 && Flags[0] == 1 && Flags[1] == 1) {
		for (int h = 0; h < howmany; h++)
			local_transpose(local_n1, N[0], n_tuples, &data[h * idist]);
	}
	if (nprocs == 1 && Flags[0] == 0 && Flags[1] == 0) {
		for (int h = 0; h < howmany; h++)
			local_transpose(N[0], N[1], n_tuples, &data[h * idist]);
	}
	if (nprocs == 1) { // Transpose is done!
		shuffle_time += MPI_Wtime();
		timings[0] += MPI_Wtime();
		timings[1] += shuffle_time;
		timings[2] += 0;
		timings[3] += 0;
		return;
	}

	// The idea is this: If The Output is to be transposed, then only one buffer is needed. The algorithm would be:
	// data --T-> send_recv   ...  send_recv --alltoall--> data
	// Otherwise buffer2 needs to be used as well:
	// data --T-> buffer_2    ...  buffer_2 --alltoall--> send_recv ... send_recv -T-> data
	if (Flags[1] == 1) {
		local_transpose_col(local_n0, nprocs_1,
				n_tuples * T_plan->local_n1_proc[0],
				n_tuples * T_plan->last_local_n1, data, send_recv);
	} else if (Flags[0] == 0 && Flags[1] == 0) {
		local_transpose_col(local_n0, nprocs_1,
				n_tuples * T_plan->local_n1_proc[0],
				n_tuples * T_plan->last_local_n1, data, T_plan->buffer_2);
	}

	shuffle_time += MPI_Wtime();
	ptr = 0;
#ifdef VERBOSE2
	if (VERBOSE >= 2)
		PCOUT << "Local Transpose:" << std::endl;
	if (VERBOSE >= 2)
		for (int h = 0; h < howmany; h++)
			for (int id = 0; id < nprocs; ++id) {
				if (procid == id)
					for (int i = 0; i < N[1]; i++) {
						std::cout << std::endl;
						for (int j = 0; j < local_n0; j++) {
							std::cout << '\t' << T_plan->buffer_2[ptr]; //data[h*idist+(i*local_n0+j)*n_tuples];
							ptr += n_tuples;
						}
					}
				std::cout << '\n';
			}
#endif

	//PCOUT<<" ============================================= "<<std::endl;
	//PCOUT<<" ==============   MPIALLTOALL  =============== "<<std::endl;
	//PCOUT<<" ============================================= "<<std::endl;

	int* scount_proc_f = T_plan->scount_proc_f;
	int* rcount_proc_f = T_plan->rcount_proc_f;
	int* soffset_proc_f = T_plan->soffset_proc_f;
	int* roffset_proc_f = T_plan->roffset_proc_f;


	T *s_buf, *r_buf;
	if (Flags[1] == 1) {
		s_buf = send_recv;
		r_buf = data;
	} else if (Flags[0] == 0 && Flags[1] == 0) {
		s_buf = buffer_2;
		r_buf = send_recv;
	}
	//PCOUT<<"nprocs_0= "<<nprocs_0<<" nprocs_1= "<<nprocs_1<<std::endl;
	comm_time -= MPI_Wtime();
	if (T_plan->is_evenly_distributed == 0)
		MPI_Alltoallv(s_buf, scount_proc_f, soffset_proc_f, T_plan->MPI_T,
				r_buf, rcount_proc_f, roffset_proc_f, T_plan->MPI_T,
				T_plan->comm);
	else
		MPI_Alltoall(s_buf, scount_proc_f[0], T_plan->MPI_T, r_buf,
				rcount_proc_f[0], T_plan->MPI_T, T_plan->comm);

	comm_time += MPI_Wtime();

	ptr = 0;
#ifdef VERBOSE2
	if (VERBOSE >= 2)
		PCOUT << "MPIAlltoAll:" << std::endl;
	if (VERBOSE >= 2)
		for (int h = 0; h < howmany; h++)
			for (int id = 0; id < nprocs; ++id) {
				if (procid == id)
					for (int i = 0; i < local_n1; i++) {
						std::cout << std::endl;
						for (int j = 0; j < N[0]; j++) {
							std::cout << '\t' << send_recv[ptr]; //send_recv[h*odist+(i*N[0]+j)*n_tuples];//<<","<<send_recv[(i*N[0]+j)*n_tuples+1];
							ptr += n_tuples;
						}
					}
				std::cout << '\n';
			}
#endif

	//PCOUT<<" ============================================= "<<std::endl;
	//PCOUT<<" ============== 2nd Local Trnaspose ========== "<<std::endl;
	//PCOUT<<" ============================================= "<<std::endl;
	reshuffle_time -= MPI_Wtime();
	ptr = 0;
	if (Flags[1] == 0)
		local_transpose(N[0], local_n1, n_tuples, send_recv, data);

#ifdef VERBOSE2
	if (VERBOSE >= 2)
		PCOUT << "2nd Transpose" << std::endl;
	if (VERBOSE >= 2)
		for (int h = 0; h < howmany; h++)
			for (int id = 0; id < nprocs_1; ++id) {
				if (procid == id)
					for (int i = 0; i < local_n1; i++) {
						std::cout << std::endl;
						for (int j = 0; j < N[0]; j++) {
							std::cout << '\t'
									<< data[h * odist
											+ (i * N[0] + j) * n_tuples];
						}
					}
				std::cout << '\n';
			}
#endif

	reshuffle_time += MPI_Wtime();

#ifdef VERBOSE1
	if (VERBOSE >= 1) {
		PCOUT << "Shuffle Time= " << shuffle_time << std::endl;
		PCOUT << "Alltoall Time= " << comm_time << std::endl;
		PCOUT << "Reshuffle Time= " << reshuffle_time << std::endl;
		PCOUT << "Total Time= " << (shuffle_time + comm_time + reshuffle_time)
				<< std::endl;
	}
#endif
	timings[0] += MPI_Wtime(); //timings[0]+=shuffle_time+comm_time+reshuffle_time;
	timings[1] += shuffle_time;
	timings[2] += comm_time;
	timings[3] += reshuffle_time;
	return;
}  // end fast_transpose_v2
template<typename T>
void fast_transpose_v3(T_Plan<T>* __restrict T_plan, T * __restrict data, double * __restrict timings, int kway,
		unsigned flags, int howmany, int tag, T* __restrict data_out) {

	if (howmany > 1) {
		return fast_transpose_v3_h(T_plan, data, timings, kway, flags, howmany,
				tag);
	}
	std::bitset < 8 > Flags(flags); // 1 Transposed in, 2 Transposed out
	if (Flags[1] == 1 && Flags[0] == 0 && T_plan->nprocs == 1) { // If Flags==Transposed_Out return
		return;
	}
	if (Flags[0] == 1) { // If Flags==Transposed_In This function can not handle it, call other versions
		transpose_v7(T_plan, (T*) data, timings, kway, flags, howmany);
		return;
	}
  if(Flags[1] == 0 && data_out !=NULL) {
    std::cout << "operation not supported\n";
    return;
  }
	timings[0] -= MPI_Wtime();
	int nprocs, procid;
	int nprocs_0, nprocs_1;
	nprocs = T_plan->nprocs;
	procid = T_plan->procid;
	nprocs_0 = T_plan->nprocs_0;
	nprocs_1 = T_plan->nprocs_1;
	ptrdiff_t *N = T_plan->N;
	T * send_recv = T_plan->buffer;
	T * buffer_2 = T_plan->buffer_2;
	ptrdiff_t local_n0 = T_plan->local_n0;
	ptrdiff_t local_n1 = T_plan->local_n1;
	ptrdiff_t n_tuples = T_plan->n_tuples;

	int idist = N[1] * local_n0 * n_tuples;
	int odist = N[0] * local_n1 * n_tuples;

	double comm_time = 0, shuffle_time = 0, reshuffle_time = 0, total_time = 0;

#ifdef VERBOSE2
	if (VERBOSE >= 2)
		PCOUT << "INPUT:" << std::endl;
	if (VERBOSE >= 2)
		for (int h = 0; h < howmany; h++)
			for (int id = 0; id < nprocs; ++id) {
				if (procid == id)
					for (int i = 0; i < local_n0; i++) {
						std::cout << std::endl;
						for (int j = 0; j < N[1]; j++) {
							std::cout << '\t'
									<< data[h * idist
											+ (i * N[1] + j) * n_tuples];
						}
					}
				std::cout << '\n';
			}
#endif

	//PCOUT<<" ============================================= "<<std::endl;
	//PCOUT<<" ==============   Local Transpose============= "<<std::endl;
	//PCOUT<<" ============================================= "<<std::endl;
	int ptr = 0;
#ifdef VERBOSE2
	if (VERBOSE >= 1)
		PCOUT << "Performing shuffle ....";
#endif
	shuffle_time -= MPI_Wtime();

	if (nprocs == 1 && Flags[0] == 1 && Flags[1] == 1) {
		for (int h = 0; h < howmany; h++)
			local_transpose(local_n1, N[0], n_tuples, &data[h * idist]);
	}
	if (nprocs == 1 && Flags[0] == 0 && Flags[1] == 0) {
		for (int h = 0; h < howmany; h++)
			local_transpose(N[0], N[1], n_tuples, &data[h * idist]);
	}
	if (nprocs == 1) { // Transpose is done!
		shuffle_time += MPI_Wtime();
		timings[0] += MPI_Wtime();
		timings[1] += shuffle_time;
		timings[2] += 0;
		timings[3] += 0;
		return;
	}

	// The idea is this: If The Output is to be transposed, then only one buffer is needed. The algorithm would be:
	// data --T-> send_recv   ...  send_recv --alltoall--> data
	// Otherwise buffer2 needs to be used as well:
	// data --T-> buffer_2    ...  buffer_2 --alltoall--> send_recv ... send_recv -T-> data
	if (Flags[1] == 1) {
		local_transpose_col(local_n0, nprocs_1,
				n_tuples * T_plan->local_n1_proc[0],
				n_tuples * T_plan->last_local_n1, data, send_recv);
	} else if (Flags[0] == 0 && Flags[1] == 0) {
		local_transpose_col(local_n0, nprocs_1,
				n_tuples * T_plan->local_n1_proc[0],
				n_tuples * T_plan->last_local_n1, data, T_plan->buffer_2);
	}

	shuffle_time += MPI_Wtime();
#ifdef VERBOSE2
	if (VERBOSE >= 1)
		PCOUT << " done\n";
	ptr = 0;
	if (VERBOSE >= 2)
		PCOUT << "Local Transpose:" << std::endl;
	if (VERBOSE >= 2)
		for (int h = 0; h < howmany; h++)
			for (int id = 0; id < nprocs; ++id) {
				if (procid == id)
					for (int i = 0; i < N[1]; i++) {
						std::cout << std::endl;
						for (int j = 0; j < local_n0; j++) {
							std::cout << '\t' << T_plan->buffer_2[ptr]; //data[h*idist+(i*local_n0+j)*n_tuples];
							ptr += n_tuples;
						}
					}
				std::cout << '\n';
			}
#endif

	//PCOUT<<" ============================================= "<<std::endl;
	//PCOUT<<" ==============   MPIALLTOALL  =============== "<<std::endl;
	//PCOUT<<" ============================================= "<<std::endl;
	int* scount_proc_f = T_plan->scount_proc_f;
	int* rcount_proc_f = T_plan->rcount_proc_f;
	int* soffset_proc_f = T_plan->soffset_proc_f;
	int* roffset_proc_f = T_plan->roffset_proc_f;

	T *s_buf, *r_buf;
	if (Flags[1] == 1) {
		s_buf = send_recv;
		r_buf = data;
	} else if (Flags[0] == 0 && Flags[1] == 0) {
		s_buf = buffer_2;
		r_buf = send_recv;
	}
	//PCOUT<<"nprocs_0= "<<nprocs_0<<" nprocs_1= "<<nprocs_1<<std::endl;
#ifdef VERBOSE1
	if (VERBOSE >= 1)
		PCOUT << "Performing alltoall ....";
#endif
	comm_time -= MPI_Wtime();
	if (T_plan->kway_async)
		par::Mpi_Alltoallv_dense<T, true>(s_buf, scount_proc_f, soffset_proc_f,
				r_buf, rcount_proc_f, roffset_proc_f, T_plan->comm, kway);
	else
		par::Mpi_Alltoallv_dense<T, false>(s_buf, scount_proc_f, soffset_proc_f,
				r_buf, rcount_proc_f, roffset_proc_f, T_plan->comm, kway);
	comm_time += MPI_Wtime();
#ifdef VERBOSE1
	if (VERBOSE >= 1)
		PCOUT << "done\n";
#endif

	ptr = 0;
#ifdef VERBOSE2
	if (VERBOSE >= 2)
		PCOUT << "MPIAlltoAll:" << std::endl;
	if (VERBOSE >= 2)
		for (int h = 0; h < howmany; h++)
			for (int id = 0; id < nprocs; ++id) {
				if (procid == id)
					for (int i = 0; i < local_n1; i++) {
						std::cout << std::endl;
						for (int j = 0; j < N[0]; j++) {
							std::cout << '\t' << send_recv[ptr]; //send_recv[h*odist+(i*N[0]+j)*n_tuples];//<<","<<send_recv[(i*N[0]+j)*n_tuples+1];
							ptr += n_tuples;
						}
					}
				std::cout << '\n';
			}
#endif

	//PCOUT<<" ============================================= "<<std::endl;
	//PCOUT<<" ============== 2nd Local Trnaspose ========== "<<std::endl;
	//PCOUT<<" ============================================= "<<std::endl;
#ifdef VERBOSE1
	if (VERBOSE >= 1)
		PCOUT << "Performing reshuffle ....";
#endif
	reshuffle_time -= MPI_Wtime();
	ptr = 0;
	if (Flags[1] == 0)
		local_transpose(N[0], local_n1, n_tuples, send_recv, data);

#ifdef VERBOSE2
	if (VERBOSE >= 2)
		PCOUT << "2nd Transpose" << std::endl;
	if (VERBOSE >= 2)
		for (int h = 0; h < howmany; h++)
			for (int id = 0; id < nprocs_1; ++id) {
				if (procid == id)
					for (int i = 0; i < local_n1; i++) {
						std::cout << std::endl;
						for (int j = 0; j < N[0]; j++) {
							std::cout << '\t'
									<< data[h * odist
											+ (i * N[0] + j) * n_tuples];
						}
					}
				std::cout << '\n';
			}
#endif

	reshuffle_time += MPI_Wtime();
#ifdef VERBOSE1
	if (VERBOSE >= 1)
		PCOUT << " done\n";

	if (VERBOSE >= 1) {
		PCOUT << "Shuffle Time= " << shuffle_time << std::endl;
		PCOUT << "Alltoall Time= " << comm_time << std::endl;
		PCOUT << "Reshuffle Time= " << reshuffle_time << std::endl;
		PCOUT << "Total Time= " << (shuffle_time + comm_time + reshuffle_time)
				<< std::endl;
	}
#endif
	timings[0] += MPI_Wtime(); //timings[0]+=shuffle_time+comm_time+reshuffle_time;
	timings[1] += shuffle_time;
	timings[2] += comm_time;
	timings[3] += reshuffle_time;
	return;
}  // end fast_transpose_v3

template<typename T>
void fast_transpose_v1_h(T_Plan<T>* __restrict T_plan, T * __restrict data, double * __restrict timings,
		unsigned flags, int howmany, int tag, T* __restrict data_out) {

	std::bitset < 8 > Flags(flags); // 1 Transposed in, 2 Transposed out
	if (Flags[1] == 1 && Flags[0] == 0 && T_plan->nprocs == 1) { // If Flags==Transposed_Out return
		return;
	}
	if (howmany == 1) {
		return fast_transpose_v1(T_plan, data, timings, flags, howmany, tag);
	}
	if (Flags[0] == 1) { // If Flags==Transposed_In This function can not handle it, call other versions
		transpose_v5(T_plan, (T*) data, timings, flags, howmany, tag);
		return;
	}
  if(Flags[1] == 0 && data_out !=NULL) {
    std::cout << "operation not supported\n";
    return;
  }
	timings[0] -= MPI_Wtime();
	int nprocs, procid;
	int nprocs_0, nprocs_1;
	nprocs = T_plan->nprocs;
	procid = T_plan->procid;
	nprocs_0 = T_plan->nprocs_0;
	nprocs_1 = T_plan->nprocs_1;
	ptrdiff_t *N = T_plan->N;
	T * send_recv = T_plan->buffer;
	T * buffer_2 = T_plan->buffer_2;
	ptrdiff_t local_n0 = T_plan->local_n0;
	ptrdiff_t local_n1 = T_plan->local_n1;
	ptrdiff_t n_tuples = T_plan->n_tuples;

	int idist = N[1] * local_n0 * n_tuples;
	int odist = N[0] * local_n1 * n_tuples;

	double comm_time = 0, shuffle_time = 0, reshuffle_time = 0, total_time = 0;

	int ptr = 0;
#ifdef VERBOSE2
	if (VERBOSE >= 2)
		PCOUT << "INPUT:" << std::endl;
	if (VERBOSE >= 2)
		for (int h = 0; h < howmany; h++)
			for (int id = 0; id < nprocs; ++id) {
				if (procid == id)
					for (int i = 0; i < local_n0; i++) {
						std::cout << std::endl;
						for (int j = 0; j < N[1]; j++) {
							ptr = h * idist + (i * N[1] + j) * n_tuples;
							std::cout << '\t' << data[ptr] << ","
									<< data[ptr + 1];
						}
					}
				std::cout << '\n';
			}
#endif

	//PCOUT<<" ============================================= "<<std::endl;
	//PCOUT<<" ==============   Local Transpose============= "<<std::endl;
	//PCOUT<<" ============================================= "<<std::endl;
	ptr = 0;
	ptrdiff_t *local_n1_proc = &T_plan->local_n1_proc[0];
	ptrdiff_t *local_n0_proc = &T_plan->local_n0_proc[0];
	ptrdiff_t *local_0_start_proc = T_plan->local_0_start_proc;
	ptrdiff_t *local_1_start_proc = T_plan->local_1_start_proc;
	shuffle_time -= MPI_Wtime();
	if (nprocs == 1 && Flags[0] == 1 && Flags[1] == 1) {
		for (int h = 0; h < howmany; h++)
			local_transpose(local_n1, N[0], n_tuples, &data[h * idist]);
	}
	if (nprocs == 1 && Flags[0] == 0 && Flags[1] == 0) {
		for (int h = 0; h < howmany; h++)
			local_transpose(N[0], N[1], n_tuples, &data[h * idist]);
	}
	if (nprocs == 1) { // Transpose is done!
		shuffle_time += MPI_Wtime();
		timings[0] += MPI_Wtime();
		timings[1] += shuffle_time;
		timings[2] += 0;
		timings[3] += 0;
		return;
	}

	// The idea is this: If The Output is to be transposed, then only one buffer is needed. The algorithm would be:
	// data --T-> send_recv   ...  send_recv --alltoall--> data
	// Otherwise buffer2 needs to be used as well:
	// data --T-> buffer_2    ...  buffer_2 --alltoall--> send_recv ... send_recv -T-> data
	ptr = 0;

	for (int proc = 0; proc < nprocs_1; ++proc)
		for (int h = 0; h < howmany; ++h) {
			for (int i = 0; i < local_n0; ++i) {
				//for(int j=local_1_start_proc[proc];j<local_1_start_proc[proc]+local_n1_proc[proc];++j){
				//  memcpy(&buffer_2[ptr],&data[h*idist+(i*N[1]+j)*n_tuples],sizeof(T)*n_tuples);
				//  //std::cout<<"proc= "<<proc<<" h= "<<h<<" (i,j)=("<<i<<","<<j<<")  data_ptr= "<<h*idist+(i*local_n1+j)*n_tuples<< " ptr= "<<ptr <<" cpy= "<<n_tuples*T_plan->local_n1_proc[0] <<std::endl;
				//  ptr+=n_tuples;
				//}
				memcpy(&buffer_2[ptr],
						&data[h * idist
								+ (i * N[1] + local_1_start_proc[proc])
										* n_tuples],
						sizeof(T) * n_tuples * local_n1_proc[proc]);
				ptr += n_tuples * local_n1_proc[proc]; // pointer is going contiguous along buffer_2
			}
		}

	shuffle_time += MPI_Wtime();
	ptr = 0;
#ifdef VERBOSE2
	if (VERBOSE >= 2)
		PCOUT << "Local Transpose:" << std::endl;
	if (VERBOSE >= 2)
		for (int id = 0; id < nprocs; ++id) {
			for (int h = 0; h < howmany; h++)
				if (procid == id)
					for (int i = 0; i < N[1]; i++) {
						std::cout << std::endl;
						for (int j = 0; j < local_n0; j++) {
							std::cout << '\t' << buffer_2[ptr] << ","
									<< buffer_2[ptr + 1];
							ptr += n_tuples;
						}
					}
			std::cout << '\n';
		}
#endif

	//PCOUT<<" ============================================= "<<std::endl;
	//PCOUT<<" ==============   MPIALLTOALL  =============== "<<std::endl;
	//PCOUT<<" ============================================= "<<std::endl;

	int* scount_proc = T_plan->scount_proc;
	int* rcount_proc = T_plan->rcount_proc;
	int* soffset_proc = T_plan->soffset_proc;
	int* roffset_proc = T_plan->roffset_proc;


	//PCOUT<<"nprocs_0= "<<nprocs_0<<" nprocs_1= "<<nprocs_1<<std::endl;
	comm_time -= MPI_Wtime();

	int soffset = 0, roffset = 0;
	MPI_Status ierr;
	MPI_Request * s_request = new MPI_Request[nprocs];
	MPI_Request * request = new MPI_Request[nprocs];
	s_request[procid] = MPI_REQUEST_NULL;
	request[procid] = MPI_REQUEST_NULL;
	int counter = 1;

	T *s_buf, *r_buf;
	s_buf = buffer_2;
	r_buf = send_recv;
	// SEND
	for (int proc = 0; proc < nprocs; ++proc) {
		if (proc != procid) {
			soffset = soffset_proc[proc];
			roffset = roffset_proc[proc];
			MPI_Isend(&s_buf[soffset * howmany], scount_proc[proc] * howmany,
					T_plan->MPI_T, proc, tag, T_plan->comm, &s_request[proc]);
			MPI_Irecv(&r_buf[roffset * howmany], rcount_proc[proc] * howmany,
					T_plan->MPI_T, proc, tag, T_plan->comm, &request[proc]);
		}
	}
	soffset = soffset_proc[procid]; //aoffset_proc[proc];//proc*count_proc[proc];
	roffset = roffset_proc[procid];
	memcpy(&r_buf[roffset * howmany], &s_buf[soffset * howmany],
			howmany * sizeof(T) * scount_proc[procid]);

	for (int proc = 0; proc < nprocs; ++proc) {
		MPI_Wait(&request[proc], &ierr);
		MPI_Wait(&s_request[proc], &ierr);
	}
	comm_time += MPI_Wtime();

	ptr = 0;
#ifdef VERBOSE2
	if (VERBOSE >= 2)
		PCOUT << "MPIAlltoAll:" << std::endl;
	if (VERBOSE >= 2)
		for (int id = 0; id < nprocs; ++id) {
			if (procid == id)
				for (int h = 0; h < howmany; h++)
					for (int i = 0; i < local_n1; i++) {
						std::cout << std::endl;
						for (int j = 0; j < N[0]; j++) {
							std::cout << '\t' << send_recv[ptr] << ","
									<< send_recv[ptr + 1];
							ptr += n_tuples;
						}
					}
			std::cout << '\n';
		}
#endif

	//PCOUT<<" ============================================= "<<std::endl;
	//PCOUT<<" ============== 2nd Local Trnaspose ========== "<<std::endl;
	//PCOUT<<" ============================================= "<<std::endl;
	reshuffle_time -= MPI_Wtime();
	ptr = 0;
	//memset(data,0,T_plan->alloc_local);
	for (int proc = 0; proc < nprocs_0; ++proc)
		for (int h = 0; h < howmany; ++h) {
			//for(int i=local_0_start_proc[proc];i<local_0_start_proc[proc]+local_n0_proc[proc];++i){
			//  memcpy(&data[h*odist+(i*local_n1)*n_tuples],&send_recv[ptr],local_n1*sizeof(T)*n_tuples);
			//  //std::cout<<"proc= "<<proc<<" h= "<<h<<" i=("<<i<<")  data_ptr= "<<h*odist+(i*local_n1)*n_tuples<< " ptr= "<<ptr <<" cpy= "<<n_tuples*local_n1_proc[proc] <<std::endl;
			//  ptr+=n_tuples*local_n1;
			//  //for(int j=0*local_1_start_proc[proc];j<0*local_1_start_proc[proc]+local_n1;++j){
			//  //  memcpy(&data[h*odist+(i*local_n1+j)*n_tuples],&send_recv[ptr],sizeof(T)*n_tuples);
			//  //  //std::cout<<"proc= "<<proc<<" h= "<<h<<" (i,j)=("<<i<<","<<j<<")  data_ptr= "<<h*idist+(i*local_n1+j)*n_tuples<< " ptr= "<<ptr <<" cpy= "<<n_tuples*T_plan->local_n1_proc[0] <<std::endl;
			//  //  ptr+=n_tuples;
			//  //}
			//}
			memcpy(
					&data[h * odist
							+ local_0_start_proc[proc] * local_n1 * n_tuples],
					&send_recv[ptr],
					local_n1 * sizeof(T) * n_tuples * local_n0_proc[proc]);
			ptr += n_tuples * local_n1 * local_n0_proc[proc];

		}

	// Right now the data is in transposed out format.
	// If the user did not want this layout, transpose again.
	if (Flags[1] == 0) {
		for (int h = 0; h < howmany; h++)
			local_transpose(N[0], local_n1, n_tuples, &data[h * odist]);
	}

#ifdef VERBOSE2
	if (VERBOSE >= 2)
		PCOUT << "2nd Transpose" << std::endl;
	if (VERBOSE >= 2)
		for (int id = 0; id < nprocs_1; ++id) {
			if (procid == id)
				for (int h = 0; h < howmany; h++)
					for (int i = 0; i < N[0]; i++) {
						std::cout << std::endl;
						for (int j = 0; j < local_n1; j++) {
							ptr = h * odist + (i * local_n1 + j) * n_tuples;
							std::cout << '\t' << data[ptr] << ","
									<< data[ptr + 1];
						}
					}
			std::cout << '\n';
		}
#endif

	reshuffle_time += MPI_Wtime();
	delete[] request;
	delete[] s_request;

#ifdef VERBOSE2
	if (VERBOSE >= 1) {
		PCOUT << "Shuffle Time= " << shuffle_time << std::endl;
		PCOUT << "Alltoall Time= " << comm_time << std::endl;
		PCOUT << "Reshuffle Time= " << reshuffle_time << std::endl;
		PCOUT << "Total Time= " << (shuffle_time + comm_time + reshuffle_time)
				<< std::endl;
	}
#endif
	timings[0] += MPI_Wtime(); //timings[0]+=shuffle_time+comm_time+reshuffle_time;
	timings[1] += shuffle_time;
	timings[2] += comm_time;
	timings[3] += reshuffle_time;
	return;
}  // end fast_transpose_v1_h

template<typename T>
void fast_transpose_v2_h(T_Plan<T>* __restrict T_plan, T * __restrict data, double * __restrict timings,
		unsigned flags, int howmany, int tag, T* __restrict data_out) {

	std::bitset < 8 > Flags(flags); // 1 Transposed in, 2 Transposed out
	if (Flags[1] == 1 && Flags[0] == 0 && T_plan->nprocs == 1) { // If Flags==Transposed_Out return
		return;
	}
	if (howmany == 1) {
		return fast_transpose_v2(T_plan, data, timings, flags, howmany, tag);
	}
	if (Flags[0] == 1) { // If Flags==Transposed_In This function can not handle it, call other versions
		transpose_v6(T_plan, (T*) data, timings, flags, howmany);
		return;
	}
  if(Flags[1] == 0 && data_out !=NULL) {
    std::cout << "operation not supported\n";
    return;
  }
	timings[0] -= MPI_Wtime();
	int nprocs, procid;
	int nprocs_0, nprocs_1;
	nprocs = T_plan->nprocs;
	procid = T_plan->procid;
	nprocs_0 = T_plan->nprocs_0;
	nprocs_1 = T_plan->nprocs_1;
	ptrdiff_t *N = T_plan->N;
	T * send_recv = T_plan->buffer;
	T * buffer_2 = T_plan->buffer_2;
	ptrdiff_t local_n0 = T_plan->local_n0;
	ptrdiff_t local_n1 = T_plan->local_n1;
	ptrdiff_t n_tuples = T_plan->n_tuples;

	int idist = N[1] * local_n0 * n_tuples;
	int odist = N[0] * local_n1 * n_tuples;

	double comm_time = 0, shuffle_time = 0, reshuffle_time = 0, total_time = 0;

	int ptr = 0;
#ifdef VERBOSE2
	if (VERBOSE >= 2)
		PCOUT << "INPUT:" << std::endl;
	if (VERBOSE >= 2)
		for (int h = 0; h < howmany; h++)
			for (int id = 0; id < nprocs; ++id) {
				if (procid == id)
					for (int i = 0; i < local_n0; i++) {
						std::cout << std::endl;
						for (int j = 0; j < N[1]; j++) {
							ptr = h * idist + (i * N[1] + j) * n_tuples;
							std::cout << '\t' << data[ptr] << ","
									<< data[ptr + 1];
						}
					}
				std::cout << '\n';
			}
#endif

	//PCOUT<<" ============================================= "<<std::endl;
	//PCOUT<<" ==============   Local Transpose============= "<<std::endl;
	//PCOUT<<" ============================================= "<<std::endl;
	ptr = 0;
	ptrdiff_t *local_n1_proc = &T_plan->local_n1_proc[0];
	ptrdiff_t *local_n0_proc = &T_plan->local_n0_proc[0];
	ptrdiff_t *local_0_start_proc = T_plan->local_0_start_proc;
	ptrdiff_t *local_1_start_proc = T_plan->local_1_start_proc;
	shuffle_time -= MPI_Wtime();
	if (nprocs == 1 && Flags[0] == 1 && Flags[1] == 1) {
		for (int h = 0; h < howmany; h++)
			local_transpose(local_n1, N[0], n_tuples, &data[h * idist]);
	}
	if (nprocs == 1 && Flags[0] == 0 && Flags[1] == 0) {
		for (int h = 0; h < howmany; h++)
			local_transpose(N[0], N[1], n_tuples, &data[h * idist]);
	}
	if (nprocs == 1) { // Transpose is done!
		shuffle_time += MPI_Wtime();
		timings[0] += MPI_Wtime();
		timings[1] += shuffle_time;
		timings[2] += 0;
		timings[3] += 0;
		return;
	}

	// The idea is this: If The Output is to be transposed, then only one buffer is needed. The algorithm would be:
	// data --T-> send_recv   ...  send_recv --alltoall--> data
	// Otherwise buffer2 needs to be used as well:
	// data --T-> buffer_2    ...  buffer_2 --alltoall--> send_recv ... send_recv -T-> data
	ptr = 0;

	for (int proc = 0; proc < nprocs_1; ++proc)
		for (int h = 0; h < howmany; ++h) {
			for (int i = 0; i < local_n0; ++i) {
				//for(int j=local_1_start_proc[proc];j<local_1_start_proc[proc]+local_n1_proc[proc];++j){
				//  memcpy(&buffer_2[ptr],&data[h*idist+(i*N[1]+j)*n_tuples],sizeof(T)*n_tuples);
				//  //std::cout<<"proc= "<<proc<<" h= "<<h<<" (i,j)=("<<i<<","<<j<<")  data_ptr= "<<h*idist+(i*local_n1+j)*n_tuples<< " ptr= "<<ptr <<" cpy= "<<n_tuples*T_plan->local_n1_proc[0] <<std::endl;
				//  ptr+=n_tuples;
				//}
				memcpy(&buffer_2[ptr],
						&data[h * idist
								+ (i * N[1] + local_1_start_proc[proc])
										* n_tuples],
						sizeof(T) * n_tuples * local_n1_proc[proc]);
				ptr += n_tuples * local_n1_proc[proc];
			}
		}

	shuffle_time += MPI_Wtime();
	ptr = 0;
#ifdef VERBOSE2
	if (VERBOSE >= 2)
		PCOUT << "Local Transpose:" << std::endl;
	if (VERBOSE >= 2)
		for (int id = 0; id < nprocs; ++id) {
			for (int h = 0; h < howmany; h++)
				if (procid == id)
					for (int i = 0; i < N[1]; i++) {
						std::cout << std::endl;
						for (int j = 0; j < local_n0; j++) {
							std::cout << '\t' << buffer_2[ptr] << ","
									<< buffer_2[ptr + 1];
							ptr += n_tuples;
						}
					}
			std::cout << '\n';
		}
#endif

	//PCOUT<<" ============================================= "<<std::endl;
	//PCOUT<<" ==============   MPIALLTOALL  =============== "<<std::endl;
	//PCOUT<<" ============================================= "<<std::endl;

	int* scount_proc_f = T_plan->scount_proc_f;
	int* rcount_proc_f = T_plan->rcount_proc_f;
	int* soffset_proc_f = T_plan->soffset_proc_f;
	int* roffset_proc_f = T_plan->roffset_proc_f;


	//PCOUT<<"nprocs_0= "<<nprocs_0<<" nprocs_1= "<<nprocs_1<<std::endl;
	comm_time -= MPI_Wtime();

	int soffset = 0, roffset = 0;
	MPI_Status ierr;
	int counter = 1;

	T *s_buf, *r_buf;
	s_buf = buffer_2;
	r_buf = send_recv;
	// SEND
	if (T_plan->is_evenly_distributed == 0)
		MPI_Alltoallv(s_buf, scount_proc_f, soffset_proc_f, T_plan->MPI_T,
				r_buf, rcount_proc_f, roffset_proc_f, T_plan->MPI_T,
				T_plan->comm);
	else
		MPI_Alltoall(s_buf, scount_proc_f[0], T_plan->MPI_T, r_buf,
				rcount_proc_f[0], T_plan->MPI_T, T_plan->comm);

	comm_time += MPI_Wtime();

	ptr = 0;
#ifdef VERBOSE2
	if (VERBOSE >= 2)
		PCOUT << "MPIAlltoAll:" << std::endl;
	if (VERBOSE >= 2)
		for (int id = 0; id < nprocs; ++id) {
			if (procid == id)
				for (int h = 0; h < howmany; h++)
					for (int i = 0; i < local_n1; i++) {
						std::cout << std::endl;
						for (int j = 0; j < N[0]; j++) {
							std::cout << '\t' << send_recv[ptr] << ","
									<< send_recv[ptr + 1];
							ptr += n_tuples;
						}
					}
			std::cout << '\n';
		}
#endif

	//PCOUT<<" ============================================= "<<std::endl;
	//PCOUT<<" ============== 2nd Local Trnaspose ========== "<<std::endl;
	//PCOUT<<" ============================================= "<<std::endl;
	reshuffle_time -= MPI_Wtime();
	ptr = 0;
	for (int proc = 0; proc < nprocs_0; ++proc)
		for (int h = 0; h < howmany; ++h) {
			//for(int i=local_0_start_proc[proc];i<local_0_start_proc[proc]+local_n0_proc[proc];++i){
			//  memcpy(&data[h*odist+(i*local_n1)*n_tuples],&send_recv[ptr],local_n1*sizeof(T)*n_tuples);
			//  ptr+=n_tuples*local_n1;
			//  //for(int j=0*local_1_start_proc[proc];j<0*local_1_start_proc[proc]+local_n1;++j){
			//  //  memcpy(&data[h*odist+(i*local_n1+j)*n_tuples],&send_recv[ptr],sizeof(T)*n_tuples);
			//  //  //std::cout<<"proc= "<<proc<<" h= "<<h<<" (i,j)=("<<i<<","<<j<<")  data_ptr= "<<h*idist+(i*local_n1+j)*n_tuples<< " ptr= "<<ptr <<" cpy= "<<n_tuples*T_plan->local_n1_proc[0] <<std::endl;
			//  //  ptr+=n_tuples;
			//  //}
			//}
			memcpy(
					&data[h * odist
							+ local_0_start_proc[proc] * local_n1 * n_tuples],
					&send_recv[ptr],
					local_n1 * sizeof(T) * n_tuples * local_n0_proc[proc]);
			ptr += n_tuples * local_n1 * local_n0_proc[proc];
		}

	// Right now the data is in transposed out format.
	// If the user did not want this layout, transpose again.
	if (Flags[1] == 0) {
		for (int h = 0; h < howmany; h++)
			local_transpose(N[0], local_n1, n_tuples, &data[h * odist]);
	}

#ifdef VERBOSE2
	if (VERBOSE >= 2)
		PCOUT << "2nd Transpose" << std::endl;
	if (VERBOSE >= 2)
		for (int id = 0; id < nprocs_1; ++id) {
			if (procid == id)
				for (int h = 0; h < howmany; h++)
					for (int i = 0; i < N[0]; i++) {
						std::cout << std::endl;
						for (int j = 0; j < local_n1; j++) {
							ptr = h * odist + (i * local_n1 + j) * n_tuples;
							std::cout << '\t' << data[ptr] << ","
									<< data[ptr + 1];
						}
					}
			std::cout << '\n';
		}
#endif

	//PCOUT<<"nprocs_0= "<<nprocs_0<<" nprocs_1= "<<nprocs_1<<std::endl;

	reshuffle_time += MPI_Wtime();

#ifdef VERBOSE1
	if (VERBOSE >= 1) {
		PCOUT << "Shuffle Time= " << shuffle_time << std::endl;
		PCOUT << "Alltoall Time= " << comm_time << std::endl;
		PCOUT << "Reshuffle Time= " << reshuffle_time << std::endl;
		PCOUT << "Total Time= " << (shuffle_time + comm_time + reshuffle_time)
				<< std::endl;
	}
#endif
	timings[0] += MPI_Wtime(); //timings[0]+=shuffle_time+comm_time+reshuffle_time;
	timings[1] += shuffle_time;
	timings[2] += comm_time;
	timings[3] += reshuffle_time;
	return;
}  // end fast_transpose_v2_h

template<typename T>
void fast_transpose_v3_h(T_Plan<T>* __restrict T_plan, T * __restrict data, double * __restrict timings, int kway,
		unsigned flags, int howmany, int tag, T* __restrict data_out) {

	std::bitset < 8 > Flags(flags); // 1 Transposed in, 2 Transposed out
	if (Flags[1] == 1 && Flags[0] == 0 && T_plan->nprocs == 1) { // If Flags==Transposed_Out return
		return;
	}
	if (howmany == 1) {
		fast_transpose_v3(T_plan, data, timings, kway, flags, howmany, tag);
		return;
	}
	if (Flags[0] == 1) { // If Flags==Transposed_In This function can not handle it, call other versions
		transpose_v7(T_plan, (T*) data, timings, kway, flags, howmany);
		return;
	}
  if(Flags[1] == 0 && data_out !=NULL) {
    std::cout << "operation not supported\n";
    return;
  }
	timings[0] -= MPI_Wtime();
	int nprocs, procid;
	int nprocs_0, nprocs_1;
	nprocs = T_plan->nprocs;
	procid = T_plan->procid;
	nprocs_0 = T_plan->nprocs_0;
	nprocs_1 = T_plan->nprocs_1;
	ptrdiff_t *N = T_plan->N;
	T * send_recv = T_plan->buffer;
	T * buffer_2 = T_plan->buffer_2;
	ptrdiff_t local_n0 = T_plan->local_n0;
	ptrdiff_t local_n1 = T_plan->local_n1;
	ptrdiff_t n_tuples = T_plan->n_tuples;

	int idist = N[1] * local_n0 * n_tuples;
	int odist = N[0] * local_n1 * n_tuples;

	double comm_time = 0, shuffle_time = 0, reshuffle_time = 0, total_time = 0;

	int ptr = 0;
#ifdef VERBOSE2
	if (VERBOSE >= 2)
		PCOUT << "INPUT:" << std::endl;
	if (VERBOSE >= 2)
		for (int h = 0; h < howmany; h++)
			for (int id = 0; id < nprocs; ++id) {
				if (procid == id)
					for (int i = 0; i < local_n0; i++) {
						std::cout << std::endl;
						for (int j = 0; j < N[1]; j++) {
							ptr = h * idist + (i * N[1] + j) * n_tuples;
							std::cout << '\t' << data[ptr] << ","
									<< data[ptr + 1];
						}
					}
				std::cout << '\n';
			}
#endif

	//PCOUT<<" ============================================= "<<std::endl;
	//PCOUT<<" ==============   Local Transpose============= "<<std::endl;
	//PCOUT<<" ============================================= "<<std::endl;
	ptr = 0;
	ptrdiff_t *local_n1_proc = &T_plan->local_n1_proc[0];
	ptrdiff_t *local_n0_proc = &T_plan->local_n0_proc[0];
	ptrdiff_t *local_0_start_proc = T_plan->local_0_start_proc;
	ptrdiff_t *local_1_start_proc = T_plan->local_1_start_proc;
	shuffle_time -= MPI_Wtime();
	if (nprocs == 1 && Flags[0] == 1 && Flags[1] == 1) {
		for (int h = 0; h < howmany; h++)
			local_transpose(local_n1, N[0], n_tuples, &data[h * idist]);
	}
	if (nprocs == 1 && Flags[0] == 0 && Flags[1] == 0) {
		for (int h = 0; h < howmany; h++)
			local_transpose(N[0], N[1], n_tuples, &data[h * idist]);
	}
	if (nprocs == 1) { // Transpose is done!
		shuffle_time += MPI_Wtime();
		timings[0] += MPI_Wtime();
		timings[1] += shuffle_time;
		timings[2] += 0;
		timings[3] += 0;
		return;
	}

	// The idea is this: If The Output is to be transposed, then only one buffer is needed. The algorithm would be:
	// data --T-> send_recv   ...  send_recv --alltoall--> data
	// Otherwise buffer2 needs to be used as well:
	// data --T-> buffer_2    ...  buffer_2 --alltoall--> send_recv ... send_recv -T-> data
	ptr = 0;

	for (int proc = 0; proc < nprocs_1; ++proc)
		for (int h = 0; h < howmany; ++h) {
			for (int i = 0; i < local_n0; ++i) {
				//for(int j=local_1_start_proc[proc];j<local_1_start_proc[proc]+local_n1_proc[proc];++j){
				//  memcpy(&buffer_2[ptr],&data[h*idist+(i*N[1]+j)*n_tuples],sizeof(T)*n_tuples);
				//  //std::cout<<"proc= "<<proc<<" h= "<<h<<" (i,j)=("<<i<<","<<j<<")  data_ptr= "<<h*idist+(i*local_n1+j)*n_tuples<< " ptr= "<<ptr <<" cpy= "<<n_tuples*T_plan->local_n1_proc[0] <<std::endl;
				//  ptr+=n_tuples;
				//}
				memcpy(&buffer_2[ptr],
						&data[h * idist
								+ (i * N[1] + local_1_start_proc[proc])
										* n_tuples],
						sizeof(T) * n_tuples * local_n1_proc[proc]);
				ptr += n_tuples * local_n1_proc[proc];
			}
		}

	shuffle_time += MPI_Wtime();
	ptr = 0;
#ifdef VERBOSE2
	if (VERBOSE >= 2)
		PCOUT << "Local Transpose:" << std::endl;
	if (VERBOSE >= 2)
		for (int id = 0; id < nprocs; ++id) {
			for (int h = 0; h < howmany; h++)
				if (procid == id)
					for (int i = 0; i < N[1]; i++) {
						std::cout << std::endl;
						for (int j = 0; j < local_n0; j++) {
							std::cout << '\t' << buffer_2[ptr] << ","
									<< buffer_2[ptr + 1];
							ptr += n_tuples;
						}
					}
			std::cout << '\n';
		}
#endif

	//PCOUT<<" ============================================= "<<std::endl;
	//PCOUT<<" ==============   MPIALLTOALL  =============== "<<std::endl;
	//PCOUT<<" ============================================= "<<std::endl;

	int* scount_proc_f = T_plan->scount_proc_f;
	int* rcount_proc_f = T_plan->rcount_proc_f;
	int* soffset_proc_f = T_plan->soffset_proc_f;
	int* roffset_proc_f = T_plan->roffset_proc_f;


	//PCOUT<<"nprocs_0= "<<nprocs_0<<" nprocs_1= "<<nprocs_1<<std::endl;
	comm_time -= MPI_Wtime();

	int soffset = 0, roffset = 0;
	MPI_Status ierr;
	int counter = 1;

	T *s_buf, *r_buf;
	s_buf = buffer_2;
	r_buf = send_recv;

	if (T_plan->kway_async)
		par::Mpi_Alltoallv_dense<T, true>(s_buf, scount_proc_f, soffset_proc_f,
				r_buf, rcount_proc_f, roffset_proc_f, T_plan->comm, kway);
	else
		par::Mpi_Alltoallv_dense<T, false>(s_buf, scount_proc_f, soffset_proc_f,
				r_buf, rcount_proc_f, roffset_proc_f, T_plan->comm, kway);

	comm_time += MPI_Wtime();

	ptr = 0;
#ifdef VERBOSE2
	if (VERBOSE >= 2)
		PCOUT << "MPIAlltoAll:" << std::endl;
	if (VERBOSE >= 2)
		for (int id = 0; id < nprocs; ++id) {
			if (procid == id)
				for (int h = 0; h < howmany; h++)
					for (int i = 0; i < local_n1; i++) {
						std::cout << std::endl;
						for (int j = 0; j < N[0]; j++) {
							std::cout << '\t' << send_recv[ptr] << ","
									<< send_recv[ptr + 1];
							ptr += n_tuples;
						}
					}
			std::cout << '\n';
		}
#endif

	//PCOUT<<" ============================================= "<<std::endl;
	//PCOUT<<" ============== 2nd Local Trnaspose ========== "<<std::endl;
	//PCOUT<<" ============================================= "<<std::endl;
	reshuffle_time -= MPI_Wtime();
	ptr = 0;
	for (int proc = 0; proc < nprocs_0; ++proc)
		for (int h = 0; h < howmany; ++h) {
			//for(int i=local_0_start_proc[proc];i<local_0_start_proc[proc]+local_n0_proc[proc];++i){
			//  memcpy(&data[h*odist+(i*local_n1)*n_tuples],&send_recv[ptr],local_n1*sizeof(T)*n_tuples);
			//  ptr+=n_tuples*local_n1;
			//  //for(int j=0*local_1_start_proc[proc];j<0*local_1_start_proc[proc]+local_n1;++j){
			//  //  memcpy(&data[h*odist+(i*local_n1+j)*n_tuples],&send_recv[ptr],sizeof(T)*n_tuples);
			//  //  //std::cout<<"proc= "<<proc<<" h= "<<h<<" (i,j)=("<<i<<","<<j<<")  data_ptr= "<<h*idist+(i*local_n1+j)*n_tuples<< " ptr= "<<ptr <<" cpy= "<<n_tuples*T_plan->local_n1_proc[0] <<std::endl;
			//  //  ptr+=n_tuples;
			//  //}
			//}
			memcpy(
					&data[h * odist
							+ local_0_start_proc[proc] * local_n1 * n_tuples],
					&send_recv[ptr],
					local_n1 * sizeof(T) * n_tuples * local_n0_proc[proc]);
			ptr += n_tuples * local_n1 * local_n0_proc[proc];
		}

	// Right now the data is in transposed out format.
	// If the user did not want this layout, transpose again.
	if (Flags[1] == 0) {
		for (int h = 0; h < howmany; h++)
			local_transpose(N[0], local_n1, n_tuples, &data[h * odist]);
	}

#ifdef VERBOSE2
	if (VERBOSE >= 2)
		PCOUT << "2nd Transpose" << std::endl;
	if (VERBOSE >= 2)
		for (int id = 0; id < nprocs_1; ++id) {
			if (procid == id)
				for (int h = 0; h < howmany; h++)
					for (int i = 0; i < N[0]; i++) {
						std::cout << std::endl;
						for (int j = 0; j < local_n1; j++) {
							ptr = h * odist + (i * local_n1 + j) * n_tuples;
							std::cout << '\t' << data[ptr] << ","
									<< data[ptr + 1];
						}
					}
			std::cout << '\n';
		}
#endif

	//PCOUT<<"nprocs_0= "<<nprocs_0<<" nprocs_1= "<<nprocs_1<<std::endl;

	reshuffle_time += MPI_Wtime();

#ifdef VERBOSE1
	if (VERBOSE >= 1) {
		PCOUT << "Shuffle Time= " << shuffle_time << std::endl;
		PCOUT << "Alltoall Time= " << comm_time << std::endl;
		PCOUT << "Reshuffle Time= " << reshuffle_time << std::endl;
		PCOUT << "Total Time= " << (shuffle_time + comm_time + reshuffle_time)
				<< std::endl;
	}
#endif
	timings[0] += MPI_Wtime(); //timings[0]+=shuffle_time+comm_time+reshuffle_time;
	timings[1] += shuffle_time;
	timings[2] += comm_time;
	timings[3] += reshuffle_time;
	return;
}  // end fast_transpose_v3_h

template<typename T>
void transpose_v5(T_Plan<T>* __restrict T_plan, T * __restrict data, double * __restrict timings, unsigned flags,
		int howmany, int tag, T* __restrict data_out) {

	std::bitset < 8 > Flags(flags); // 1 Transposed in, 2 Transposed out
	if (Flags[1] == 1 && Flags[0] == 0 && T_plan->nprocs == 1) { // If Flags==Transposed_Out return
		return;
	}
  if(Flags[1] == 0 && data_out !=NULL) {
    std::cout << "operation not supported\n";
    return;
  }
	timings[0] -= MPI_Wtime();
	int nprocs, procid;
	int nprocs_0, nprocs_1;
	nprocs = T_plan->nprocs;
	procid = T_plan->procid;
	nprocs_0 = T_plan->nprocs_0;
	nprocs_1 = T_plan->nprocs_1;
	ptrdiff_t *N = T_plan->N;
	T * send_recv = T_plan->buffer;
	ptrdiff_t local_n0 = T_plan->local_n0;
	ptrdiff_t local_n1 = T_plan->local_n1;
	ptrdiff_t n_tuples = T_plan->n_tuples;

	int idist = N[1] * local_n0 * n_tuples;
	int odist = N[0] * local_n1 * n_tuples;

	double comm_time = 0, shuffle_time = 0, reshuffle_time = 0, total_time = 0;

#ifdef VERBOSE2
	if (VERBOSE >= 2)
		PCOUT << "INPUT:" << std::endl;
	if (VERBOSE >= 2)
		for (int h = 0; h < howmany; h++)
			for (int id = 0; id < nprocs; ++id) {
				if (procid == id)
					for (int i = 0; i < local_n0; i++) {
						std::cout << std::endl;
						for (int j = 0; j < N[1]; j++) {
							std::cout << '\t'
									<< data[h * idist
											+ (i * N[1] + j) * n_tuples];
						}
					}
				std::cout << '\n';
			}
#endif

	//PCOUT<<" ============================================= "<<std::endl;
	//PCOUT<<" ==============   Local Transpose============= "<<std::endl;
	//PCOUT<<" ============================================= "<<std::endl;
	int ptr = 0;
	shuffle_time -= MPI_Wtime();

	if (Flags[0] == 0) {
		if (howmany == 1)
			local_transpose(local_n0, N[1], n_tuples, data);
		else {
			for (int i = 0; i < howmany; i++)
				local_transpose(local_n0, N[1], n_tuples, &data[i * idist]);
		}
	}
	shuffle_time += MPI_Wtime();
#ifdef VERBOSE2
	if (VERBOSE >= 2)
		PCOUT << "Local Transpose:" << std::endl;
	if (VERBOSE >= 2)
		for (int h = 0; h < howmany; h++)
			for (int id = 0; id < nprocs; ++id) {
				if (procid == id)
					for (int i = 0; i < N[1]; i++) {
						std::cout << std::endl;
						for (int j = 0; j < local_n0; j++) {
							std::cout << '\t'
									<< data[h * idist
											+ (i * local_n0 + j) * n_tuples];
						}
					}
				std::cout << '\n';
			}
#endif
	if (nprocs == 1 && Flags[0] == 1 && Flags[1] == 1) {
		for (int h = 0; h < howmany; h++)
			local_transpose(local_n1, N[0], n_tuples, &data[h * idist]);
	}
	if (nprocs == 1) { // Transpose is done!
		timings[0] += MPI_Wtime();
		timings[1] += shuffle_time;
		timings[2] += 0;
		timings[3] += 0;
		return;
	}

	//PCOUT<<" ============================================= "<<std::endl;
	//PCOUT<<" ==============   MPIALLTOALL  =============== "<<std::endl;
	//PCOUT<<" ============================================= "<<std::endl;

	int* scount_proc = T_plan->scount_proc;
	int* rcount_proc = T_plan->rcount_proc;
	int* soffset_proc = T_plan->soffset_proc;
	int* roffset_proc = T_plan->roffset_proc;

	MPI_Datatype *stype = T_plan->stype;
	MPI_Datatype *rtype = T_plan->rtype;

	comm_time -= MPI_Wtime();

	int soffset = 0, roffset = 0;
	MPI_Status ierr;
	MPI_Request * s_request = new MPI_Request[nprocs];
	MPI_Request * request = new MPI_Request[nprocs];

	// SEND
	for (int proc = 0; proc < nprocs; ++proc) {
		if (proc != procid) {
			soffset = soffset_proc[proc];
			MPI_Isend(&data[soffset], 1, stype[proc], proc, tag, T_plan->comm,
					&s_request[proc]);
		}
	}
	// RECV
	for (int proc = 0; proc < nprocs; ++proc) {
		if (proc != procid) {
			roffset = roffset_proc[proc];
			MPI_Irecv(&send_recv[roffset], 1, rtype[proc], proc, tag,
					T_plan->comm, &request[proc]);
		} else {
			soffset = soffset_proc[proc]; //aoffset_proc[proc];//proc*count_proc[proc];
			roffset = roffset_proc[proc];
			for (int h = 0; h < howmany; h++)
				memcpy(&send_recv[h * odist + roffset],
						&data[h * idist + soffset],
						sizeof(T) * scount_proc[proc]);
		}
	}

	for (int proc = 0; proc < nprocs; ++proc) {
		if (proc != procid) {
			MPI_Wait(&request[proc], &ierr);
			MPI_Wait(&s_request[proc], &ierr);
		}
	}
	comm_time += MPI_Wtime();

#ifdef VERBOSE2
	if (VERBOSE >= 2)
		PCOUT << "MPIAlltoAll:" << std::endl;
	if (VERBOSE >= 2)
		for (int h = 0; h < howmany; h++)
			for (int id = 0; id < nprocs; ++id) {
				if (procid == id)
					for (int i = 0; i < local_n1; i++) {
						std::cout << std::endl;
						for (int j = 0; j < N[0]; j++) {
							std::cout << '\t'
									<< send_recv[h * odist
											+ (i * N[0] + j) * n_tuples]; //<<","<<send_recv[(i*N[0]+j)*n_tuples+1];
						}
					}
				std::cout << '\n';
			}
#endif

	//PCOUT<<" ============================================= "<<std::endl;
	//PCOUT<<" ============== 2nd Local Trnaspose ========== "<<std::endl;
	//PCOUT<<" ============================================= "<<std::endl;
	reshuffle_time -= MPI_Wtime();
	ptr = 0;

	// int first_local_n0, last_local_n0;
	// first_local_n0=local_n0_proc[0]; last_local_n0=local_n0_proc[nprocs_1-1];

	//local_transpose(nprocs_1,local_n1,n_tuples*local_n0,send_recv,data );

	int last_ntuples = 0, first_ntuples = T_plan->local_n0_proc[0] * n_tuples;
	if (local_n1 != 0)
		last_ntuples = T_plan->last_recv_count / ((int) local_n1);
	for (int i = 0; i < howmany; i++) {
		if (local_n1 == 1)
			memcpy(&data[i * odist], &send_recv[i * odist],
					T_plan->alloc_local / howmany); // you are done, no additional transpose is needed.
		else if (last_ntuples != first_ntuples) {
			local_transpose((nprocs_0 - 1), local_n1, first_ntuples,
					&send_recv[i * odist]);
			local_transpose(2, local_n1, (nprocs_0 - 1) * first_ntuples,
					last_ntuples, &send_recv[i * odist], &data[i * odist]);
		} else if (last_ntuples == first_ntuples) {
			//local_transpose(nprocs_0,local_n1,n_tuples*local_n0,send_recv );
			local_transpose(nprocs_0, local_n1, first_ntuples,
					&send_recv[i * odist], &data[i * odist]);
		}
	}

#ifdef VERBOSE2
	if (VERBOSE >= 2)
		PCOUT << "2nd Transpose" << std::endl;
	if (VERBOSE >= 2)
		for (int h = 0; h < howmany; h++)
			for (int id = 0; id < nprocs_1; ++id) {
				if (procid == id)
					for (int i = 0; i < local_n1; i++) {
						std::cout << std::endl;
						for (int j = 0; j < N[0]; j++) {
							std::cout << '\t'
									<< data[h * odist
											+ (i * N[0] + j) * n_tuples];
						}
					}
				std::cout << '\n';
			}
#endif

	if (Flags[1] == 1) { // Transpose output
		if (howmany == 1)
			local_transpose(local_n1, N[0], n_tuples, data);
		else {
			for (int h = 0; h < howmany; h++)
				local_transpose(local_n1, N[0], n_tuples, &data[h * odist]);
		}
	}
	reshuffle_time += MPI_Wtime();
	delete[] request;
	delete[] s_request;

#ifdef VERBOSE2
	if (VERBOSE >= 2)
		PCOUT << "Transposed Out" << std::endl;
	if (VERBOSE >= 2)
		for (int h = 0; h < howmany; h++)
			for (int id = 0; id < nprocs_1; ++id) {
				if (procid == id)
					for (int i = 0; i < N[0]; i++) {
						std::cout << std::endl;
						for (int j = 0; j < local_n1; j++) {
							std::cout << '\t'
									<< data[h * odist
											+ (i * local_n1 + j) * n_tuples];
						}
					}
				std::cout << '\n';
			}
#endif

#ifdef VERBOSE1
	if (VERBOSE >= 1) {
		PCOUT << "Shuffle Time= " << shuffle_time << std::endl;
		PCOUT << "Alltoall Time= " << comm_time << std::endl;
		PCOUT << "Reshuffle Time= " << reshuffle_time << std::endl;
		PCOUT << "Total Time= " << (shuffle_time + comm_time + reshuffle_time)
				<< std::endl;
	}
#endif
	timings[0] += MPI_Wtime(); //timings[0]+=shuffle_time+comm_time+reshuffle_time;
	timings[1] += shuffle_time;
	timings[2] += comm_time;
	timings[3] += reshuffle_time;
	return;
} // end fast_transpose_v5

template<typename T>
void transpose_v6(T_Plan<T>* __restrict T_plan, T * __restrict data, double * __restrict timings, unsigned flags,
		int howmany, T* __restrict data_out) {

	std::bitset < 8 > Flags(flags); // 1 Transposed in, 2 Transposed out
	if (Flags[1] == 1 && Flags[0] == 0 && T_plan->nprocs == 1) { // If Flags==Transposed_Out return
		return;
	}
  if(Flags[1] == 0 && data_out !=NULL) {
    std::cout << "operation not supported\n";
    return;
  }
	timings[0] -= MPI_Wtime();
	int nprocs, procid;
	int nprocs_0, nprocs_1;
	nprocs = T_plan->nprocs;
	procid = T_plan->procid;
	nprocs_0 = T_plan->nprocs_0;
	nprocs_1 = T_plan->nprocs_1;
	ptrdiff_t *N = T_plan->N;
	T * send_recv = T_plan->buffer;
	ptrdiff_t local_n0 = T_plan->local_n0;
	ptrdiff_t local_n1 = T_plan->local_n1;
	ptrdiff_t n_tuples = T_plan->n_tuples;
	int idist = N[1] * local_n0 * n_tuples;
	int odist = N[0] * local_n1 * n_tuples;

	double comm_time = 0, shuffle_time = 0, reshuffle_time = 0, total_time = 0;

#ifdef VERBOSE2
	if (VERBOSE >= 2)
		PCOUT << "INPUT:" << std::endl;
	if (VERBOSE >= 2)
		for (int h = 0; h < howmany; h++)
			for (int id = 0; id < nprocs; ++id) {
				if (procid == id)
					for (int i = 0; i < local_n0; i++) {
						std::cout << std::endl;
						for (int j = 0; j < N[1]; j++) {
							std::cout << '\t'
									<< data[h * idist
											+ (i * N[1] + j) * n_tuples];
						}
					}
				std::cout << '\n';
			}
#endif

	//PCOUT<<" ============================================= "<<std::endl;
	//PCOUT<<" ==============   Local Transpose============= "<<std::endl;
	//PCOUT<<" ============================================= "<<std::endl;
	int ptr = 0;
	shuffle_time -= MPI_Wtime();

	if (Flags[0] == 0) {
		for (int i = 0; i < howmany; i++)
			local_transpose(local_n0, N[1], n_tuples,
					&data[i * local_n0 * N[1] * n_tuples]);
	}

	shuffle_time += MPI_Wtime();
#ifdef VERBOSE2
	if (VERBOSE >= 2)
		PCOUT << "Local Transpose:" << std::endl;
	if (VERBOSE >= 2)
		for (int h = 0; h < howmany; h++)
			for (int id = 0; id < nprocs; ++id) {
				if (procid == id)
					for (int i = 0; i < N[1]; i++) {
						std::cout << std::endl;
						for (int j = 0; j < local_n0; j++) {
							std::cout << '\t'
									<< data[idist * h
											+ (i * local_n0 + j) * n_tuples]
									<< ","
									<< data[idist * h
											+ (i * local_n0 + j) * n_tuples + 1];
						}
					}
				std::cout << '\n';
			}
#endif

	if (nprocs == 1 && Flags[0] == 1 && Flags[1] == 1) {
		local_transpose(local_n1, N[0], n_tuples, data);
	}
	if (nprocs == 1) { // Transpose is done!
		timings[0] += MPI_Wtime();
		timings[1] += shuffle_time;
		timings[2] += 0;
		timings[3] += 0;
		return;
	}

	//PCOUT<<" ============================================= "<<std::endl;
	//PCOUT<<" ==============   MPIALLTOALL  =============== "<<std::endl;
	//PCOUT<<" ============================================= "<<std::endl;

	int* scount_proc = T_plan->scount_proc;
	int* rcount_proc = T_plan->rcount_proc;
	int* soffset_proc = T_plan->soffset_proc;
	int* roffset_proc = T_plan->roffset_proc;
	MPI_Datatype *stype = T_plan->stype;
	MPI_Datatype *rtype = T_plan->rtype;


	comm_time -= MPI_Wtime();

	MPI_Status status;

	if (howmany > 1) {
		MPI_Alltoallw(data, T_plan->scount_proc_w, T_plan->soffset_proc_w,
				stype, send_recv, T_plan->rcount_proc_w, T_plan->roffset_proc_w,
				rtype, T_plan->comm);
	} else if (T_plan->is_evenly_distributed == 0)
		MPI_Alltoallv(data, scount_proc, soffset_proc, T_plan->MPI_T, send_recv,
				rcount_proc, roffset_proc, T_plan->MPI_T, T_plan->comm);
	else
		MPI_Alltoall(data, scount_proc[0], T_plan->MPI_T, send_recv,
				rcount_proc[0], T_plan->MPI_T, T_plan->comm);

	comm_time += MPI_Wtime();

#ifdef VERBOSE2
	if (VERBOSE >= 2)
		PCOUT << "MPIAlltoAll:" << std::endl;
	if (VERBOSE >= 2)
		for (int h = 0; h < howmany; h++)
			for (int id = 0; id < nprocs; ++id) {
				if (procid == id)
					for (int i = 0; i < local_n1; i++) {
						std::cout << std::endl;
						for (int j = 0; j < N[0]; j++) {
							std::cout << '\t'
									<< send_recv[odist * h
											+ (i * N[0] + j) * n_tuples]; //<<","<<send_recv[(i*N[0]+j)*n_tuples+1];
						}
					}
				std::cout << '\n';
			}
#endif

	//PCOUT<<" ============================================= "<<std::endl;
	//PCOUT<<" ============== 2nd Local Trnaspose ========== "<<std::endl;
	//PCOUT<<" ============================================= "<<std::endl;

	reshuffle_time -= MPI_Wtime();
	ptr = 0;

	// int first_local_n0, last_local_n0;
	// first_local_n0=local_n0_proc[0]; last_local_n0=local_n0_proc[nprocs_1-1];

	//local_transpose(nprocs_1,local_n1,n_tuples*local_n0,send_recv,data );

	int last_ntuples = 0, first_ntuples = T_plan->local_n0_proc[0] * n_tuples;
	if (local_n1 != 0)
		last_ntuples = T_plan->last_recv_count / ((int) local_n1);
	for (int i = 0; i < howmany; i++) {
		if (local_n1 == 1)
			memcpy(&data[i * odist], &send_recv[i * odist],
					T_plan->alloc_local / howmany); // you are done, no additional transpose is needed.
		else if (last_ntuples != first_ntuples) {
			local_transpose((nprocs_0 - 1), local_n1, first_ntuples,
					&send_recv[i * odist]);
			local_transpose(2, local_n1, (nprocs_0 - 1) * first_ntuples,
					last_ntuples, &send_recv[i * odist], &data[i * odist]);
		} else if (last_ntuples == first_ntuples) {
			//local_transpose(nprocs_0,local_n1,n_tuples*local_n0,send_recv );
			local_transpose(nprocs_0, local_n1, first_ntuples,
					&send_recv[i * odist], &data[i * odist]);
		}
	}

#ifdef VERBOSE2
	if (VERBOSE >= 2)
		PCOUT << "2nd Transpose" << std::endl;
	if (VERBOSE >= 2)
		for (int h = 0; h < howmany; h++)
			for (int id = 0; id < nprocs_1; ++id) {
				if (procid == id)
					for (int i = 0; i < local_n1; i++) {
						std::cout << std::endl;
						for (int j = 0; j < N[0]; j++) {
							std::cout << '\t'
									<< data[odist * h
											+ (i * N[0] + j) * n_tuples];
						}
					}
				std::cout << '\n';
			}
#endif

	if (Flags[1] == 1) { // Transpose output
		for (int h = 0; h < howmany; h++)
			local_transpose(local_n1, N[0], n_tuples, &data[odist * h]);

#ifdef VERBOSE2
		if (VERBOSE >= 2)
			PCOUT << "Transposed Out" << std::endl;
		if (VERBOSE >= 2)
			for (int h = 0; h < howmany; h++)
				for (int id = 0; id < nprocs_1; ++id) {
					if (procid == id)
						for (int i = 0; i < N[0]; i++) {
							std::cout << std::endl;
							for (int j = 0; j < local_n1; j++) {
								std::cout << '\t'
										<< data[odist * h
												+ (i * local_n1 + j) * n_tuples];
							}
						}
					std::cout << '\n';
				}
#endif

	}
	reshuffle_time += MPI_Wtime();

#ifdef VERBOSE1
	if (VERBOSE >= 1) {
		PCOUT << "Shuffle Time= " << shuffle_time << std::endl;
		PCOUT << "Alltoall Time= " << comm_time << std::endl;
		PCOUT << "Reshuffle Time= " << reshuffle_time << std::endl;
		PCOUT << "Total Time= " << (shuffle_time + comm_time + reshuffle_time)
				<< std::endl;
	}
#endif
	timings[0] += MPI_Wtime(); //timings[0]+=shuffle_time+comm_time+reshuffle_time;
	timings[1] += shuffle_time;
	timings[2] += comm_time;
	timings[3] += reshuffle_time;
	return;
} // end fast_transpose_v6

template<typename T>
void transpose_v7(T_Plan<T>* __restrict T_plan, T * __restrict data, double * __restrict timings, int kway,
		unsigned flags, int howmany, T* __restrict data_out) {
	std::bitset < 8 > Flags(flags); // 1 Transposed in, 2 Transposed out
	if (Flags[1] == 1 && Flags[0] == 0 && T_plan->nprocs == 1) { // If Flags==Transposed_Out return
		return;
	}
  if(Flags[1] == 0 && data_out !=NULL) {
    std::cout << "operation not supported\n";
    return;
  }

	timings[0] -= MPI_Wtime();
	int nprocs, procid;
	int nprocs_0, nprocs_1;
	nprocs = T_plan->nprocs;
	procid = T_plan->procid;
	nprocs_0 = T_plan->nprocs_0;
	nprocs_1 = T_plan->nprocs_1;
	ptrdiff_t *N = T_plan->N;
	T * send_recv = T_plan->buffer;
	ptrdiff_t local_n0 = T_plan->local_n0;
	ptrdiff_t local_n1 = T_plan->local_n1;
	ptrdiff_t n_tuples = T_plan->n_tuples;
	int idist = N[1] * local_n0 * n_tuples;
	int odist = N[0] * local_n1 * n_tuples;

	double comm_time = 0, shuffle_time = 0, reshuffle_time = 0, total_time = 0;

#ifdef VERBOSE2
	if (VERBOSE >= 2)
		PCOUT << "INPUT:" << std::endl;
	if (VERBOSE >= 2)
		for (int id = 0; id < nprocs; ++id) {
			if (procid == id)
				for (int i = 0; i < local_n0; i++) {
					std::cout << std::endl;
					for (int j = 0; j < N[1]; j++) {
						std::cout << '\t' << data[(i * N[1] + j) * n_tuples];
					}
				}
			std::cout << '\n';
		}
#endif

	//PCOUT<<" ============================================= "<<std::endl;
	//PCOUT<<" ==============   Local Transpose============= "<<std::endl;
	//PCOUT<<" ============================================= "<<std::endl;
	int ptr = 0;
	shuffle_time -= MPI_Wtime();

	if (Flags[0] == 0) {
		for (int i = 0; i < howmany; i++)
			local_transpose(local_n0, N[1], n_tuples,
					&data[i * local_n0 * N[1] * n_tuples]);
	}

	shuffle_time += MPI_Wtime();
#ifdef VERBOSE2
	if (VERBOSE >= 2)
		PCOUT << "Local Transpose:" << std::endl;
	if (VERBOSE >= 2)
		for (int id = 0; id < nprocs; ++id) {
			if (procid == id)
				for (int i = 0; i < N[1]; i++) {
					std::cout << std::endl;
					for (int j = 0; j < local_n0; j++) {
						std::cout << '\t' << data[(i * local_n0 + j) * n_tuples]
								<< ","
								<< data[(i * local_n0 + j) * n_tuples + 1];
					}
				}
			std::cout << '\n';
		}
#endif
	if (nprocs == 1 && Flags[0] == 1 && Flags[1] == 1) {
		local_transpose(local_n1, N[0], n_tuples, data);
	}

	if (nprocs == 1) { // Transpose is done!
		timings[0] += MPI_Wtime();
		timings[1] += shuffle_time;
		timings[2] += 0;
		timings[3] += 0;
		return;
	}

	//PCOUT<<" ============================================= "<<std::endl;
	//PCOUT<<" ==============   MPIALLTOALL  =============== "<<std::endl;
	//PCOUT<<" ============================================= "<<std::endl;

	int* scount_proc = T_plan->scount_proc;
	int* rcount_proc = T_plan->rcount_proc;
	int* soffset_proc = T_plan->soffset_proc;
	int* roffset_proc = T_plan->roffset_proc;

	comm_time -= MPI_Wtime();

	if (T_plan->kway_async)
		par::Mpi_Alltoallv_dense<T, true>(data, scount_proc, soffset_proc,
				send_recv, rcount_proc, roffset_proc, T_plan->comm, kway);
	else
		par::Mpi_Alltoallv_dense<T, false>(data, scount_proc, soffset_proc,
				send_recv, rcount_proc, roffset_proc, T_plan->comm, kway);
	comm_time += MPI_Wtime();

#ifdef VERBOSE2
	if (VERBOSE >= 2)
		PCOUT << "MPIAlltoAll:" << std::endl;
	if (VERBOSE >= 2)
		for (int id = 0; id < nprocs; ++id) {
			if (procid == id)
				for (int i = 0; i < local_n1; i++) {
					std::cout << std::endl;
					for (int j = 0; j < N[0]; j++) {
						std::cout << '\t'
								<< send_recv[(i * N[0] + j) * n_tuples]; //<<","<<send_recv[(i*N[0]+j)*n_tuples+1];
					}
				}
			std::cout << '\n';
		}
#endif

	//PCOUT<<" ============================================= "<<std::endl;
	//PCOUT<<" ============== 2nd Local Trnaspose ========== "<<std::endl;
	//PCOUT<<" ============================================= "<<std::endl;

	reshuffle_time -= MPI_Wtime();
	ptr = 0;

	// int first_local_n0, last_local_n0;
	// first_local_n0=local_n0_proc[0]; last_local_n0=local_n0_proc[nprocs_1-1];

	//local_transpose(nprocs_1,local_n1,n_tuples*local_n0,send_recv,data );

	int last_ntuples = 0, first_ntuples = T_plan->local_n0_proc[0] * n_tuples;
	if (local_n1 != 0)
		last_ntuples = T_plan->last_recv_count / ((int) local_n1);

	for (int i = 0; i < howmany; i++) {
		if (local_n1 == 1)
			memcpy(&data[i * odist], &send_recv[i * odist],
					T_plan->alloc_local / howmany); // you are done, no additional transpose is needed.
		else if (last_ntuples != first_ntuples) {
			local_transpose((nprocs_0 - 1), local_n1, first_ntuples,
					&send_recv[i * odist]);
			local_transpose(2, local_n1, (nprocs_0 - 1) * first_ntuples,
					last_ntuples, &send_recv[i * odist], &data[i * odist]);
		} else if (last_ntuples == first_ntuples) {
			//local_transpose(nprocs_0,local_n1,n_tuples*local_n0,send_recv );
			local_transpose(nprocs_0, local_n1, first_ntuples,
					&send_recv[i * odist], &data[i * odist]);
		}
	}

#ifdef VERBOSE2
	if (VERBOSE >= 2)
		PCOUT << "2nd Transpose" << std::endl;
	if (VERBOSE >= 2)
		for (int id = 0; id < nprocs_1; ++id) {
			if (procid == id)
				for (int i = 0; i < local_n1; i++) {
					std::cout << std::endl;
					for (int j = 0; j < N[0]; j++) {
						std::cout << '\t' << data[(i * N[0] + j) * n_tuples];
					}
				}
			std::cout << '\n';
		}
#endif

	if (Flags[1] == 1) { // Transpose output
		local_transpose(local_n1, N[0], n_tuples, data);
	}
	reshuffle_time += MPI_Wtime();
#ifdef VERBOSE2
	if (VERBOSE >= 1) {
		PCOUT << "Shuffle Time= " << shuffle_time << std::endl;
		PCOUT << "Alltoall Time= " << comm_time << std::endl;
		PCOUT << "Reshuffle Time= " << reshuffle_time << std::endl;
		PCOUT << "Total Time= " << (shuffle_time + comm_time + reshuffle_time)
				<< std::endl;
	}
#endif
	timings[0] += MPI_Wtime(); //timings[0]+=shuffle_time+comm_time+reshuffle_time;
	timings[1] += shuffle_time;
	timings[2] += comm_time;
	timings[3] += reshuffle_time;
	return;
} // end fast_transpose_v7
template<typename T>
void transpose_v8(T_Plan<T>* __restrict T_plan, T * __restrict data, double * __restrict timings, unsigned flags,
		int howmany, int tag, T* __restrict data_out) {

	std::bitset < 8 > Flags(flags); // 1 Transposed in, 2 Transposed out
	if (Flags[1] == 1 && Flags[0] == 0 && T_plan->nprocs == 1) { // If Flags==Transposed_Out return
		return;
	}
  if(Flags[1] == 0 && data_out !=NULL) {
    std::cout << "operation not supported\n";
    return;
  }
	timings[0] -= MPI_Wtime();
	int nprocs, procid;
	int nprocs_0, nprocs_1;
	nprocs = T_plan->nprocs;
	procid = T_plan->procid;

	nprocs_0 = T_plan->nprocs_0;
	nprocs_1 = T_plan->nprocs_1;
	ptrdiff_t *N = T_plan->N;
	T * send_recv = T_plan->buffer;
	ptrdiff_t local_n0 = T_plan->local_n0;
	ptrdiff_t local_n1 = T_plan->local_n1;
	ptrdiff_t n_tuples = T_plan->n_tuples;

	int idist = N[1] * local_n0 * n_tuples;
	int odist = N[0] * local_n1 * n_tuples;

	double comm_time = 0, shuffle_time = 0, reshuffle_time = 0, total_time = 0;

#ifdef VERBOSE2
	if (VERBOSE >= 2)
		PCOUT << "INPUT:" << std::endl;
	if (VERBOSE >= 2)
		for (int h = 0; h < howmany; h++)
			for (int id = 0; id < nprocs; ++id) {
				if (procid == id)
					for (int i = 0; i < local_n0; i++) {
						std::cout << std::endl;
						for (int j = 0; j < N[1]; j++) {
							std::cout << '\t'
									<< data[h * idist
											+ (i * N[1] + j) * n_tuples];
						}
					}
				std::cout << '\n';
			}
#endif

	//PCOUT<<" ============================================= "<<std::endl;
	//PCOUT<<" ==============   Local Transpose============= "<<std::endl;
	//PCOUT<<" ============================================= "<<std::endl;
	int ptr = 0;
	shuffle_time -= MPI_Wtime();
	if (nprocs == 1 && Flags[0] == 1 && Flags[1] == 1) {
		for (int h = 0; h < howmany; h++)
			local_transpose(local_n1, N[0], n_tuples, &data[h * idist]);
	}
	if (nprocs == 1 && Flags[0] == 0 && Flags[1] == 0) {
		for (int h = 0; h < howmany; h++)
			local_transpose(N[0], N[1], n_tuples, &data[h * idist]);
	}
	if (nprocs == 1) { // Transpose is done!
		shuffle_time += MPI_Wtime();
		timings[0] += MPI_Wtime();
		timings[1] += shuffle_time;
		timings[2] += 0;
		timings[3] += 0;
		return;
	}

	PCOUT << "\n flags=" << flags << std::endl;
	// if the input is transposed, then transpose it to get to non transposed input
	if (Flags[0] == 1) {
		for (int i = 0; i < howmany; i++)
			local_transpose(local_n0, N[1], n_tuples, &data[i * idist]);
#ifdef VERBOSE2
		if (VERBOSE >= 2)
			PCOUT << "Local Transpose:" << std::endl;
		if (VERBOSE >= 2)
			for (int h = 0; h < howmany; h++)
				for (int id = 0; id < nprocs; ++id) {
					if (procid == id)
						for (int i = 0; i < N[1]; i++) {
							std::cout << std::endl;
							for (int j = 0; j < local_n0; j++) {
								std::cout << '\t'
										<< data[h * idist
												+ (i * local_n0 + j) * n_tuples];
							}
						}
					std::cout << '\n';
				}
#endif
	}
	shuffle_time += MPI_Wtime();

	// if the input is transposed and the output needs to be transposed locally then do the local transpose

	//PCOUT<<" ============================================= "<<std::endl;
	//PCOUT<<" ==============   MPIALLTOALL  =============== "<<std::endl;
	//PCOUT<<" ============================================= "<<std::endl;

	int* scount_proc = T_plan->scount_proc_v8;
	int* rcount_proc = T_plan->rcount_proc_v8;
	int* soffset_proc = T_plan->soffset_proc_v8;
	int* roffset_proc = T_plan->roffset_proc_v8;

	MPI_Datatype* stype = T_plan->stype_v8;
	MPI_Datatype* rtype = T_plan->rtype_v8;
	comm_time -= MPI_Wtime();

	int soffset = 0, roffset = 0;
	MPI_Status ierr;
	MPI_Request * s_request = new MPI_Request[nprocs];
	MPI_Request * request = new MPI_Request[nprocs];

	// SEND
	for (int proc = 0; proc < nprocs; ++proc) {
		soffset = soffset_proc[proc];
		MPI_Isend(&data[soffset], 1, stype[proc], proc, tag, T_plan->comm,
				&s_request[proc]);
	}
	// RECV
	for (int proc = 0; proc < nprocs; ++proc) {
		roffset = roffset_proc[proc];
		MPI_Irecv(&send_recv[roffset], 1, rtype[proc], proc, tag, T_plan->comm,
				&request[proc]);
	}

	for (int proc = 0; proc < nprocs; ++proc) {
		MPI_Wait(&request[proc], &ierr);
		MPI_Wait(&s_request[proc], &ierr);
	}
	comm_time += MPI_Wtime();

#ifdef VERBOSE2
	if (VERBOSE >= 2)
		PCOUT << "MPIAlltoAll:" << std::endl;
	if (VERBOSE >= 2)
		for (int h = 0; h < howmany; h++)
			for (int id = 0; id < nprocs; ++id) {
				if (procid == id)
					for (int i = 0; i < local_n1; i++) {
						std::cout << std::endl;
						for (int j = 0; j < N[0]; j++) {
							std::cout << '\t'
									<< send_recv[h * odist
											+ (i * N[0] + j) * n_tuples]; //<<","<<send_recv[(i*N[0]+j)*n_tuples+1];
						}
					}
				std::cout << '\n';
			}
#endif

	//PCOUT<<" ============================================= "<<std::endl;
	//PCOUT<<" ============== 2nd Local Trnaspose ========== "<<std::endl;
	//PCOUT<<" ============================================= "<<std::endl;
	reshuffle_time -= MPI_Wtime();
	ptr = 0;

	// TODO :  interleave this memcpy into the send recvs!
	// write a tranposed out dat type.
	// implement the how many case.
	// If none of these worked try the mpi start!
	memcpy(data, send_recv, T_plan->alloc_local);
	if (Flags[1] == 1) { // Transpose output
		if (howmany == 1)
			local_transpose(local_n1, N[0], n_tuples, data);
		else {
			for (int h = 0; h < howmany; h++)
				local_transpose(local_n1, N[0], n_tuples, &data[h * odist]);
		}

#ifdef VERBOSE2
		if (VERBOSE >= 2)
			PCOUT << "Transposed Out" << std::endl;
		if (VERBOSE >= 2)
			for (int h = 0; h < howmany; h++)
				for (int id = 0; id < nprocs_1; ++id) {
					if (procid == id)
						for (int i = 0; i < N[0]; i++) {
							std::cout << std::endl;
							for (int j = 0; j < local_n1; j++) {
								std::cout << '\t'
										<< data[h * odist
												+ (i * local_n1 + j) * n_tuples];
							}
						}
					std::cout << '\n';
				}
#endif
	}
	reshuffle_time += MPI_Wtime();
	delete[] request;
	delete[] s_request;

#ifdef VERBOSE1
	if (VERBOSE >= 1) {
		PCOUT << "Shuffle Time= " << shuffle_time << std::endl;
		PCOUT << "Alltoall Time= " << comm_time << std::endl;
		PCOUT << "Reshuffle Time= " << reshuffle_time << std::endl;
		PCOUT << "Total Time= " << (shuffle_time + comm_time + reshuffle_time)
				<< std::endl;
	}
#endif
	timings[0] += MPI_Wtime(); //timings[0]+=shuffle_time+comm_time+reshuffle_time;
	timings[1] += shuffle_time;
	timings[2] += comm_time;
	timings[3] += reshuffle_time;
	return;
} // end fast_transpose_v8

template class T_Plan<double> ;
template class T_Plan<float> ;

template class Mem_Mgr<double> ;
template class Mem_Mgr<float> ;
