#include <iostream>
#include <cuda.h>
#include <stdlib.h>
#include <vector>

template <typename T>
__global__ void local_transpose_cuda1(int r, int c, int n_tuples, T * in, T*out) {
	T* in_=& in[(blockIdx.x*c+blockIdx.y)*n_tuples];
	T* out_=&out[(blockIdx.y*r+blockIdx.x)*n_tuples];
	for(int z=threadIdx.x;z<n_tuples;z+=blockDim.x) out_[z]=in_[z];
}

template <typename T>
__global__ void local_transpose_cuda2(int r, int c, int n_tuples, int n_tuples2, T* in, T *out) {
	size_t n_tup=n_tuples;
	if(blockIdx.x==r-1) n_tup=n_tuples2;

	T* in_=& in[blockIdx.x*((c )*n_tuples )+blockIdx.y*n_tup ];
	T* out_=&out[blockIdx.y*((r-1)*n_tuples+n_tuples2)+blockIdx.x*n_tuples];
	for(int z=threadIdx.x;z<n_tup;z+=blockDim.x) out_[z]=in_[z];
}

template <typename T>
__global__ void local_transpose_cuda3(int r, int c, int n_tuples, T * A) {
	__shared__ T buff[512];
	size_t i_, j_;
	int i=blockIdx.x;
	int j=blockIdx.y;

	size_t src=j+c*i;
	size_t trg=i+r*j;
	if(src==trg) return; // nothing to be done

	size_t cycle_len=0;
	while(src<trg) { // find cycle
		i_=trg/c;
		j_=trg%c;
		trg=i_+r*j_;
		cycle_len++;
	}
	if(src!=trg) return;

	for(size_t offset=0;offset<n_tuples;offset+=512) {
		//memcpy(buff, A+trg*n_tuples, n_tuples*sizeof(T));
		for(int z=threadIdx.x;z<512 && offset+z<n_tuples;z+=blockDim.x) buff[z]=A[trg*n_tuples+offset+z];
		for(size_t k=0;k<cycle_len;k++) { // reverse cycle
			j_=trg/r;
			i_=trg%r;
			src=j_+c*i_;
			//memcpy(A+trg*n_tuples, A+src*n_tuples, n_tuples*sizeof(T));
			for(int z=threadIdx.x;z<512 && offset+z<n_tuples;z+=blockDim.x) A[trg*n_tuples+offset+z]=A[src*n_tuples+offset+z];
			trg=src;
		}
		//memcpy(A+trg*n_tuples, buff, n_tuples*sizeof(T));
		for(int z=threadIdx.x;z<512 && offset+z<n_tuples;z+=blockDim.x) A[trg*n_tuples+offset+z]=buff[z];
	}
}

template <typename T>
__global__ void local_transpose_cuda4(int r, int c, int n_tuples, int n_tuples2,T* in, T* out ) {
	size_t n_tup=n_tuples;
	if(blockIdx.y==c-1) n_tup=n_tuples2;
	T* in_=& in[blockIdx.x*((c-1)*n_tuples+n_tuples2)+blockIdx.y*n_tuples];
	T* out_=&out[blockIdx.y*r*n_tuples+blockIdx.x*n_tup];
	for(int z=threadIdx.x;z<n_tup;z+=blockDim.x) out_[z]=in_[z];
}

template <typename T>
__global__ void memcpy_cuda(T** src_ptr_d, T** trg_ptr_d, size_t* vec_len_d) {
	size_t k=blockIdx.x;
	T* src_ptr=src_ptr_d[k];
	T* trg_ptr=trg_ptr_d[k];
	size_t vec_len=vec_len_d[k];
	for(int i=threadIdx.x;i<vec_len;i+=blockDim.x) trg_ptr[i]=src_ptr[i];
}

// outplace local transpose multiple n_tuples for the last col
template<typename T>
void local_transpose_col_cuda(int r, int c, int n_tuples, int n_tuples2, T* in,
		T* out) {
	T* in_d = (T*) in;
	T* out_d = (T*) out;
	//size_t size=c*((r-1)*n_tuples+n_tuples2);
	//cudaMalloc(& in_d, size*sizeof(T));
	//cudaMalloc(&out_d, size*sizeof(T));
	//cudaMemcpy(in_d, in, size*sizeof(T), cudaMemcpyHostToDevice);

	dim3 blocks(r, c);
	int threads = 128;
local_transpose_cuda4<<<blocks,threads>>>(r,c,n_tuples,n_tuples2,in_d,out_d);

//cudaMemcpy(out, out_d, size*sizeof(T), cudaMemcpyDeviceToHost);
//cudaFree(out_d);
//cudaFree(in_d);
}

template<typename T>
void memcpy_v1_h1(int nprocs_1, int howmany, int local_n0, int n_tuples,
	ptrdiff_t* local_n1_proc, T* send_recv_d, T* data, int idist, int N1,
	ptrdiff_t* local_1_start_proc) {

static std::vector<T*> src_ptr;
static std::vector<T*> trg_ptr;
static std::vector<size_t> vec_len;

static T** src_ptr_d = NULL;
static T** trg_ptr_d = NULL;
static size_t* vec_len_d = NULL;

size_t cpy_cnt = nprocs_1 * howmany * local_n0;
if (src_ptr.size() < cpy_cnt) {
	src_ptr.resize(cpy_cnt);
	trg_ptr.resize(cpy_cnt);
	vec_len.resize(cpy_cnt);
	cudaFree(src_ptr_d);
	cudaFree(trg_ptr_d);
	cudaFree(vec_len_d);
	cudaMalloc(&src_ptr_d, cpy_cnt * sizeof(T*));
	cudaMalloc(&trg_ptr_d, cpy_cnt * sizeof(T*));
	cudaMalloc(&vec_len_d, cpy_cnt * sizeof(size_t));
}

size_t k = 0;
size_t ptr = 0;
for (int proc = 0; proc < nprocs_1; ++proc)
	for (int h = 0; h < howmany; ++h) {
		for (int i = 0; i < local_n0; ++i) {
			trg_ptr[k] = &send_recv_d[ptr];
			src_ptr[k] = &data[h * idist
					+ (i * N1 + local_1_start_proc[proc]) * n_tuples];
			vec_len[k] = n_tuples * local_n1_proc[proc];
			ptr += n_tuples * local_n1_proc[proc];
			k++;
		}
	}

cudaMemcpy(src_ptr_d, &src_ptr[0], sizeof(T*) * cpy_cnt,
		cudaMemcpyHostToDevice);
cudaMemcpy(trg_ptr_d, &trg_ptr[0], sizeof(T*) * cpy_cnt,
		cudaMemcpyHostToDevice);
cudaMemcpy(vec_len_d, &vec_len[0], sizeof(size_t) * cpy_cnt,
		cudaMemcpyHostToDevice);
memcpy_cuda<<<cpy_cnt,128>>>(src_ptr_d, trg_ptr_d, vec_len_d);
	//for(size_t i=0;i<cpy_cnt;i++){
	//  cudaMemcpy(trg_ptr[i], src_ptr[i], sizeof(T)*vec_len[i], cudaMemcpyDeviceToDevice);
	//}
}

template<typename T>
void memcpy_v1_h2(int nprocs_0, int howmany, ptrdiff_t* local_0_start_proc,
ptrdiff_t* local_n0_proc, T* data, int odist, int local_n1, int n_tuples,
T* send_recv_cpu) {

static std::vector<T*> src_ptr;
static std::vector<T*> trg_ptr;
static std::vector<size_t> vec_len;

static T** src_ptr_d = NULL;
static T** trg_ptr_d = NULL;
static size_t* vec_len_d = NULL;

size_t cpy_cnt = 0;
for (int proc = 0; proc < nprocs_0; ++proc)
for (int h = 0; h < howmany; ++h) {
	cpy_cnt += local_n0_proc[proc];
}

if (src_ptr.size() < cpy_cnt) {
src_ptr.resize(cpy_cnt);
trg_ptr.resize(cpy_cnt);
vec_len.resize(cpy_cnt);
cudaFree(src_ptr_d);
cudaFree(trg_ptr_d);
cudaFree(vec_len_d);
cudaMalloc(&src_ptr_d, cpy_cnt * sizeof(T*));
cudaMalloc(&trg_ptr_d, cpy_cnt * sizeof(T*));
cudaMalloc(&vec_len_d, cpy_cnt * sizeof(size_t));
}

size_t k = 0;
size_t ptr = 0;
for (int proc = 0; proc < nprocs_0; ++proc)
for (int h = 0; h < howmany; ++h) {
	for (int i = local_0_start_proc[proc];
			i < local_0_start_proc[proc] + local_n0_proc[proc]; ++i) {
		trg_ptr[k] = &data[h * odist + (i * local_n1) * n_tuples];
		src_ptr[k] = &send_recv_cpu[ptr];
		vec_len[k] = local_n1 * n_tuples;
		ptr += n_tuples * local_n1;
		k++;
	}
}

cudaMemcpy(src_ptr_d, &src_ptr[0], sizeof(T*) * cpy_cnt,
	cudaMemcpyHostToDevice);
cudaMemcpy(trg_ptr_d, &trg_ptr[0], sizeof(T*) * cpy_cnt,
	cudaMemcpyHostToDevice);
cudaMemcpy(vec_len_d, &vec_len[0], sizeof(size_t) * cpy_cnt,
	cudaMemcpyHostToDevice);
memcpy_cuda<<<cpy_cnt,128>>>(src_ptr_d, trg_ptr_d, vec_len_d);
  //for(size_t i=0;i<cpy_cnt;i++){
  //  cudaMemcpy(trg_ptr[i], src_ptr[i], sizeof(T)*vec_len[i], cudaMemcpyDeviceToDevice);
  //}
}

// outplace local transpose multiple n_tuples
template<typename T>
void local_transpose_cuda(int r, int c, int n_tuples, int n_tuples2, T * in,
T *out) {
T* in_d = (T*) in;
T* out_d = (T*) out;
  //size_t size=c*((r-1)*n_tuples+n_tuples2);
  //cudaMalloc(& in_d, size*sizeof(T));
  //cudaMalloc(&out_d, size*sizeof(T));
  //cudaMemcpy(in_d, in, size*sizeof(T), cudaMemcpyHostToDevice);

dim3 blocks(r, c);
int threads = 128;
local_transpose_cuda2<<<blocks,threads>>>(r,c,n_tuples,n_tuples2,in_d,out_d);

  //cudaMemcpy(out, out_d, size*sizeof(T), cudaMemcpyDeviceToHost);
  //cudaFree(out_d);
  //cudaFree(in_d);
}

// in place local transpose
template<typename T>
void local_transpose_cuda(int r, int c, int n_tuples, T* A) {
T* A_d = (T*) A;
  //cudaMalloc(&A_d, r*c*n_tuples*sizeof(T));
  //cudaMemcpy(A_d, A, r*c*n_tuples*sizeof(T), cudaMemcpyHostToDevice);

dim3 blocks(r, c);
int threads = 256;
local_transpose_cuda3<<<blocks,threads>>>(r,c,n_tuples,A_d);

  //cudaMemcpy(A, A_d, r*c*n_tuples*sizeof(T), cudaMemcpyDeviceToHost);
  //cudaFree(A_d);
}

// outplace local transpose
template<typename T>
void local_transpose_cuda(int r, int c, int n_tuples, T * in, T *out) {
T* in_d = (T*) in;
T* out_d = (T*) out;
  //cudaMalloc(& in_d, r*c*n_tuples*sizeof(T));
  //cudaMalloc(&out_d, r*c*n_tuples*sizeof(T));
  //cudaMemcpy(in_d, in, r*c*n_tuples*sizeof(T), cudaMemcpyHostToDevice);

dim3 blocks(r, c);
int threads = 128;
local_transpose_cuda1<<<blocks,threads>>>(r,c,n_tuples,in_d,out_d);

  //cudaMemcpy(out, out_d, r*c*n_tuples*sizeof(T), cudaMemcpyDeviceToHost);
  //cudaFree(out_d);
  //cudaFree(in_d);
}

// forward decleration for double
template void local_transpose_cuda<double>(int r, int c, int n_tuples,
int n_tuples2, double * in, double *out);
template void local_transpose_cuda<double>(int r, int c, int n_tuples,
double* A);
template void local_transpose_cuda<double>(int r, int c, int n_tuples,
double * in, double *out);
template void local_transpose_col_cuda<double>(int r, int c, int n_tuples,
int n_tuples2, double* in, double* out);
template void memcpy_v1_h1<double>(int nprocs_1, int howmany, int local_n0,
int n_tuples, ptrdiff_t* local_n1_proc, double* send_recv_d, double* data,
int idist, int N1, ptrdiff_t* local_1_start_proc);
template void memcpy_v1_h2<double>(int nprocs_0, int howmany,
ptrdiff_t* local_0_start_proc, ptrdiff_t* local_n0_proc, double* data,
int odist, int local_n1, int n_tuples, double* send_recv_cpu);

template __global__ void local_transpose_cuda1<double>(int r, int c, int n_tuples, double* in, double*out);
template __global__ void local_transpose_cuda2<double>(int r, int c, int n_tuples, int n_tuples2, double* in, double *out);
template __global__ void local_transpose_cuda3<double>(int r, int c, int n_tuples, double* A);
template __global__ void local_transpose_cuda4<double>(int r, int c, int n_tuples, int n_tuples2,double* in, double* out );
template __global__ void memcpy_cuda<double>(double** src_ptr_d, double** trg_ptr_d, size_t* vec_len_d);

// forward decleration for float

template void local_transpose_cuda<float>(int r, int c, int n_tuples,
int n_tuples2, float * in, float *out);
template void local_transpose_cuda<float>(int r, int c, int n_tuples, float* A);
template void local_transpose_cuda<float>(int r, int c, int n_tuples,
float * in, float *out);
template void local_transpose_col_cuda<float>(int r, int c, int n_tuples,
int n_tuples2, float* in, float* out);
template void memcpy_v1_h1<float>(int nprocs_1, int howmany, int local_n0,
int n_tuples, ptrdiff_t* local_n1_proc, float* send_recv_d, float* data,
int idist, int N1, ptrdiff_t* local_1_start_proc);
template void memcpy_v1_h2<float>(int nprocs_0, int howmany,
ptrdiff_t* local_0_start_proc, ptrdiff_t* local_n0_proc, float* data, int odist,
int local_n1, int n_tuples, float* send_recv_cpu);

template __global__ void local_transpose_cuda1<float>(int r, int c, int n_tuples, float* in, float*out);
template __global__ void local_transpose_cuda2<float>(int r, int c, int n_tuples, int n_tuples2, float* in, float* out);
template __global__ void local_transpose_cuda3<float>(int r, int c, int n_tuples, float* A);
template __global__ void local_transpose_cuda4<float>(int r, int c, int n_tuples, int n_tuples2,float* in, float* out );
template __global__ void memcpy_cuda<float>(float** src_ptr_d, float** trg_ptr_d, size_t* vec_len_d);
