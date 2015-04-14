# -*-Makefile-*-
MPICXX ?= mpicxx
BINDIR = bin
MKDIR_P = mkdir -p
OUT_DIR = bin

CXXFLAGS= -O3 -Wall -fopenmp

CUDA_ROOT ?= /usr/local/cuda-6.5
CUDA_LIB ?= $(CUDA_ROOT)/lib64
CUDA_INC ?= $(CUDA_ROOT)/include
CUDA_SM ?= 35

FFTW_LIB ?= /usr/lib
FFTW_INC ?= /usr/include

LDFLAGS=  -L$(FFTW_LIB) -lfftw3 -lfftw3_threads -fopenmp

INCDIR= -I$(FFTW_INC) -I./ -I$(shell pwd)/include

# OpenMPI
#MPI_LIB ?= $(shell mpicxx --showme:link)
#MPI_INC ?= $(shell mpicxx --showme:compile)
# MVAPICH
#MPI_LIB ?= $(shell mpicxx -compile-info)
#MPI_INC ?= $(shell mpicxx -link-info)


N_INC= -I$(CUDA_INC) -I./ -I./alltoallkway -I$(FFTW_INC) 
N_LIB= -L$(CUDA_LIB) -L$(FFTW_LIB)
N_FLAGS=-c  -O3 -gencode arch=compute_$(CUDA_SM),code=sm_$(CUDA_SM)  -Xcompiler -fopenmp -DENABLE_GPU
N_FLAGS += $(addprefix -Xcompiler , $(N_INC))
#N_FLAGS += $(addprefix -Xcompiler , $(MPI_INC))


N_CXXFLAGS= -O3 -lcudart -lcufft -fopenmp -lfftw3 -lfftw3_threads -DENABLE_GPU
N_CXXINC= -I$(CUDA_INC) -I$(FFTW_INC)
N_CXXLIB= -L$(CUDA_LIB) -L$(FFTW_LIB)


all: lib/libaccfft.a


src/transpose.o: src/transpose.cpp
	$(MPICXX) $(CXXFLAGS) -c src/transpose.cpp -o $@  $(INCDIR) $(LDFLAGS)
src/accfft.o: src/accfft.cpp
	$(MPICXX) $(CXXFLAGS) -c src/accfft.cpp -o $@  $(INCDIR) $(LDFLAGS)
src/accfft_common.o: src/accfft_common.cpp
	$(MPICXX) $(CXXFLAGS) -c src/accfft_common.cpp -o $@  $(INCDIR) $(LDFLAGS)

lib/libaccfft.a:	src/transpose.o src/accfft.o src/accfft_common.o
	mkdir -p lib
	ar crf lib/libaccfft.a $^  
	echo "Please add the following to your .bashrc: export AccFFT_DIR=$(shell pwd)"

src/accfft_gpu.o: src/accfft_gpu.cpp
	$(MPICXX) $(CXXFLAGS) -c $^ -o $@  $(INCDIR) $(LDFLAGS) $(N_CXXFLAGS) $(N_CXXLIB) $(N_CXXINC)

src/transpose_gpu.o: src/transpose_gpu.cpp
	$(MPICXX) $(CXXFLAGS) -c $^ -o $@  $(INCDIR) $(LDFLAGS) $(N_CXXFLAGS) $(N_CXXLIB) $(N_CXXINC)

src/transpose_cuda.o: src/transpose_cuda.cu
	nvcc $^ -o $@ $(N_LIB) $(N_FLAGS) -DENABLE_GPU

lib/libaccfft_gpu.a:	src/transpose_gpu.o src/transpose_cuda.o src/accfft_gpu.o src/accfft_common.o
	mkdir -p lib
	ar crf lib/libaccfft_gpu.a $^  
	echo "Please add the following to your .bashrc: export AccFFT_DIR=$(shell pwd)"


clean:
	-rm -f src/*.o src/*~ lib/*
