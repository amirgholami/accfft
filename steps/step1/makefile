# -*-Makefile-*-
MPICXX ?= mpicxx
BINDIR = bin
MKDIR_P = mkdir -p
OUT_DIR = bin

CXXFLAGS= -O3 -Wall -DENABLE_GPU -fopenmp
CXXFLAGS += -xhost # For TACC Systems

CUDA_ROOT ?= /usr/local/cuda-6.5
CUDA_LIB ?= $(CUDA_ROOT)/lib64
CUDA_INC ?= $(CUDA_ROOT)/include
CUDA_SM ?= 35

FFTW_LIB ?= /usr/lib
FFTW_INC ?= /usr/include
LDFLAGS=-L$(ACCFFT_DIR)/lib -laccfft -L$(FFTW_LIB)  -lfftw3 -lfftw3_threads

INCDIR= -I$(FFTW_INC) -I$(ACCFFT_DIR)/include/



all: step1

step1.o:	step1.cpp
	$(MPICXX) $(CXXFLAGS) -c $^ -o $@  $(INCDIR)

step1:	step1.o
	$(MPICXX) $(CXXFLAGS) $^  -o $@ $(LDFLAGS)


# To be used for compiling *.cu files
N_INC= -I$(CUDA_INC) 
N_LIB= -L$(CUDA_LIB) -L$(FFTW_LIB) 
N_FLAGS = -c -O3 -gencode arch=compute_$(CUDA_SM),code=sm_$(CUDA_SM) -Xcompiler -fopenmp -DENABLE_GPU 
N_FLAGS += -I../../include

# To be used for compiling gpu *.cpp files
N_CXXFLAGS= -O3 -lcudart -lcufft -fopenmp -lfftw3 -lfftw3_threads -DENABLE_GPU
N_CXXINC= -I$(CUDA_INC) -I$(FFTW_INC) -I$(ACCFFT_DIR)/include
N_CXXLIB= -L$(CUDA_LIB) -L$(FFTW_LIB) -L$(ACCFFT_DIR)/lib -laccfft_gpu -lfftw3

step1_gpu.o: step1_gpu.cpp
	$(MPICXX) $(N_CXXFLAGS) -c $^ $(N_CXXINC) -o $@  

step1_gpu:	step1_gpu.o kernels.o
	$(MPICXX) $(N_CXXFLAGS) $^ $(N_CXXLIB) -o $@ 

kernels.o:	kernels.cu
	nvcc $^ -o $@ $(N_LIB) $(N_FLAGS)

clean:
	-rm -f *~ *.o step1 step1_gpu
