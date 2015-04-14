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
CUDA_SM ?= 20

FFTW_LIB ?= /usr/lib
FFTW_INC ?= /usr/include
LDFLAGS=-L$(AccFFT_DIR)/lib -laccfft -L$(FFTW_LIB)  -lfftw3 -lfftw3_threads
LDFLAGS_GPU=-L$(AccFFT_DIR)/lib -laccfft_gpu -lcufft

INCDIR= -I$(FFTW_INC) -I$(AccFFT_DIR)/include/

# OpenMPI
#MPI_LIB ?= $(shell mpicxx --showme:link)
#MPI_INC ?= $(shell mpicxx --showme:compile)
# MVAPICH
#MPI_LIB ?= $(shell mpicxx -compile-info)
#MPI_INC ?= $(shell mpicxx -link-info)


N_INC= -I$(CUDA_INC) -I./ -I$(FFTW_INC) 
N_LIB= -L$(CUDA_LIB) -L$(FFTW_LIB)

N_CXXFLAGS= -O3 -lcudart -lcufft -fopenmp -lfftw3 -lfftw3_threads -DENABLE_GPU
N_CXXINC= -I$(CUDA_INC) -I$(FFTW_INC)
N_CXXLIB= -L$(CUDA_LIB) -L$(FFTW_LIB)

N_FLAGS = -c -O3 -gencode arch=compute_$(CUDA_SM),code=sm_$(CUDA_SM) -Xcompiler -fopenmp -DENABLE_GPU 
N_FLAGS += -Xcompiler -I../../include
N_FLAGS += $(addprefix -Xcompiler ,$(MPI_INC))
N_FLAGS += $(addprefix -Xcompiler ,$(N_INC))
N_FLAGS += $(addprefix -Xcompiler ,$(N_CXXINC))


all: step1

step1.o:	step1.cpp
	$(MPICXX) $(CXXFLAGS) -c $^ -o $@  $(INCDIR)

step1_gpu.o: step1_gpu.cpp
	$(MPICXX) $(CXXFLAGS) -c $^ -o $@  $(INCDIR) $(N_CXXFLAGS) $(N_CXXLIB) $(N_CXXINC)

step1:	step1.o
	$(MPICXX) $(CXXFLAGS) $^  -o $@ $(LDFLAGS)

clean:
	-rm -f *~ *.o step1
