CXX=mpicxx
BINDIR = bin
MKDIR_P = mkdir -p
OUT_DIR= bin

CXXFLAGS= -O3 -xhost -Wall 

LDFLAGS=  -L$(FFTW_LIB) -lfftw3 -lfftw3_threads -fopenmp 

INCDIR= -I$(FFTW_INC) -I./alltoallkway -I./ -I./include


N_FLAGS=-c  -O3 -gencode arch=compute_35,code=sm_35  -Xcompiler -fopenmp -DENABLE_GPU
N_INC= -I$(TACC_CUDA_INC) -I./ -I./alltoallkway -I$(FFTW_INC) 
N_LIB= -L$(TACC_CUDA_LIB) -L$(FFTW_LIB)

N_CXXFLAGS= -O3 -lcudart -lcufft -fopenmp -lfftw3 -lfftw3_threads
N_CXXLIB= -I$(TACC_CUDA_INC) -I$(FFTW_INC)
N_CXXINC= -L$(TACC_CUDA_LIB) -L$(FFTW_LIB)

MPI_INC = -I/opt/apps/intel14/mvapich2/2.0b/include   -DMPICH_IGNORE_CXX_SEEK -DMPICH_SKIP_MPICXX
MPI_LIB = -L/opt/apps/intel14/mvapich2/2.0b/lib

all: build/libaccfft.a


transpose_gpu.o: transpose.cpp
	nvcc transpose.cpp -o transpose_gpu.o $(MPI_INC) $(MPI_DIR) $(N_INC) $(N_LIB) $(N_FLAGS)
transpose_cuda.o:	transpose_cuda.cu
	nvcc transpose_cuda.cu -o transpose_cuda.o $(MPI_INC) $(MPI_DIR) $(N_INC) $(N_LIB) $(N_FLAGS)

src/transpose.o: src/transpose.cpp
	$(CXX) $(CXXFLAGS) -c src/transpose.cpp -o $@  $(INCDIR) $(LDFLAGS) $(MPI_INC) $(MPI_LIB)
src/accfft.o: src/accfft.cpp
	$(CXX) $(CXXFLAGS) -c src/accfft.cpp -o $@  $(INCDIR) $(LDFLAGS) $(MPI_INC) $(MPI_LIB)
src/accfft_common.o: src/accfft_common.cpp
	$(CXX) $(CXXFLAGS) -c src/accfft_common.cpp -o $@  $(INCDIR) $(LDFLAGS) $(MPI_INC) $(MPI_LIB)

build/libaccfft.a:	src/transpose.o src/accfft.o src/accfft_common.o
	mkdir -p lib
	ar crf lib/libaccfft.a src/transpose.o src/accfft.o src/accfft_common.o  
	echo "export AccFFT_DIR=$(shell pwd)" >> ~/.bashrc
	source ~/.bashrc

clean:
	-rm src/*.o src/*~ lib/*
