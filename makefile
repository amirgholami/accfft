CXX=mpicxx
BINDIR = bin
MKDIR_P = mkdir -p
OUT_DIR= bin

CXXFLAGS= -O3 -xhost -Wall 

LDFLAGS=  -L$(FFTW_LIB) -lfftw3 -lfftw3_threads -fopenmp

INCDIR= -I$(FFTW_INC) -I./ -I/work/02187/gholami/maverick/accfft/include/


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

src/accfft_gpu.o: src/accfft_gpu.cpp
	$(CXX) $(CXXFLAGS) -c src/accfft_gpu.cpp -o $@  $(INCDIR) $(LDFLAGS) $(MPI_INC) $(MPI_LIB) $(N_CXXFLAGS) $(N_CXXLIB) $(N_CXXINC) -DENABLE_GPU

src/transpose_gpu.o: src/transpose_gpu.cpp
	$(CXX) $(CXXFLAGS) -c src/transpose_gpu.cpp -o $@  $(INCDIR) $(LDFLAGS) $(MPI_INC) $(MPI_LIB) $(N_CXXFLAGS) $(N_CXXLIB) $(N_CXXINC) -DENABLE_GPU
src/transpose_cuda.o: src/transpose_cuda.cu
	nvcc src/transpose_cuda.cu -o $@ $(MPI_INC) $(MPI_DIR) $(N_INC) $(N_LIB) $(N_FLAGS) -DENABLE_GPU
build/libaccfft_gpu.a:	src/transpose_gpu.o src/transpose_cuda.o src/accfft_gpu.o src/accfft_common.o
	mkdir -p lib
	ar crf lib/libaccfft_gpu.a src/transpose_gpu.o src/transpose_cuda.o src/accfft_gpu.o src/accfft_common.o  
	echo "export AccFFT_DIR=$(shell pwd)" >> ~/.bashrc


clean:
	-rm src/*.o src/*~ lib/*
