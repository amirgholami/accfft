CXX=mpicxx
BINDIR = bin
MKDIR_P = mkdir -p
OUT_DIR= bin

CXXFLAGS= -O3 -xhost -Wall 

LDFLAGS= -L$(AccFFT_DIR)/lib -laccfft -L$(FFTW_LIB)  -lfftw3 -lfftw3_threads -fopenmp

INCDIR= -I$(FFTW_INC)  -I$(AccFFT_DIR)/include/


N_FLAGS=-c  -O3 -gencode arch=compute_35,code=sm_35  -Xcompiler -fopenmp -DENABLE_GPU
N_INC= -I$(TACC_CUDA_INC) -I./ -I$(FFTW_INC) 
N_LIB= -L$(TACC_CUDA_LIB) -L$(FFTW_LIB)

N_CXXFLAGS= -O3 -lcudart -lcufft -fopenmp -lfftw3 -lfftw3_threads
N_CXXLIB= -I$(TACC_CUDA_INC) -I$(FFTW_INC)
N_CXXINC= -L$(TACC_CUDA_LIB) -L$(FFTW_LIB)

MPI_INC = -I/opt/apps/intel14/mvapich2/2.0b/include   -DMPICH_IGNORE_CXX_SEEK -DMPICH_SKIP_MPICXX
MPI_LIB = -L/opt/apps/intel14/mvapich2/2.0b/lib

all: step2


step2.o:	step2.cpp
	$(CXX) $(CXXFLAGS) -c step2.cpp -o $@  $(INCDIR) $(LDFLAGS) $(MPI_INC) $(MPI_LIB)
step2:	step2.o
	$(CXX) $(CXXFLAGS)  step2.o  $(LDFLAGS)  -o $@ $(INCDIR) 
clean:
	-rm *~ *.o step2
