# require cmake 2.8.0 for findCUDA.cmake
CMAKE_MINIMUM_REQUIRED(VERSION 2.8.0 FATAL_ERROR)

##################
project("accfft")
# The version number.
set (accfft_VERSION_MAJOR 0)
set (accfft_VERSION_MINOR 0)
##################

# guard against in-source builds
if(${CMAKE_SOURCE_DIR} STREQUAL ${CMAKE_BINARY_DIR})
  message(FATAL_ERROR "In-source builds not allowed. Please make a new directory (called a build directory) and run CMake from there. You may need to remove CMakeCache.txt. ")
endif()

# Set default install path t o build
if (CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    set (CMAKE_INSTALL_PREFIX "${CMAKE_BINARY_DIR}" CACHE PATH "default install path" FORCE )
endif()

# allow cmake to use custom modules (e.g. FindFFTW)
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)


find_package(FFTW REQUIRED)
find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)
find_package(PNETCDF)
# Make the include path visible to all targets
INCLUDE_DIRECTORIES (
  ${accfft_SOURCE_DIR}/include
  ${accfft_BINARY_DIR}/include
  )



# Add OpenMP and FFTW cxx flags to all targets
# FOR PGI COMPILER
#set(OpenMP_C_FLAGS "-lpthread")
#set(OpenMP_CXX_FLAGS "-lpthread")
set( CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3  ${OpenMP_C_FLAGS} -I${FFTW_INCLUDES}" )
set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 ${OpenMP_CXX_FLAGS} -I${FFTW_INCLUDES}" )

# some default behavior
if(BUILD_SHARED)
  set(BUILD_SHARED_LIBS ON)
else()
  set(BUILD_SHARED_LIBS OFF)
endif(BUILD_SHARED)

# ###########
# HOST BUILD 
# ###########
include(FindMPI)

# set cmake compiler to mpi compiler wrapper found by FindMPI (usually mpicxx) 
set(CMAKE_CXX_COMPILER ${MPI_CXX_COMPILER})
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}" )

#include_directories(SYSTEM ${MPI_INCLUDE_PATH})

# libaccfft source files list
set(libaccfft_SRCS
  src/transpose.cpp 
  src/accfft.cpp 
  src/accfft_common.cpp
  src/pnetcdf_io.cpp
  )

# build CPU libaccfft (shared or static ?)
if(BUILD_SHARED_LIBS)
  add_library(accfft SHARED ${libaccfft_SRCS})
else(BUILD_SHARED_LIBS)
  add_library(accfft STATIC ${libaccfft_SRCS})
endif(BUILD_SHARED_LIBS)

# install
install(TARGETS accfft DESTINATION lib)
#install(DIRECTORY accfft DESTINATION include FILES_MATCHING PATTERN "include/*.h include/*.txx")
install (DIRECTORY include/ DESTINATION include FILES_MATCHING PATTERN "*.txx")
install (DIRECTORY include/ DESTINATION include FILES_MATCHING PATTERN "*.h"
  PATTERN "**~" EXCLUDE
  PATTERN "*_gpu*" EXCLUDE
  PATTERN "*_cuda*" EXCLUDE
  )

# ##########
# GPU BUILD
# ##########
if(BUILD_GPU)
  find_package(CUDA REQUIRED)
  include(FindCUDA)
  list(APPEND CUDA_NVCC_FLAGS --compiler-options -fno-strict-aliasing -lineinfo)
  list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_20,code=sm_20)
  list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_30,code=sm_30)
  list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_35,code=sm_35)
  
  # libaccfft_gpu source files list
  set(libaccfft_gpu_SRCS
    src/transpose_gpu.cpp
    src/transpose_cuda.cu
    src/accfft_gpu.cpp 
    src/accfft_common.cpp
    )
  
  
  # GPU libaccfft_gpu
  if(BUILD_SHARED_LIBS)
    cuda_add_library(accfft_gpu SHARED ${libaccfft_gpu_SRCS} OPTIONS "-DENABLE_GPU")
  else(BUILD_SHARED_LIBS)
    cuda_add_library(accfft_gpu STATIC ${libaccfft_gpu_SRCS} OPTIONS "-DENABLE_GPU")
  endif(BUILD_SHARED_LIBS)
  
  # the following is necessary so that .cpp sources files passed to 
  # cuda_add_library get the right compile flags
  set_target_properties(accfft_gpu PROPERTIES COMPILE_FLAGS "-DENABLE_GPU")
  
  # install
  install(TARGETS accfft_gpu DESTINATION lib)
  install (DIRECTORY include/ DESTINATION include FILES_MATCHING PATTERN
    "*gpu*")
  install (DIRECTORY include/ DESTINATION include FILES_MATCHING PATTERN
    "*cuda*")
endif(BUILD_GPU)
####################
# EXAMPLES / TESTS #
####################
if(BUILD_STEPS)
add_subdirectory(steps/step1)
endif(BUILD_STEPS)
