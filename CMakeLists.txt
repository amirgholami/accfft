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

# Set default install path to build
if (CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    set (CMAKE_INSTALL_PREFIX "${CMAKE_BINARY_DIR}" CACHE PATH "default install path" FORCE )
endif()

# Required for setting install_name on OSX
set(CMAKE_INSTALL_NAME_DIR "${CMAKE_INSTALL_PREFIX}/lib")
set(MACOSX_RPATH ON)
if(POLICY CMP0068)
  cmake_policy(SET CMP0068 NEW) # CMake 3.9+: `RPATH` settings on macOS do not affect `install_name`.
endif()
if(POLICY CMP0074)
  cmake_policy(SET CMP0074 NEW) # find_package in CMake 3.12+ adds extra searches to <PackageName>_ROOT directories
endif()

# allow cmake to use custom modules (e.g. FindFFTW)
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)


find_package(FFTW REQUIRED)
find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)
# Make the include path visible to all targets
INCLUDE_DIRECTORIES (
  ${accfft_SOURCE_DIR}/include
  ${accfft_BINARY_DIR}/include
  )


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

# Add OpenMP and FFTW cxx flags to all targets
# FOR PGI COMPILER
#set(OpenMP_C_FLAGS "-lpthread")
#set(OpenMP_CXX_FLAGS "-lpthread")
set( CMAKE_C_FLAGS "-O3 ${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS} -I${FFTW_INCLUDES}" )
set( CMAKE_CXX_FLAGS "-O3 ${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS} -I${FFTW_INCLUDES}" )

#include_directories(SYSTEM ${MPI_INCLUDE_PATH})

# libaccfft source files list
set(libaccfft_SRCS
  src/transpose.cpp 
  src/accfft.cpp 
  src/accfft_common.cpp
  )


# build CPU libaccfft (shared or static ?)
if(BUILD_SHARED_LIBS)
  add_library(accfft SHARED ${libaccfft_SRCS})
else(BUILD_SHARED_LIBS)
  add_library(accfft STATIC ${libaccfft_SRCS})
endif(BUILD_SHARED_LIBS)
target_link_libraries(accfft ${FFTW_LIBRARIES})

# install
install(TARGETS accfft DESTINATION lib)
#install(DIRECTORY accfft DESTINATION include FILES_MATCHING PATTERN "include/*.h include/*.txx")
install (DIRECTORY include/ DESTINATION include FILES_MATCHING PATTERN "*.txx")
install (DIRECTORY include/ DESTINATION include FILES_MATCHING PATTERN "*.h"
  PATTERN "**~" EXCLUDE
  PATTERN "*_gpu*" EXCLUDE
  PATTERN "*_cuda*" EXCLUDE
  )


# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>#
#             CPU UTILS BUILD             #
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>#
find_package(PNETCDF QUIET)

if (PNETCDF_FOUND)
  set (libaccfft_utils_SRCS ${libaccfft_SRCS} 
    src/operators.cpp
    src/netcdf.cpp
    )
  include_directories(${PNETCDF_INCLUDE_DIRS})
  add_definitions(-DUSE_PNETCDF)
else()
  set (libaccfft_utils_SRCS ${libaccfft_SRCS} 
    src/operators.cpp
    src/netcdf.cpp
    )
endif()

# build CPU libaccfft (shared or static ?)
if(BUILD_SHARED_LIBS)
  add_library(accfft_utils SHARED ${libaccfft_utils_SRCS})
else(BUILD_SHARED_LIBS)
  add_library(accfft_utils STATIC ${libaccfft_utils_SRCS})
endif(BUILD_SHARED_LIBS)

target_link_libraries(accfft_utils accfft ${PNETCDF_LIBRARIES})

install(TARGETS accfft_utils DESTINATION lib)

# ##########
# GPU BUILD
# ##########
if(BUILD_GPU)
  find_package(CUDA REQUIRED)

  # flags advice from https://ankitshah009.blogspot.com/2019/01/nvidia-graphic-card-and-cuda.html
  if(NOT CUDA_NVCC_FLAGS AND NOT CUDA_VERSION VERSION_LESS "7.0")
    if(CUDA_VERSION VERSION_LESS "9.0") # versions 7-8
      list(APPEND CUDA_NVCC_FLAGS -arch=sm_30)
      list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_20,code=sm_20) # ended support in version 9
    else() # 9+
      list(APPEND CUDA_NVCC_FLAGS -arch=sm_50) # new default arch
    endif()

    # supported architectures on 7+
    list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_30,code=sm_30)
    list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_50,code=sm_50)
    list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_52,code=sm_52)
 
    if(NOT CUDA_VERSION VERSION_LESS "8.0") # cuda-8+
      list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_60,code=sm_60)
      list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_61,code=sm_61)
    endif()
    if(NOT CUDA_VERSION VERSION_LESS "9.0") # cuda-9+
      list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_70,code=sm_70)
    endif()
    if(NOT CUDA_VERSION VERSION_LESS "10.0") # cuda 10+
      list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_75,code=sm_75)
    endif()
 
    if(CUDA_VERSION VERSION_LESS "8.0") # cuda-7
      list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_52,code=compute_52)
    elseif(CUDA_VERSION VERSION_LESS "9.0") # cuda-8
      list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_61,code=compute_61)
    elseif(CUDA_VERSION VERSION_LESS "10.0") # cuda-9
      list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_70,code=compute_70)
    elseif(CUDA_VERSION VERSION_LESS "11.0") # cuda-10
      list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_75,code=compute_75)
    endif()
  endif()

  list(APPEND CUDA_NVCC_FLAGS --compiler-options -fno-strict-aliasing -lineinfo)

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


  # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>#
  #             GPU UTILS BUILD             #
  # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>#
  set (libaccfft_utils_gpu_SRCS ${libaccfft_gpu_SRCS} 
    src/operators_gpu.cpp
    src/operators_cuda.cu
    )
  # build CPU libaccfft (shared or static ?)
  if(BUILD_SHARED_LIBS)
    cuda_add_library(accfft_utils_gpu SHARED ${libaccfft_utils_gpu_SRCS} OPTIONS "-DENABLE_GPU")
  else(BUILD_SHARED_LIBS)
    cuda_add_library(accfft_utils_gpu STATIC ${libaccfft_utils_gpu_SRCS} OPTIONS "-DENABLE_GPU")
  endif(BUILD_SHARED_LIBS)
  target_link_libraries(accfft_utils_gpu accfft_gpu ${PNETCDF_LIBRARIES})

  # the following is necessary so that .cpp sources files passed to 
  # cuda_add_library get the right compile flags
  set_target_properties(accfft_utils_gpu PROPERTIES COMPILE_FLAGS "-DENABLE_GPU")

  install(TARGETS accfft_utils_gpu DESTINATION lib)


endif(BUILD_GPU)

####################
# EXAMPLES / TESTS #
####################
if(BUILD_STEPS)
  add_subdirectory(steps)
endif(BUILD_STEPS)
