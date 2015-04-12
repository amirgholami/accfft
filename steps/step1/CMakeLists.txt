# include dirs
include_directories(${MPI_INCLUDE_PATH})
include_directories(${FFTW_INCLUDE_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# source files list
set(step1_SRCS
  step1.cpp
)

#
# CPU executable
#
add_executable(step1 ${step1_SRCS})
target_link_libraries(step1
  accfft
  ${FFTW_LIBRARIES}
)

#
# GPU executable
#
# NOTE:
# We use a dirty hack: nvcc doesn't support __float128, so nvcc won't
# preprocess correctly any source file that include fftw3.h
# By using this trick, we make nvcc's preprocessor avoid this problem...
# We should really find something else here...
# Do accfft_gpu.h really need to include fftw3.h ?
#
cuda_add_executable(step1_gpu ${step1_SRCS} step1_gpu.cu OPTIONS "-DENABLE_GPU -D__INTEL_COMPILER")
set_target_properties(step1_gpu PROPERTIES COMPILE_FLAGS "-DENABLE_GPU")

target_link_libraries(step1_gpu
  accfft
  accfft_gpu
  ${FFTW_LIBRARIES}
)
cuda_add_cufft_to_target(step1_gpu)
