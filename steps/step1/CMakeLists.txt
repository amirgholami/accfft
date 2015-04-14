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
add_executable(step1_gpu ${step1_SRCS} step1_gpu.cpp)
set_target_properties(step1_gpu PROPERTIES COMPILE_FLAGS "-DENABLE_GPU")
target_link_libraries(step1_gpu
  accfft
  accfft_gpu
  ${FFTW_LIBRARIES}
  ${CUDA_LIBRARIES}
)
cuda_add_cufft_to_target(step1_gpu)
