# optional: use parallel-netcdf to dump data
find_package(PNETCDF)
if (PNETCDF_FOUND)
  add_definitions (-DUSE_PNETCDF)
endif()

# include dirs
include_directories(${FFTW_INCLUDE_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# source files list
set (step4_SRCS
  step4.cpp
)

#
# CPU executable
#
add_executable(step4 ${step4_SRCS})
target_link_libraries(step4
  accfft
  accfft_utils
  ${FFTW_THREADS_LIB}
  ${FFTW_LIB}
)
if (PNETCDF_FOUND)
  target_link_libraries(step4 ${PNETCDF_LIBRARIES})
endif()

#
# GPU executable
#
if(BUILD_GPU)
  set(step4_gpu_SRCS
    step4_gpu.cpp kernels.cu
    )
  
  cuda_add_executable(step4_gpu ${step4_gpu_SRCS})
  set_target_properties(step4_gpu PROPERTIES COMPILE_FLAGS "-DENABLE_GPU")
  target_link_libraries(step4_gpu
    accfft_gpu
    accfft_utils
    ${FFTW_LIB}
    ${FFTW_THREADS_LIB}
    ${CUDA_LIBRARIES}
  )

  if (PNETCDF_FOUND)
    target_link_libraries(step4_gpu ${PNETCDF_LIBRARIES})
  endif()

  cuda_add_cufft_to_target(step4_gpu)
endif(BUILD_GPU)
