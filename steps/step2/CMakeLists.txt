# optional: use parallel-netcdf to dump data
find_package(PNETCDF)
if (PNETCDF_FOUND)
  add_definitions (-DUSE_PNETCDF)
endif()

# include dirs
include_directories(${FFTW_INCLUDE_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# source files list
set (step2_SRCS
  step2.cpp
)

#
# CPU executable
#
add_executable(step2 ${step2_SRCS})
target_link_libraries(step2
  accfft
  accfft_utils
  ${FFTW_LIBRARIES}
)
if (PNETCDF_FOUND)
  target_link_libraries(step2 ${PNETCDF_LIBRARIES})
endif()

#
# GPU executable
#
if(BUILD_GPU)
  set(step2_gpu_SRCS
    step2_gpu.cpp kernels.cu
    )
  
  cuda_add_executable(step2_gpu ${step2_gpu_SRCS})
  set_target_properties(step2_gpu PROPERTIES COMPILE_FLAGS "-DENABLE_GPU")
  target_link_libraries(step2_gpu
    accfft_gpu
    accfft_utils
    ${FFTW_LIBRARIES}
    ${CUDA_LIBRARIES}
  )

  if (PNETCDF_FOUND)
    target_link_libraries(step2_gpu ${PNETCDF_LIBRARIES})
  endif()

  cuda_add_cufft_to_target(step2_gpu)
endif(BUILD_GPU)
