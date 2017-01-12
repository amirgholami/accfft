# optional: use parallel-netcdf to dump data
find_package(PNETCDF)
if (PNETCDF_FOUND)
  add_definitions (-DUSE_PNETCDF)
endif()

# include dirs
include_directories(${FFTW_INCLUDE_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

#
# CPU executable single precision
#

# source files list
set (step2f_SRCS
  step2f.cpp
)

add_executable(step2f ${step2f_SRCS})
target_link_libraries(step2f
  accfft
  accfft_utils
  ${FFTW_THREADS_LIB}
  ${FFTW_LIB}
  ${FFTWF_THREADS_LIB}
  ${FFTWF_LIB}
)
if (PNETCDF_FOUND)
  target_link_libraries(step2f ${PNETCDF_LIBRARIES})
endif()

#
# CPU executable double precision
#

# source files list
set (step2_SRCS
  step2.cpp
)

add_executable(step2 ${step2_SRCS})
target_link_libraries(step2
  accfft
  accfft_utils
  ${FFTW_THREADS_LIB}
  ${FFTW_LIB}
  ${FFTWF_THREADS_LIB}
  ${FFTWF_LIB}
)
if (PNETCDF_FOUND)
  target_link_libraries(step2 ${PNETCDF_LIBRARIES})
endif()

if(BUILD_GPU)
  #
  # GPU executable single precision
  #
  set(step2_gpuf_SRCS
    step2_gpuf.cpp kernelsf.cu
    )
  
  cuda_add_executable(step2_gpuf ${step2_gpuf_SRCS})
  set_target_properties(step2_gpuf PROPERTIES COMPILE_FLAGS "-DENABLE_GPU")
  target_link_libraries(step2_gpuf
    accfft_gpu
    accfft_utils
    ${FFTW_LIB}
    ${FFTW_THREADS_LIB}
    ${CUDA_LIBRARIES}
  )

  if (PNETCDF_FOUND)
    target_link_libraries(step2_gpuf ${PNETCDF_LIBRARIES})
  endif()

  cuda_add_cufft_to_target(step2_gpuf)


  #
  # GPU executable double precision
  #
  set(step2_gpu_SRCS
    step2_gpu.cpp kernels.cu
    )
  
  cuda_add_executable(step2_gpu ${step2_gpu_SRCS})
  set_target_properties(step2_gpu PROPERTIES COMPILE_FLAGS "-DENABLE_GPU")
  target_link_libraries(step2_gpu
    accfft_gpu
    accfft_utils
    ${FFTW_LIB}
    ${FFTW_THREADS_LIB}
    ${CUDA_LIBRARIES}
  )

  if (PNETCDF_FOUND)
    target_link_libraries(step2_gpu ${PNETCDF_LIBRARIES})
  endif()

  cuda_add_cufft_to_target(step2_gpu)
endif(BUILD_GPU)
