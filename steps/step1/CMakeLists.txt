# optional: use parallel-netcdf to dump data
find_package(PNETCDF)
if (PNETCDF_FOUND)
  add_definitions (-DUSE_PNETCDF)
endif()

# include dirs
include_directories(${FFTW_INCLUDE_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR})


#
# CPU executable double precision
#

# source files list
set (step1_SRCS
  step1.cpp
)

add_executable(step1 ${step1_SRCS})
target_link_libraries(step1
  accfft
  accfft_utils
  ${FFTW_THREADS_LIB}
  ${FFTW_LIB}
  ${FFTWF_THREADS_LIB}
  ${FFTWF_LIB}
)
if (PNETCDF_FOUND)
  target_link_libraries(step1 ${PNETCDF_LIBRARIES})
endif()


#
# CPU executable single precision
#

# source files list
set (step1f_SRCS
  step1f.cpp
)

add_executable(step1f ${step1f_SRCS})
target_link_libraries(step1f
  accfft
  accfft_utils
  ${FFTW_THREADS_LIB}
  ${FFTW_LIB}
  ${FFTWF_THREADS_LIB}
  ${FFTWF_LIB}
)
if (PNETCDF_FOUND)
  target_link_libraries(step1f ${PNETCDF_LIBRARIES})
endif()




if(BUILD_GPU)
  #
  # GPU executable double precision
  #
  set(step1_gpu_SRCS
    step1_gpu.cpp kernels.cu
    )
  
  cuda_add_executable(step1_gpu ${step1_gpu_SRCS})
  set_target_properties(step1_gpu PROPERTIES COMPILE_FLAGS "-DENABLE_GPU")
  target_link_libraries(step1_gpu
    accfft_gpu
    accfft_utils
    ${FFTW_LIB}
    ${FFTW_THREADS_LIB}
    ${CUDA_LIBRARIES}
    )

  if (PNETCDF_FOUND)
    target_link_libraries(step1_gpu ${PNETCDF_LIBRARIES})
  endif()

  cuda_add_cufft_to_target(step1_gpu)


  #
  # GPU executable single precision
  #
  set(step1_gpuf_SRCS
    step1_gpuf.cpp kernelsf.cu
    )

  cuda_add_executable(step1_gpuf ${step1_gpuf_SRCS})
  set_target_properties(step1_gpuf PROPERTIES COMPILE_FLAGS "-DENABLE_GPU")
  target_link_libraries(step1_gpuf
    accfft_gpu
    accfft_utils
    ${FFTW_LIB}
    ${FFTW_THREADS_LIB}
    ${CUDA_LIBRARIES}
    )

  if (PNETCDF_FOUND)
    target_link_libraries(step1_gpuf ${PNETCDF_LIBRARIES})
  endif()

  cuda_add_cufft_to_target(step1_gpuf)



endif(BUILD_GPU)
