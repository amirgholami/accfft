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
set (step5f_SRCS
  step5f.cpp
)

add_executable(step5f ${step5f_SRCS})
target_link_libraries(step5f
  accfft
  accfft_utils
  ${FFTW_THREADS_LIB}
  ${FFTW_LIB}
  ${FFTWF_THREADS_LIB}
  ${FFTWF_LIB}
)
if (PNETCDF_FOUND)
  target_link_libraries(step5f ${PNETCDF_LIBRARIES})
endif()

#
# CPU executable double precision
#
# source files list
set (step5_SRCS
  step5.cpp
)

add_executable(step5 ${step5_SRCS})
target_link_libraries(step5
  accfft
  accfft_utils
  ${FFTW_THREADS_LIB}
  ${FFTW_LIB}
  ${FFTWF_THREADS_LIB}
  ${FFTWF_LIB}
)
if (PNETCDF_FOUND)
  target_link_libraries(step5 ${PNETCDF_LIBRARIES})
endif()

if(BUILD_GPU)
  #
  # GPU executable single precision
  #
  set(step5_gpuf_SRCS
    step5_gpuf.cpp
    )
  
  cuda_add_executable(step5_gpuf ${step5_gpuf_SRCS})
  set_target_properties(step5_gpuf PROPERTIES COMPILE_FLAGS "-DENABLE_GPU")
  target_link_libraries(step5_gpuf
    accfft_gpu
    accfft_utils_gpu
    ${FFTW_LIB}
    ${FFTW_THREADS_LIB}
    ${CUDA_LIBRARIES}
  )

  if (PNETCDF_FOUND)
    target_link_libraries(step5_gpuf ${PNETCDF_LIBRARIES})
  endif()

  cuda_add_cufft_to_target(step5_gpuf)


  #
  # GPU executable double precision
  #
  set(step5_gpu_SRCS
    step5_gpu.cpp
    )
  
  cuda_add_executable(step5_gpu ${step5_gpu_SRCS})
  set_target_properties(step5_gpu PROPERTIES COMPILE_FLAGS "-DENABLE_GPU")
  target_link_libraries(step5_gpu
    accfft_gpu
    accfft_utils_gpu
    ${FFTW_LIB}
    ${FFTW_THREADS_LIB}
    ${CUDA_LIBRARIES}
  )

  if (PNETCDF_FOUND)
    target_link_libraries(step5_gpu ${PNETCDF_LIBRARIES})
  endif()

  cuda_add_cufft_to_target(step5_gpu)
endif(BUILD_GPU)
