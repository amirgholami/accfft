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
set (step3f_SRCS
  step3f.cpp
)

add_executable(step3f ${step3f_SRCS})
target_link_libraries(step3f
  accfft
  accfft_utils
  ${FFTW_THREADS_LIB}
  ${FFTW_LIB}
  ${FFTWF_THREADS_LIB}
  ${FFTWF_LIB}
)
if (PNETCDF_FOUND)
  target_link_libraries(step3f ${PNETCDF_LIBRARIES})
endif()

#
# CPU executable double precision
#
# source files list
set (step3_SRCS
  step3.cpp
)

add_executable(step3 ${step3_SRCS})
target_link_libraries(step3
  accfft
  accfft_utils
  ${FFTW_THREADS_LIB}
  ${FFTW_LIB}
  ${FFTWF_THREADS_LIB}
  ${FFTWF_LIB}
)
if (PNETCDF_FOUND)
  target_link_libraries(step3 ${PNETCDF_LIBRARIES})
endif()

if(BUILD_GPU)
  #
  # GPU executable single precision
  #
  set(step3_gpuf_SRCS
    step3_gpuf.cpp
    )
  
  cuda_add_executable(step3_gpuf ${step3_gpuf_SRCS})
  set_target_properties(step3_gpuf PROPERTIES COMPILE_FLAGS "-DENABLE_GPU")
  target_link_libraries(step3_gpuf
    accfft_gpu
    accfft_utils
    ${FFTW_LIB}
    ${FFTW_THREADS_LIB}
    ${CUDA_LIBRARIES}
  )

  if (PNETCDF_FOUND)
    target_link_libraries(step3_gpuf ${PNETCDF_LIBRARIES})
  endif()

  cuda_add_cufft_to_target(step3_gpuf)


  #
  # GPU executable double precision
  #
  set(step3_gpu_SRCS
    step3_gpu.cpp
    )
  
  cuda_add_executable(step3_gpu ${step3_gpu_SRCS})
  set_target_properties(step3_gpu PROPERTIES COMPILE_FLAGS "-DENABLE_GPU")
  target_link_libraries(step3_gpu
    accfft_gpu
    accfft_utils
    ${FFTW_LIB}
    ${FFTW_THREADS_LIB}
    ${CUDA_LIBRARIES}
  )

  if (PNETCDF_FOUND)
    target_link_libraries(step3_gpu ${PNETCDF_LIBRARIES})
  endif()

  cuda_add_cufft_to_target(step3_gpu)
endif(BUILD_GPU)
