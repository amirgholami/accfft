function(build_step step)
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
    set (${step}_SRCS
      ${step}.cpp
    )

    add_executable(${step} ${${step}_SRCS})
    target_link_libraries(${step}
      accfft
      accfft_utils
      ${FFTW_LIB}
      ${FFTW_THREADS_LIB}
    )
    if (PNETCDF_FOUND)
      target_link_libraries(${step} ${PNETCDF_LIBRARIES})
    endif()


    #
    # CPU executable single precision
    #

    # source files list
    set (${step}f_SRCS
      ${step}f.cpp
    )

    add_executable(${step}f ${${step}f_SRCS})
    target_link_libraries(${step}f
      accfft
      accfft_utils
      ${FFTWF_LIB}
      ${FFTWF_THREADS_LIB}
    )
    if (PNETCDF_FOUND)
      target_link_libraries(${step}f ${PNETCDF_LIBRARIES})
    endif()




    if(BUILD_GPU)
      #
      # GPU executable double precision
      #
      set(${step}_gpu_SRCS
        ${step}_gpu.cpp kernels.cu
        )
      
      cuda_add_executable(${step}_gpu ${${step}_gpu_SRCS})
      set_target_properties(${step}_gpu PROPERTIES COMPILE_FLAGS "-DENABLE_GPU")
      target_link_libraries(${step}_gpu
        accfft_gpu
        accfft_utils
        ${FFTW_LIB}
        ${FFTW_THREADS_LIB}
        ${CUDA_LIBRARIES}
        )

      if (PNETCDF_FOUND)
        target_link_libraries(${step}_gpu ${PNETCDF_LIBRARIES})
      endif()

      cuda_add_cufft_to_target(${step}_gpu)


      #
      # GPU executable single precision
      #
      set(${step}_gpuf_SRCS
        ${step}_gpuf.cpp kernelsf.cu
        )

      cuda_add_executable(${step}_gpuf ${${step}_gpuf_SRCS})
      set_target_properties(${step}_gpuf PROPERTIES COMPILE_FLAGS "-DENABLE_GPU")
      target_link_libraries(${step}_gpuf
        accfft_gpu
        accfft_utils
        ${FFTWF_LIB}
        ${FFTWF_THREADS_LIB}
        ${CUDA_LIBRARIES}
        )

      if (PNETCDF_FOUND)
        target_link_libraries(${step}_gpuf ${PNETCDF_LIBRARIES})
      endif()

      cuda_add_cufft_to_target(${step}_gpuf)
    endif(BUILD_GPU)
endfunction(build_step)

add_subdirectory(step1)
add_subdirectory(step2)
add_subdirectory(step3)
add_subdirectory(step4)
add_subdirectory(step5)
add_subdirectory(step6)
