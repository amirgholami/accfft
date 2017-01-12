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
set (step6_SRCS
  step6.cpp
)

add_executable(step6 ${step6_SRCS})
target_link_libraries(step6
  accfft
  accfft_utils
  ${FFTW_THREADS_LIB}
  ${FFTW_LIB}
  ${FFTWF_THREADS_LIB}
  ${FFTWF_LIB}
)
if (PNETCDF_FOUND)
  target_link_libraries(step6 ${PNETCDF_LIBRARIES})
endif()
