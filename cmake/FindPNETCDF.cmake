#######################################################################
#
#  CMake Module that uses the BASIC_FIND macro to find the PIO
#  library and include directory
#
#######################################################################

include(basicFind)
BASIC_FIND (PNETCDF "pnetcdf.h" "pnetcdf")
