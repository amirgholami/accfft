#######################################################################
#
#  This is a basic, generic find_package function intended to find
#  any package with the following conditions:
#
#  (1) The package has a CMake variable associated with it
#      that is a path directory pointing to the installation directory
#      of the package.  It is assumed that the libraries and include
#      files are either contiained within this directory or the
#      sub-directories lib/ or include/, respectively.
#
#  (2) The name(s) of the required library file(s) will be given to
#      the macro as a string list argument, and each will be tested in 
#      sequence.
#
#  (3) The name(s) of the required include file(s) will be given to 
#      the macro as a string list argument, and each will be tested in 
#      sequence.
#
#  (4) For a package with associated environment variable VAR, this
#      macro will define the variables:
#
#        VAR_FOUND - Boolean indicating if all files/libraries were found
#        VAR_INCLUDE_DIRS - Directory path(s) to include files
#        VAR_LIBRARIES - Full path(s) to each libraries
#
# AUTHOR:  Kevin Paul <kpaul@ucar.edu>
# DATE:    11 Feb 2014
#   
#######################################################################

function(BASIC_FIND PCKG REQ_INCS REQ_LIBS)

#If environment variable ${PCKG}_DIR is specified, 
# it has same effect as local ${PCKG}_DIR
if( (NOT ${PCKG}_DIR) AND DEFINED ENV{${PCKG}_DIR} )
  set( ${PCKG}_DIR "$ENV{${PCKG}_DIR}" )
  message(STATUS " ${PCKG}_DIR is set from environment: ${${PCKG}_DIR}")
endif()

if (NOT ${PCKG}_DIR)
  message (WARNING " Option ${PCKG}_DIR not set.  ")
else (NOT ${PCKG}_DIR)
  message (STATUS " ${PCKG}_DIR set to ${${PCKG}_DIR}")
endif (NOT ${PCKG}_DIR)

message (STATUS " Searching for package ${PCKG}...")
set (${PCKG}_FOUND FALSE PARENT_SCOPE)

# Look for each required include file
foreach(INC_FILE ${REQ_INCS})
  message (STATUS " Searching for include file: ${INC_FILE}")
  set (INC_DIR ${INC_FILE}-NOTFOUND)
  find_path(INC_DIR ${INC_FILE}
    HINTS ${${PCKG}_DIR} ${${PCKG}_DIR}/include)
  if (EXISTS ${INC_DIR}/${INC_FILE})
    message (STATUS " Found include file ${INC_FILE} in ${INC_DIR} required by ${PCKG}")
    set (${PCKG}_INCLUDE_DIRS ${${PCKG}_INCLUDE_DIRS} ${INC_DIR} PARENT_SCOPE)
  else ()
    message (WARNING " Failed to find include file ${INC_FILE} required by ${PCKG}")
  endif ()
endforeach()

# Look for each required library
foreach(LIB_NAME ${REQ_LIBS})
  message (STATUS " Searching for library: ${LIB_NAME}")
  set (LIB ${LIB_NAME}-NOTFOUND)
  find_library(LIB NAMES "lib${LIB_NAME}.a" "${LIB_NAME}"
    HINTS ${${PCKG}_DIR} ${${PCKG}_DIR}/lib)
  if (EXISTS ${LIB})
    message (STATUS " Found library at ${LIB} required by ${PCKG}")
    set (${PCKG}_LIBRARIES ${${PCKG}_LIBRARIES} ${LIB} PARENT_SCOPE)
    set (${PCKG}_FOUND TRUE PARENT_SCOPE)
    set (${PCKG}_FOUND TRUE )
  else ()
    message (WARNING " Failed to find library lib${LIB_NAME} required by ${PCKG}")
    set (${PCKG}_FOUND FALSE PARENT_SCOPE)
    set (${PCKG}_FOUND FALSE )
  endif ()
endforeach()

# If we made it this far, then we call the package "FOUND"
if(${PCKG}_FOUND)
  message (STATUS "All required include files and libraries found.")
else()
  message ("WARNING! ${PCKG} not found.")
endif()

endfunction(BASIC_FIND)

