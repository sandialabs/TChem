# Check MKL - this is not working; I do not know why
IF (CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
  ADD_LIBRARY(mkl UNKNOWN IMPORTED)
  SET_TARGET_PROPERTIES(mkl PROPERTIES 
    INTERFACE_COMPILE_OPTIONS "-mkl"
    INTERFACE_LINK_OPTIONS "-mkl")
  SET(MKL_FOUND ON)    
ELSE()
  MESSAGE(FATAL_ERROR "-- MKL is not enabled as the compiler is not Intel")
ENDIF()