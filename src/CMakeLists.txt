# Kokkos requires 3.3
CMAKE_MINIMUM_REQUIRED(VERSION 3.3)

# Project TChem
PROJECT(TChem)

# CMAKE setup
SET(CMAKE_VERBOSE_MAKEFILE ON) # see all output
SET(CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR})

# Define BUILD_TESTING in cmake
INCLUDE(CTest)

# Installation
IF (CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  SET(CMAKE_INSTALL_PREFIX "${PROJECT_BINARY_DIR}" CACHE PATH "FOO install prefix" FORCE)
ENDIF()

SET(TCHEM_INSTALL_INCLUDE_PATH include/tchem)
SET(TCHEM_INSTALL_TEST_PATH    unit-test)
SET(TCHEM_INSTALL_EXAMPLE_PATH example)

#
# Options : use TCHEM prefix
#
OPTION(TCHEM_ENABLE_KOKKOS "Flag to enable Kokkos" OFF)
OPTION(TCHEM_ENABLE_KOKKOSKERNELS "Flag to enable KokkosKernels" OFF)
OPTION(TCHEM_ENABLE_MKL "Flag to enable MKL" OFF)
OPTION(TCHEM_ENABLE_TEST "Flag to enable unit tests" OFF)
OPTION(TCHEM_ENABLE_EXAMPLE "Flag to enable unit examples" ON)
OPTION(TCHEM_ENABLE_VERBOSE "Flag to enable TChem verbose flag" OFF)
OPTION(TCHEM_ENABLE_DEBUG "Flag to enable TChem debug flag" OFF)
OPTION(TCHEM_ENABLE_PROBLEMS_NUMERICAL_JACOBIAN "Flag to enable numerical jacobian" OFF)
OPTION(TCHEM_ENABLE_NEWTONSOLVER_USE_WRMS_NORMS "Flag to enable newton solver to use wrms norms" ON)
OPTION(TCHEM_ENABLE_TRBDF2_USE_WRMS_NORMS "Flag to enable time integrator to use wrms norms" ON)

OPTION(TCHEM_ENABLE_PROBLEM_DAE_CSTR "Flag to enable DAE solver in CSTR" OFF)

OPTION(LAPACKE_INSTALL_PATH "Path to LAPACKE installation")
OPTION(OPENBLAS_INSTALL_PATH "Path to OpenBLAS installation")
OPTION(KOKKOS_INSTALL_PATH "Path to Kokkos installation")
OPTION(KOKKOSKERNELS_INSTALL_PATH "Path to KokkosKernels installation")
OPTION(GTEST_INSTALL_PATH "Path to gtest installation")

# option to manually set lib install path
OPTION(TCHEM_INSTALL_LIB_PATH "Installation directory for libraries" lib)

INCLUDE_DIRECTORIES(${TChem_BINARY_DIR})

# TChem link libraries for tests
SET(TCHEM_LINK_LIBRARIES tchemcore)

#
# Kokkos
#
IF (TCHEM_ENABLE_KOKKOS)
  IF (KOKKOS_INSTALL_PATH)
    #
    # Kokkos is installed separately
    #
    MESSAGE(STATUS "TChem uses Kokkos installed at ${KOKKOS_INSTALL_PATH}")
    FIND_PACKAGE(Kokkos REQUIRED)
    SET(TCHEM_INTERNAL_KOKKOS_TARGET "Kokkos::kokkos")
  ELSE()
    #
    # Submodule Kokkos (includes, cxx flags and lib)
    #
    MESSAGE(STATUS "TChem uses submodule of Kokkos located at external/kokkos")

    # Serial space is always required
    IF (NOT Kokkos_ENABLE_SERIAL)
      MESSAGE(STATUS "TChem always requires Kokkos::Serial and forces to set -DKokkos_ENABLE_SERIAL=ON")
      SET(Kokkos_ENABLE_SERIAL ON CACHE BOOL "" FORCE)
    ENDIF()

    # Turn on Kokkos debugging flags
    IF (TCHEM_ENABLE_DEBUG)
      MESSAGE(STATUS "TChem enables DEBUG and forces to set Kokkos debug options i.e., -DKokkos_ENABLE_DEBUG=ON and -DKokkos_ENABLE_DEBUG_BOUNDS_CHECK=ON")
      SET(Kokkos_ENABLE_DEBUG ON CACHE BOOL "" FORCE)
      SET(Kokkos_ENABLE_DEBUG_BOUNDS_CHECK ON CACHE BOOL "" FORCE)
    ENDIF()

    # When Kokkos is used as a submodule and enalbes Kokkos, enable lambda and disable uvm.
    IF (Kokkos_ENABLE_CUDA)
      MESSAGE(STATUS "-- Cuda is enabled and forces to enable cuda lambda and disable cuda uvm i.e., -DKokkos_ENABLE_CUDA_LAMBDA=ON and -DKokkos_ENABLE_CUDA_UVM=OFF.")
      SET(Kokkos_ENABLE_CUDA_LAMBDA ON CACHE BOOL "Enable LAMBDAs for CUDA")
      SET(Kokkos_ENABLE_CUDA_UVM OFF CACHE BOOL "Disable UVMs for CUDA")
    ENDIF()

    ADD_SUBDIRECTORY(external/kokkos)
    MESSAGE(STATUS "KOKKOS_INCLUDE = ${Kokkos_INCLUDE_DIRS_RET}")
    INCLUDE_DIRECTORIES(${Kokkos_INCLUDE_DIRS_RET})
    SET(TCHEM_INTERNAL_KOKKOS_TARGET "kokkos")
  ENDIF()
  SET(TCHEM_ENABLE_TPL_KOKKOS ON CACHE BOOL "Enable Kokkos")
  LIST(APPEND TCHEM_LINK_LIBRARIES ${TCHEM_INTERNAL_KOKKOS_TARGET})
ENDIF()

#
# KokkosKernels
#
IF (TCHEM_ENABLE_KOKKOSKERNELS)
  IF (KOKKOSKERNELS_INSTALL_PATH)
    #
    # KokkosKernels is installed separately
    #
    MESSAGE(STATUS "TChem uses KokkosKernels installed at ${KOKKOSKERNELS_INSTALL_PATH}")
    FIND_PACKAGE(KokkosKernels REQUIRED)
    SET(TCHEM_INTERNAL_KOKKOSKERNELS_TARGET "Kokkos::kokkoskernels")
  ELSE()
    #
    # Submodule KokkosKernels (includes, cxx flags and lib)
    #
    MESSAGE(STATUS "TChem uses submodule of KokkosKernels located at external/kokkoskernels")

    ADD_SUBDIRECTORY(external/kokkoskernels)
    SET(TCHEM_INTERNAL_KOKKOSKERNELS_TARGET "kokkoskernels")
  ENDIF()
  SET(TCHEM_ENABLE_TPL_KOKKOSKERNELS ON CACHE BOOL "Enable KokkosKernels")
  LIST(APPEND TCHEM_LINK_LIBRARIES ${TCHEM_INTERNAL_KOKKOSKERNELS_TARGET})
ENDIF()

#
# gtest
#
IF (TCHEM_ENABLE_TEST)
  IF (GTEST_INSTALL_PATH)
    # Interface installed gtest
    MESSAGE("-- TChem uses gtest installed at ${GTEST_INSTALL_PATH}")
    FIND_PACKAGE(Gtest REQUIRED)
    SET(TCHEM_INTERNAL_GTEST_TARGET "GTest::gtest_main")
  ELSE()
    # Process submodule Kokkos (includes, cxx flags and lib)
    MESSAGE("-- TChem uses submodule of gtest located at external/gtest")
    ADD_SUBDIRECTORY(external/gtest)
    SET(TCHEM_INTERNAL_GTEST_TARGET "gtest_main")
  ENDIF()
  LIST(APPEND TCHEM_LINK_LIBRARIES ${TCHEM_INTERNAL_GTEST_TARGET})
ENDIF()

#
# OpenBLAS
#
IF (OPENBLAS_INSTALL_PATH)
  FIND_PACKAGE (OpenBLAS)
  IF (OPENBLAS_FOUND)
    SET(TCHEM_ENABLE_TPL_OPENBLAS ON CACHE BOOL "Enable OpenBLAS")
    SET(TCHEM_INTERNAL_OPENBLAS_TARGET "openblas")
    IF (OPENBLAS_FOUND_CBLAS_HEADER)
      SET(TCHEM_ENABLE_TPL_OPENBLAS_CBLAS_HEADER ON CACHE BOOL "Enable OpenBLAS from MacPort")
      SET(TCHEM_ENABLE_TPL_LAPACKE ON CACHE BOOL "Enable LAPACKE from MacPort")
    ENDIF()
    LIST(APPEND TCHEM_LINK_LIBRARIES openblas)
    IF (LAPACKE_FOUND)
      SET(TCHEM_ENABLE_TPL_LAPACKE ON CACHE BOOL "Enable LAPACKE from OpenBlas")
    ENDIF()
  ELSE()
    MESSAGE(FATAL_ERROR "-- OpenBLAS is not found at ${OPENBLAS_INSTALL_PATH}")
  ENDIF()
ENDIF()

#
# LAPACKE
#
IF (LAPACKE_INSTALL_PATH)
  IF (NOT TCHEM_ENABLE_TPL_LAPACKE)
    FIND_PACKAGE (LAPACKE)
    IF (LAPACKE_FOUND)
      SET(TCHEM_ENABLE_TPL_LAPACKE ON CACHE BOOL "Enable LAPACKE")
      SET(TCHEM_INTERNAL_LAPACKE_TARGET "lapacke")
      LIST(APPEND TCHEM_LINK_LIBRARIES lapacke)
    ELSE()
      MESSAGE(FATAL_ERROR "-- LAPACKE is not found at ${LAPACKE_INSTALL_PATH}")
    ENDIF()
  ELSE()
    MESSAGE(WARNING "-- LAPACKE is enabled from MacPort OpenBlas")
  ENDIF()
ENDIF()

#
# MKL
#
IF (TCHEM_ENABLE_MKL)
  IF (CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
    SET(TCHEM_ENABLE_TPL_MKL ON CACHE BOOL "Enable MKL")
    SET(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -mkl")
    SET(CMAKE_EXE_LINKER_FLAGS  "${CMAKE_EXE_LINKER_FLAGS} -mkl")
  ELSE()
    MESSAGE(FATAL_ERROR "-- MKL is not enabled (use intel compiler supporting -mkl)")
  ENDIF()
ENDIF()

#
# Linking libraries used in example and test
#
MESSAGE("-- TCHEM_LINK_LIBRARIES = ${TCHEM_LINK_LIBRARIES}")

#
# Define configure-related preprocessor macros
#
CONFIGURE_FILE (
  ${TChem_SOURCE_DIR}/core/TChem_Config.hpp.in
  ${TChem_BINARY_DIR}/TChem_Config.hpp
  )
INSTALL(FILES ${CMAKE_CURRENT_BINARY_DIR}/TChem_Config.hpp
  DESTINATION "${CMAKE_INSTALL_PREFIX}/${TCHEM_INSTALL_INCLUDE_PATH}")

#
# Process subdirectories
#

ADD_SUBDIRECTORY (core)
IF (TCHEM_ENABLE_TEST)
  ADD_SUBDIRECTORY (unit-test)
ENDIF()
IF (TCHEM_ENABLE_EXAMPLE)
  ADD_SUBDIRECTORY (example)
ENDIF()
