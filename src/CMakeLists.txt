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

IF (NOT CMAKE_INSTALL_LIBDIR)
  SET(CMAKE_INSTALL_LIBDIR lib)
ENDIF()
MESSAGE(STATUS "CMAKE_INSTALL_LIBDIR is set ${CMAKE_INSTALL_LIBDIR}")

SET(TCHEM_INSTALL_LIB_PATH ${CMAKE_INSTALL_LIBDIR})
SET(TCHEM_INSTALL_INCLUDE_PATH include/tchem)
SET(TCHEM_INSTALL_TEST_PATH    unit-test)
SET(TCHEM_INSTALL_EXAMPLE_PATH example)

#
# Options : use TCHEM prefix
#
OPTION(TCHEM_ENABLE_TEST "Flag to enable unit tests" OFF)
OPTION(TCHEM_ENABLE_EXAMPLE "Flag to enable unit examples" ON)
OPTION(TCHEM_ENABLE_VERBOSE "Flag to enable TChem verbose flag" OFF)
OPTION(TCHEM_ENABLE_DEBUG "Flag to enable TChem debug flag" OFF)
OPTION(TCHEM_ENABLE_PYTHON "Flag to enable python binding" OFF)
OPTION(TCHEM_ENABLE_FORTRAN "Flag to enable fortran binding" OFF)

OPTION(TCHEM_ENABLE_NO_NEG_CONCENTRATION_ARBITRARY_ORDER "Flag to enable no negative values of concentration only in arbitrary-order reactions" OFF)
OPTION(TCHEM_ENABLE_PROBLEMS_NUMERICAL_JACOBIAN "Flag to enable numerical jacobian" OFF)
OPTION(TCHEM_ENABLE_NEWTONSOLVER_USE_WRMS_NORMS "Flag to enable newton solver to use wrms norms" ON)
OPTION(TCHEM_ENABLE_TRBDF2_USE_WRMS_NORMS "Flag to enable time integrator to use wrms norms" ON)
OPTION(TCHEM_ENABLE_SACADO_JACOBIAN_PLUG_FLOW_REACTOR "Flag to enable sacado jacobian for plug flow reactor" OFF)
OPTION(TCHEM_ENABLE_SACADO_JACOBIAN_IGNITION_ZERO_D_REACTOR "Flag to enable sacado jacobian for ignition zero D reactor" OFF)
OPTION(TCHEM_ENABLE_SACADO_JACOBIAN_CONSTANT_VOLUME_IGNITION_REACTOR "Flag to enable sacado jacobian for constant volume ignition reactor" OFF)
OPTION(TCHEM_ENABLE_SACADO_JACOBIAN_TRANSIENT_CONT_STIRRED_TANK_REACTOR "Flag to enable sacado jacobian for  transient cont stirred tank reactor" OFF)
OPTION(TCHEM_ENABLE_NUMERICAL_JACOBIAN_IGNITION_ZERO_D_REACTOR "Flag to enable numerical jacobian for ignition zero D reactor" OFF)
OPTION(TCHEM_ENABLE_SACADO_JACOBIAN_ATMOSPHERIC_CHEMISTRY "Flag to enable sacado jacobian for atmospheric chemistry" OFF)

OPTION(TCHEM_CSTR_ALLOW_NEGATIVE_MASS_FLOW_OUTLET "Flag to allow negative values for mass flow at outlet in CSTR " OFF)

OPTION(KOKKOS_INSTALL_PATH "Path to Kokkos installation")
OPTION(TINES_INSTALL_PATH "Path to Tines installation")
OPTION(GTEST_INSTALL_PATH "Path to gtest installation")

INCLUDE_DIRECTORIES(${TChem_BINARY_DIR})

# Fortran
IF (TCHEM_ENABLE_FORTRAN)
  ENABLE_LANGUAGE(Fortran)
ENDIF()

# TChem
IF (TCHEM_ENABLE_PYTHON)
  SET(TCHEM_ENABLE_SHARED_BUILD ON)
ELSE()
  SET(TCHEM_ENABLE_SHARED_BUILD OFF)
ENDIF()


# TChem link libraries for tests
SET(TCHEM_LINK_LIBRARIES tchem)

#
# Kokkos
#
IF (KOKKOS_INSTALL_PATH)
  MESSAGE(STATUS "TChem attempts to use Kokkos installed at ${KOKKOS_INSTALL_PATH}")
  FIND_PACKAGE(Kokkos REQUIRED)
  SET(TCHEM_INTERNAL_KOKKOS_TARGET "Kokkos::kokkos")
  SET(TCHEM_ENABLE_TPL_KOKKOS ON CACHE BOOL "Enable Kokkos")
  LIST(APPEND TCHEM_LINK_LIBRARIES ${TCHEM_INTERNAL_KOKKOS_TARGET})
ELSE()
  MESSAGE(FATAL_ERROR "-- Kokkos install path is not provided")
ENDIF()

#
# Tines
#
IF (TINES_INSTALL_PATH)
  MESSAGE(STATUS "TChem attempts to use Tines installed at ${TINES_INSTALL_PATH}")
  FIND_PACKAGE(Tines REQUIRED)
  SET(TCHEM_INTERNAL_TINES_TARGET "Tines::tines")
  SET(TCHEM_ENABLE_TPL_TINES ON CACHE BOOL "Enable Tines")
  LIST(APPEND TCHEM_LINK_LIBRARIES ${TCHEM_INTERNAL_TINES_TARGET})
ELSE()
  MESSAGE(FATAL_ERROR "-- Tines install path is not provided")
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
    MESSAGE(FATAL_ERROR "-- Test is enabled but Gtest install path is not provided")
  ENDIF()
  LIST(APPEND TCHEM_LINK_LIBRARIES ${TCHEM_INTERNAL_GTEST_TARGET})
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
IF (TCHEM_ENABLE_PYTHON)
  ADD_SUBDIRECTORY(pytchem)
ENDIF()
