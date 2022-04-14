#!/bin/bash

#=======================================================================================
#=======================================================================================
# User configuration  -- begin

# macbook Pro Mojave using macport to install compilers:
MY_CC=/opt/local/bin/gcc-mp-10
MY_CXX=/opt/local/bin/g++-mp-10
MY_FC=/opt/local/bin/gfortran-mp-10
JFLAG="-j 4"
CUDA="OFF"

# macbook air:
#MY_CC=/opt/local/bin/gcc
#MY_CXX=/opt/local/bin/g++
#MY_FC=/opt/local/bin/gfortran
#JFLAG="-j 4"
#CUDA="OFF"

# ubuntu + NVIDIA GPU:
#MY_CC=/usr/bin/gcc
#MY_CXX=/usr/bin/g++
#MY_FC=/usr/bin/gfortran
#JFLAG="-j 4"
#CUDA="ON"

REPO_BASE=${PWD}/repos
BUILD_BASE=${PWD}/build
INSTALL_BASE=${PWD}/install

# Flag to create and install TChem main executable that uses/produces inputs/ouputs in json-format file.
# set ON to enable MAIN code, but it requires Boost 1.75.
TCHEM_ENABLE_MAIN=ON


# User configuration  -- end
#=======================================================================================

#=======================================================================================
# OpenBLAS
# nb. to make sure this openblas gets used when running outside this script
# add in your .bashrc/.bash_profile :
# on mac need this
#    export LIBRARY_PATH="${LIBRARY_PATH}:${OPENBLAS_INSTALL_PATH}/lib"
# on linux need this
#    export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${OPENBLAS_INSTALL_PATH}/lib"

get_openblas (){
echo "get OpenBLAS:"
if [ -d "${OPENBLAS_REPOSITORY_PATH}" ] && [ "$(ls -A ${OPENBLAS_REPOSITORY_PATH})" ]; then
  echo "${OPENBLAS_REPOSITORY_PATH} exists and is not empty ... aborting clone"; return
fi
git clone https://github.com/xianyi/OpenBLAS          ${OPENBLAS_REPOSITORY_PATH}
}

clean_openblas (){
echo "Cleaning OpenBLAS:"
cd ${OPENBLAS_REPOSITORY_PATH}
make clean
}

build_openblas (){
echo "Building OpenBLAS:"
cd ${OPENBLAS_REPOSITORY_PATH}
make CC=${MY_CC} FC=${MY_FC} HOSTCC=${MY_CC}
}

install_openblas (){
echo "Installing OpenBLAS:"
cd ${OPENBLAS_REPOSITORY_PATH}
make CC=${MY_CC} FC=${MY_FC} HOSTCC=${MY_CC} PREFIX=${OPENBLAS_INSTALL_PATH} install
}
#=======================================================================================

#=======================================================================================
# kokkos
get_kokkos (){
echo "get kokkos:"
if [ -d "${KOKKOS_REPOSITORY_PATH}" ] && [ "$(ls -A ${KOKKOS_REPOSITORY_PATH})" ]; then
  echo "${KOKKOS_REPOSITORY_PATH} exists and is not empty ... aborting clone"; return
fi
git clone https://github.com/kokkos/kokkos            ${KOKKOS_REPOSITORY_PATH}
}

build_install_kokkos(){
echo "Building kokkos:"
mkdir ${KOKKOS_BUILD_PATH}
mkdir ${KOKKOS_INSTALL_PATH}
cd ${KOKKOS_BUILD_PATH}
# kokkos install bin directory is not yet created
cmake \
    -D CMAKE_INSTALL_PREFIX=${KOKKOS_INSTALL_PATH} \
    -D CMAKE_CXX_COMPILER=${KOKKOS_CXX_REPO_COMPILER} \
    -D CMAKE_CXX_FLAGS="-fopenmp -g" \
    -D Kokkos_ENABLE_SERIAL=ON \
    -D Kokkos_ENABLE_OPENMP=ON \
    -D Kokkos_ENABLE_CUDA=${CUDA} \
    -D Kokkos_ENABLE_DEPRECATED_CODE=OFF \
    -D Kokkos_ENABLE_CUDA_CONSTEXPR=${CUDA} \
    -D Kokkos_ENABLE_CUDA_LAMBDA=${CUDA} \
    ${KOKKOS_REPOSITORY_PATH}
make ${JFLAG} install
}
#=======================================================================================

#=======================================================================================
# gtest
get_gtest (){
echo "get gtest:"
if [ -d "${GTEST_REPOSITORY_PATH}" ] && [ "$(ls -A ${GTEST_REPOSITORY_PATH})" ]; then
  echo "${GTEST_REPOSITORY_PATH} exists and is not empty ... aborting clone"; return
fi
git clone https://github.com/google/googletest.git ${GTEST_REPOSITORY_PATH}
}

build_install_gtest(){
echo "Building gtest:"
mkdir ${GTEST_BUILD_PATH}
mkdir ${GTEST_INSTALL_PATH}
cd ${GTEST_BUILD_PATH}
cmake \
    -D CMAKE_INSTALL_PREFIX="${GTEST_INSTALL_PATH}" \
    -D CMAKE_CXX_COMPILER="${MY_CXX}"  \
    ${GTEST_REPOSITORY_PATH}
make ${JFLAG} install
}
#=======================================================================================

#=======================================================================================
# Tines
get_tines (){
echo "get Tines:"
if [ -d "${TINES_REPOSITORY_PATH}" ] && [ "$(ls -A ${TINES_REPOSITORY_PATH})" ]; then
  echo "${TINES_REPOSITORY_PATH} exists and is not empty ... aborting clone"; return
fi
git clone https://github.com/sandialabs/Tines         ${TINES_REPOSITORY_PATH}
}

build_install_tines(){
echo "Building tines:"
mkdir ${TINES_BUILD_PATH}
mkdir ${TINES_INSTALL_PATH}
cd ${TINES_BUILD_PATH}
cmake \
    -D CMAKE_INSTALL_PREFIX="${TINES_INSTALL_PATH}" \
    -D CMAKE_CXX_COMPILER="${KOKKOS_CXX_COMPILER}" \
    -D CMAKE_CXX_FLAGS="-g" \
    -D CMAKE_C_COMPILER="${MY_CC}" \
    -D CMAKE_EXE_LINKER_FLAGS="-lgfortran" \
    -D TINES_ENABLE_DEBUG=OFF \
    -D TINES_ENABLE_VERBOSE=OFF \
    -D TINES_ENABLE_TEST=ON \
    -D TINES_ENABLE_EXAMPLE=ON \
    -D YAML_INSTALL_PATH="${YAML_INSTALL_PATH}" \
    -D KOKKOS_INSTALL_PATH="${KOKKOS_INSTALL_PATH}" \
    -D GTEST_INSTALL_PATH="${GTEST_INSTALL_PATH}" \
    -D OPENBLAS_INSTALL_PATH="${OPENBLAS_INSTALL_PATH}" \
    ${TINES_REPOSITORY_PATH}/src
make ${JFLAG} install
# if using e.g. yum or apt-get installed openblas, may need this:
# Not needed when using above installed openblas and macport version on osx
#    -D LAPACKE_INSTALL_PATH="${LAPACKE_INSTALL_PATH}" \
}
#=======================================================================================

#=======================================================================================
# TChem
get_tchem (){
echo "get TChem:"
if [ -d "${TCHEM_REPOSITORY_PATH}" ] && [ "$(ls -A ${TCHEM_REPOSITORY_PATH})" ]; then
  echo "${TCHEM_REPOSITORY_PATH} exists and is not empty ... aborting clone"; return
fi
git clone https://github.com/sandialabs/TChem         ${TCHEM_REPOSITORY_PATH}
}

build_install_tchem(){
echo "Building TChem:"
mkdir ${TCHEM_BUILD_PATH}
mkdir ${TCHEM_INSTALL_PATH}
cd ${TCHEM_BUILD_PATH}
cmake \
    -D CMAKE_INSTALL_PREFIX="${TCHEM_INSTALL_PATH}" \
    -D CMAKE_CXX_COMPILER="${KOKKOS_CXX_COMPILER}" \
    -D CMAKE_C_COMPILER="${MY_CC}" \
    -D CMAKE_EXE_LINKER_FLAGS="-lgfortran" \
    -D CMAKE_BUILD_TYPE=RELEASE \
    -D TCHEM_ENABLE_VERBOSE=OFF \
    -D TCHEM_ENABLE_TEST=ON \
    -D TCHEM_ENABLE_EXAMPLE=ON \
    -D TCHEM_ENABLE_PYTHON=OFF \
    -D TCHEM_ENABLE_MAIN=${TCHEM_ENABLE_MAIN} \
    -D TCHEM_ENABLE_SACADO_JACOBIAN_TRANSIENT_CONT_STIRRED_TANK_REACTOR=OFF \
    -D TCHEM_ENABLE_NUMERICAL_JACOBIAN_IGNITION_ZERO_D_REACTOR=OFF \
    -D TCHEM_ENABLE_SACADO_JACOBIAN_CONSTANT_VOLUME_IGNITION_REACTOR=OFF \
    -D TCHEM_ENABLE_SACADO_JACOBIAN_IGNITION_ZERO_D_REACTOR=OFF \
    -D TCHEM_ENABLE_SACADO_JACOBIAN_PLUG_FLOW_REACTOR=OFF \
    -D KOKKOS_INSTALL_PATH="${KOKKOS_INSTALL_PATH}" \
    -D TINES_INSTALL_PATH="${TINES_INSTALL_PATH}" \
    -D GTEST_INSTALL_PATH="${GTEST_INSTALL_PATH}" \
    ${TCHEM_REPOSITORY_PATH}/src
make ${JFLAG} install
}
# yaml
get_yaml (){
echo "get gtest:"
if [ -d "${YAML_REPOSITORY_PATH}" ] && [ "$(ls -A ${YAML_REPOSITORY_PATH})" ]; then
  echo "${YAML_REPOSITORY_PATH} exists and is not empty ... aborting clone"; return
fi
git clone -b yaml-cpp-0.6.3 https://github.com/jbeder/yaml-cpp.git ${YAML_REPOSITORY_PATH}
}

build_install_yaml(){
echo "Building gtest:"
mkdir ${YAML_BUILD_PATH}
mkdir ${YAML_INSTALL_PATH}
cd ${YAML_BUILD_PATH}
cmake \
    -D CMAKE_INSTALL_PREFIX="${YAML_INSTALL_PATH}" \
    -D CMAKE_CXX_COMPILER="${MY_CXX}"  \
    -D CMAKE_C_COMPILER="${MY_CC}" \
    -D CMAKE_CXX_FLAGS="-g -c" \
    -D CMAKE_EXE_LINKER_FLAGS="" \
    -D CMAKE_BUILD_TYPE=RELEASE \
    ${YAML_REPOSITORY_PATH}
make ${JFLAG} install
}
#=======================================================================================
# main


mkdir -p ${REPO_BASE}
mkdir -p ${BUILD_BASE}
mkdir -p ${INSTALL_BASE}

OPENBLAS_REPOSITORY_PATH=${REPO_BASE}/openblas
OPENBLAS_INSTALL_PATH=${INSTALL_BASE}/openblas
get_openblas
#clean_openblas
build_openblas
install_openblas
#
KOKKOS_REPOSITORY_PATH=${REPO_BASE}/kokkos
KOKKOS_BUILD_PATH=${BUILD_BASE}/kokkos
KOKKOS_INSTALL_PATH=${INSTALL_BASE}/kokkos
if [ "${CUDA}" = "ON" ]; then
    KOKKOS_CXX_COMPILER="${KOKKOS_INSTALL_PATH}/bin/nvcc_wrapper"
    KOKKOS_CXX_REPO_COMPILER="${KOKKOS_REPOSITORY_PATH}/bin/nvcc_wrapper"
else
    KOKKOS_CXX_COMPILER="${MY_CXX}"
    KOKKOS_CXX_REPO_COMPILER="${MY_CXX}"
fi
get_kokkos
build_install_kokkos

GTEST_REPOSITORY_PATH=${REPO_BASE}/gtest
GTEST_BUILD_PATH=${BUILD_BASE}/gtest
GTEST_INSTALL_PATH=${INSTALL_BASE}/gtest
get_gtest
build_install_gtest

YAML_REPOSITORY_PATH=${REPO_BASE}/yaml
YAML_BUILD_PATH=${BUILD_BASE}/yaml
YAML_INSTALL_PATH=${INSTALL_BASE}/yaml
get_yaml
build_install_yaml

TINES_REPOSITORY_PATH=${REPO_BASE}/tines
TINES_BUILD_PATH=${BUILD_BASE}/tines
TINES_INSTALL_PATH=${INSTALL_BASE}/tines
get_tines
build_install_tines

TCHEM_REPOSITORY_PATH=${REPO_BASE}/tchem
TCHEM_BUILD_PATH=${BUILD_BASE}/tchem
TCHEM_INSTALL_PATH=${INSTALL_BASE}/tchem
get_tchem
build_install_tchem

exit
